# finetune.py

import argparse
import os
import random
from pathlib import Path
import json
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchaudio
import numpy as np # Added numpy

# Import necessary components from dia
from dia.model import Dia, DiaConfig
from dia.layers import DiaModel # Import DiaModel directly for type hints
from dia.pruning_utils import apply_unstructured_pruning, apply_structured_pruning, make_pruning_permanent, check_pruning_sparsity
from dia.audio import build_delay_indices, apply_audio_delay # Needed for target generation
from dia.state import EncoderInferenceState, DecoderInferenceState, KVCache # Needed for forward pass

# Scheduler imports
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, StepLR # Added schedulers

# PEFT/LoRA integration (requires `pip install peft`)
try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT library not found. LoRA fine-tuning will be disabled. Install with: pip install peft")


# --- Dataset Definition ---
class FineTuneDataset(Dataset):
    def __init__(self, metadata_path: str, audio_dir: str, config: DiaConfig, dac_model: nn.Module, target_sr: int = 44100):
        self.audio_dir = Path(audio_dir)
        self.config = config
        self.target_sr = target_sr
        self.metadata = self._load_metadata(metadata_path)
        self.dac_model = dac_model # Receive loaded DAC model
        self.device = self.dac_model.device # Use DAC model's device for consistency

        if not self.dac_model:
             raise ValueError("DAC model instance must be provided to FineTuneDataset.")

        # Precompute delay indices for target generation (assuming fixed max length)
        self.delay_precomp = build_delay_indices(
            B=1, # Process one sample at a time
            T=self.config.data.audio_length + 1, # Need T+1 for shifting
            C=self.config.data.channels,
            delay_pattern=self.config.data.delay_pattern,
        )


    def _load_metadata(self, metadata_path: str):
        data = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|', 1)
                if len(parts) == 2:
                    wav_name, text = parts
                    audio_path = self.audio_dir / wav_name
                    if audio_path.exists(): # Only add if audio file exists
                        data.append({"audio_path": audio_path, "text": text})
                    else:
                        print(f"Warning: Audio file not found for metadata entry: {wav_name}. Skipping.")
        print(f"Loaded {len(data)} valid metadata entries.")
        return data

    def __len__(self):
        return len(self.metadata)

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encodes text prompt, pads."""
        text_pad_value = self.config.data.text_pad_value
        max_len = self.config.data.text_length

        byte_text = text.encode("utf-8")
        replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
        text_tokens = list(replaced_bytes)

        current_len = len(text_tokens)
        if current_len > max_len:
            text_tokens = text_tokens[:max_len]
            current_len = max_len

        src_tokens = torch.full((max_len,), fill_value=text_pad_value, dtype=torch.long)
        src_tokens[:current_len] = torch.tensor(text_tokens, dtype=torch.long)
        return src_tokens

    def _encode_audio(self, audio_path: Path) -> torch.Tensor | None:
        """Loads audio, resamples, and encodes using DAC. Returns [T_codes, C]."""
        try:
            audio, sr = torchaudio.load(audio_path) # C, T
            if audio.shape[0] > 1: audio = torch.mean(audio, dim=0, keepdim=True) # Mono
            if sr != self.target_sr:
                audio = torchaudio.functional.resample(audio, sr, self.target_sr)

            audio = audio.unsqueeze(0).to(self.device) # B, C, T
            with torch.no_grad(): # Ensure no grads during encoding
                audio_data = self.dac_model.preprocess(audio, self.target_sr)
                _, encoded_frame, _, _, _ = self.dac_model.encode(audio_data) # B, C, T_codes

            return encoded_frame.squeeze(0).transpose(0, 1) # T_codes, C
        except Exception as e:
            print(f"Error encoding audio {audio_path}: {e}")
            return None

    def _prepare_targets(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """Creates target tokens by applying delay pattern (like inference prefill)."""
        # audio_tokens shape: [T_orig, C]
        T_orig, C = audio_tokens.shape
        bos_token = torch.full((1, C), fill_value=self.config.data.audio_bos_value, dtype=torch.long, device=self.device)
        # Prepend BOS
        tokens_with_bos = torch.cat([bos_token, audio_tokens], dim=0) # [T_orig+1, C]

        # Pad to max_length + 1 if needed for delay application
        T_needed = self.config.data.audio_length + 1
        if tokens_with_bos.shape[0] < T_needed:
            padding = torch.full((T_needed - tokens_with_bos.shape[0], C),
                                 fill_value=self.config.data.audio_pad_value, dtype=torch.long, device=self.device)
            tokens_padded = torch.cat([tokens_with_bos, padding], dim=0)
        else:
            tokens_padded = tokens_with_bos[:T_needed, :] # Truncate if too long

        # Apply delay pattern
        # apply_audio_delay expects [B, T, C]
        delayed_tokens = apply_audio_delay(
            audio_BxTxC=tokens_padded.unsqueeze(0),
            pad_value=self.config.data.audio_pad_value,
            bos_value=self.config.data.audio_bos_value,
            precomp=self.delay_precomp,
        ).squeeze(0) # [T_needed, C]

        # Targets are the delayed tokens up to the original length
        # Input for step t should predict target[t]
        # Target sequence is the delayed sequence
        return delayed_tokens[:self.config.data.audio_length, :] # Return shape [T_max, C]


    def __getitem__(self, idx):
        item = self.metadata[idx]
        text_tensor = self._encode_text(item["text"]) # [T_text]
        audio_tensor_encoded = self._encode_audio(item["audio_path"]) # [T_codes, C]

        if audio_tensor_encoded is None:
            print(f"Warning: Failed to process audio for index {idx}. Returning None.")
            # Need to handle this in collate_fn or skip
            return None # Signal error

        # Pad/Truncate encoded audio to fixed length T_max
        current_audio_len = audio_tensor_encoded.shape[0]
        max_audio_len = self.config.data.audio_length
        if current_audio_len < max_audio_len:
            padding = torch.full((max_audio_len - current_audio_len, self.config.data.channels),
                                 fill_value=self.config.data.audio_pad_value, dtype=torch.long, device=self.device)
            audio_tensor_padded = torch.cat([audio_tensor_encoded, padding], dim=0)
        else:
            audio_tensor_padded = audio_tensor_encoded[:max_audio_len, :] # [T_max, C]

        # Create target tokens by applying delay pattern
        target_tensor = self._prepare_targets(audio_tensor_padded) # [T_max, C]

        # The input to the decoder during training is typically the target sequence shifted right
        # Input[t] = Target[t-1], with Input[0] = BOS
        # Our target_tensor already incorporates the delay, which acts like a shift.
        # Let's use the `target_tensor` as the input to the decoder embeddings.
        # The actual target for the loss function will be the original `audio_tensor_padded`.
        # --> Correction: The standard is input[t] predicts target[t].
        # So, decoder_input = BOS + audio_tokens[:-1] (conceptually)
        # And target = audio_tokens
        # The delay pattern complicates this. Let's assume the `target_tensor` (delayed)
        # is what the model *predicts*. The input that *causes* this prediction
        # should be the appropriately shifted version.
        # Let's try using `target_tensor` as the *target* for the loss,
        # and derive the decoder input from it.
        # Decoder input at step t should be the token that *would have been* generated at t-1.
        # This is complex with the delay. Let's stick to the simpler approach for now:
        # Decoder Input = BOS + audio_tokens_padded[:-1] (conceptually, needs delay)
        # Target = audio_tokens_padded

        # Let's prepare decoder input: Prepend BOS, take up to T_max-1, pad back to T_max
        bos_token = torch.full((1, self.config.data.channels), fill_value=self.config.data.audio_bos_value, dtype=torch.long, device=self.device)
        decoder_input_unpadded = torch.cat([bos_token, audio_tensor_padded[:-1, :]], dim=0) # [T_max, C]
        # Apply delay pattern to the decoder input sequence
        decoder_input_padded = torch.cat([decoder_input_unpadded, torch.full((1, self.config.data.channels), fill_value=self.config.data.audio_pad_value, dtype=torch.long, device=self.device)], dim=0) # Pad to T_max+1 for delay
        decoder_input_delayed = apply_audio_delay(
             decoder_input_padded.unsqueeze(0),
             pad_value=self.config.data.audio_pad_value,
             bos_value=self.config.data.audio_bos_value,
             precomp=self.delay_precomp,
        ).squeeze(0)[:max_audio_len, :] # [T_max, C]


        return {
            "text_input_ids": text_tensor, # [T_text]
            "decoder_input_ids": decoder_input_delayed, # [T_max, C] - Delayed version of BOS + audio[:-1]
            "target_ids": audio_tensor_padded, # [T_max, C] - Original audio tokens (no delay/BOS) used as target for loss
            "attention_mask": (text_tensor != self.config.data.text_pad_value), # [T_text] - For encoder mask
            "decoder_attention_mask": (audio_tensor_padded != self.config.data.audio_pad_value) # [T_max, C] - Mask for loss calculation
        }

# --- Collate Function ---
def safe_collate_fn(batch):
    # Filter out None items (resulting from dataset errors)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if the whole batch failed

    # Collate valid items
    text_batch = torch.stack([item["text_input_ids"] for item in batch])
    decoder_input_batch = torch.stack([item["decoder_input_ids"] for item in batch])
    target_batch = torch.stack([item["target_ids"] for item in batch])
    attn_mask_batch = torch.stack([item["attention_mask"] for item in batch])
    decoder_attn_mask_batch = torch.stack([item["decoder_attention_mask"] for item in batch])


    return {
        "text_input_ids": text_batch,
        "decoder_input_ids": decoder_input_batch,
        "target_ids": target_batch,
        "attention_mask": attn_mask_batch,
        "decoder_attention_mask": decoder_attn_mask_batch,
    }


# --- LR Scheduler Helper ---
def get_scheduler(optimizer, scheduler_type, num_training_steps, warmup_steps):
    if scheduler_type == "linear":
        # Basic linear decay from initial lr to 0
        lr_lambda = lambda step: max(0.0, 1.0 - step / num_training_steps)
        return LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=0)
    elif scheduler_type == "step":
        # Example: decay LR by 0.1 every 1/3 of total steps
        return StepLR(optimizer, step_size=num_training_steps // 3, gamma=0.1)
    elif scheduler_type == "constant":
        return LambdaLR(optimizer, lambda step: 1.0) # No change
    # Add warmup: Wrap the chosen scheduler
    # Basic linear warmup
    if warmup_steps > 0:
        base_scheduler = get_scheduler(optimizer, scheduler_type, num_training_steps, 0) # Get scheduler without warmup first
        def warmup_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            # After warmup, use the base scheduler's logic
            # This requires knowing the base scheduler's behavior, complex to generalize here
            # Simpler: Use constant LR after warmup for this example
            # Or use transformers library scheduler which handles this better
            return 1.0 # Placeholder: Constant LR after warmup

        # For simplicity, let's just return the base scheduler without integrated warmup here
        # Proper warmup often needs libraries like `transformers.get_linear_schedule_with_warmup`
        print("Warning: Basic scheduler selected. For integrated warmup, consider using transformers library schedulers.")
        return base_scheduler
    else:
        return get_scheduler(optimizer, scheduler_type, num_training_steps, 0)


# --- Main Fine-tuning Logic ---
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Dia TTS Model")
    # Model & Data Args
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pretrained Dia model directory (containing config.json and pytorch_model.bin) or HF repo ID.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to the generated dataset directory (containing wavs/ and metadata.csv).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save fine-tuned checkpoints and logs.")

    # Pruning Args
    parser.add_argument("--prune-mode", type=str, default="none", choices=["none", "unstructured", "structured"], help="Pruning mode before fine-tuning.")
    parser.add_argument("--prune-amount", type=float, default=0.5, help="Fraction of weights/structures to prune (0.0-1.0).")
    parser.add_argument("--prune-dim", type=int, default=0, help="Dimension for structured pruning (e.g., 0 for output channels/neurons).")
    parser.add_argument("--prune-norm", type=int, default=2, help="Norm (1 or 2) for structured pruning.")

    # Fine-tuning Args
    parser.add_argument("--adapter-mode", type=str, default="none", choices=["none", "lora"], help="Adapter mode for fine-tuning.")
    parser.add_argument("--lora-rank", type=int, default=8, help="Rank for LoRA adapters.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="Alpha for LoRA adapters.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="Dropout for LoRA adapters.")
    parser.add_argument("--lora-target-modules", nargs='+', default=["q_proj", "v_proj"], help="Modules to apply LoRA to (e.g., q_proj k_proj v_proj o_proj).")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Steps for gradient accumulation.")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine", choices=["linear", "cosine", "step", "constant"], help="Learning rate scheduler type.")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Number of warmup steps for LR scheduler (basic implementation).")
    parser.add_argument("--logging-steps", type=int, default=50, help="Log training loss every N steps.")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping.")


    # Infrastructure Args
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, cpu, mps). Auto-detects if None.")
    parser.add_argument("--compute-dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Compute dtype for training.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")

    args = parser.parse_args()

    # --- Setup ---
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Warning: MPS support is experimental for training.")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_path / "training_args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Determine compute dtype
    compute_dtype_torch = torch.float32
    if args.compute_dtype == "float16":
        compute_dtype_torch = torch.float16
    elif args.compute_dtype == "bfloat16":
        compute_dtype_torch = torch.bfloat16

    if device.type == 'cpu' and compute_dtype_torch != torch.float32:
        print("Warning: CPU device selected, forcing compute_dtype to float32.")
        compute_dtype_torch = torch.float32
    elif device.type == 'mps' and compute_dtype_torch == torch.bfloat16:
        print("Warning: MPS device does not support bfloat16, forcing compute_dtype to float16.")
        compute_dtype_torch = torch.float16


    # --- Load Model ---
    print(f"Loading base model from {args.model_path}...")
    try:
        # Use the Dia class loader which handles config loading
        dia_model_wrapper = Dia.from_pretrained(args.model_path, compute_dtype=args.compute_dtype, device=device)
        model = dia_model_wrapper.model # Get the underlying DiaModel (nn.Module)
        config = dia_model_wrapper.config
        dac_model = dia_model_wrapper.dac_model # Get the loaded DAC model
        print("Base model and DAC loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Apply Pruning (Before LoRA) ---
    if args.prune_mode == "unstructured":
        print(f"Applying UNSTRUCTURED pruning (amount={args.prune_amount})...")
        # Prune float32 model then cast? Or prune directly? Prune float32 for stability.
        model.float()
        apply_unstructured_pruning(model, args.prune_amount)
        model.to(dtype=compute_dtype_torch) # Cast back
        check_pruning_sparsity(model)
    elif args.prune_mode == "structured":
        print(f"Applying STRUCTURED pruning (amount={args.prune_amount}, dim={args.prune_dim}, norm={args.prune_norm})...")
        model.float()
        apply_structured_pruning(model, args.prune_amount, dim=args.prune_dim, n=args.prune_norm)
        model.to(dtype=compute_dtype_torch) # Cast back
        check_pruning_sparsity(model)

    # --- Apply LoRA / PEFT ---
    if args.adapter_mode == "lora":
        if not PEFT_AVAILABLE:
            print("Error: PEFT library not found, cannot apply LoRA. Exiting.")
            return
        print("Applying LoRA adapters...")
        # Optional: Prepare model for k-bit training if using quantization (e.g., bitsandbytes)
        # model = prepare_model_for_kbit_training(model) # Uncomment if using k-bit

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none", # Common practice
            # Task type might need adjustment for TTS, using placeholder
            task_type=TaskType.CAUSAL_LM, # Or maybe SEQ_2_SEQ_LM? Needs verification.
            # modules_to_save = ["logits_dense"] # Example: if you want to train output layer too
        )
        # Ensure model is on the correct device before applying PEFT
        model = get_peft_model(model.to(device), lora_config)
        print("LoRA adapters applied.")
        model.print_trainable_parameters()
    else:
        print("Fine-tuning mode: Full model (or remaining weights after pruning).")
        model.to(device) # Ensure model is on device

    # --- Setup DataLoader, Optimizer, Scheduler ---
    print("Setting up dataset and dataloader...")
    try:
        dataset = FineTuneDataset(
            metadata_path=Path(args.dataset_dir) / "metadata.csv",
            audio_dir=Path(args.dataset_dir) / "wavs",
            config=config,
            dac_model=dac_model # Pass the loaded DAC model
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=safe_collate_fn, # Use safe collate
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Filter parameters for optimizer
    if args.adapter_mode == "lora":
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"Optimizing {len(trainable_params)} LoRA/trainable parameters.")
    else:
        # If pruned, optimize all remaining non-frozen parameters.
        # Pruning utilities don't freeze, just zero out weights.
        trainable_params = model.parameters()
        num_total_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)
        print(f"Optimizing {num_trainable}/{num_total_params} model parameters.")


    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Setup Scheduler
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_training_steps = args.epochs * num_update_steps_per_epoch
    num_warmup_steps = args.warmup_steps

    scheduler = get_scheduler(optimizer, args.lr_scheduler_type, num_training_steps, num_warmup_steps)
    print(f"Scheduler: {args.lr_scheduler_type}, Total steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")

    # --- Loss Function ---
    # Cross Entropy Loss, ignore padding index in targets
    # Target shape: [B, T_max, C], Logits shape: [B, T_max, C, V]
    # Reshape for loss: Logits -> [B*T_max*C, V], Target -> [B*T_max*C]
    loss_fct = nn.CrossEntropyLoss(ignore_index=config.data.audio_pad_value)


    # --- Training Loop ---
    print("Starting fine-tuning...")
    global_step = 0
    model.train() # Ensure model is in training mode

    for epoch in range(args.epochs):
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        epoch_loss = 0.0
        steps_in_epoch = 0

        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):

            if batch is None: # Skip batch if collate_fn returned None
                 print(f"Warning: Skipping empty batch at step {step}.")
                 continue

            # Move batch to device
            text_inputs = batch["text_input_ids"].to(device) # [B, T_text]
            decoder_inputs = batch["decoder_input_ids"].to(device) # [B, T_max, C]
            targets = batch["target_ids"].to(device) # [B, T_max, C]
            encoder_mask = batch["attention_mask"].to(device) # [B, T_text]
            decoder_target_mask = batch["decoder_attention_mask"].to(device) # [B, T_max, C]

            # --- Forward Pass ---
            # Use autocast for mixed precision training
            with torch.amp.autocast(device_type=device.type, dtype=compute_dtype_torch, enabled=device.type != 'cpu'):
                # 1. Encoder Pass
                # Create encoder state dynamically for the batch
                enc_state = EncoderInferenceState(
                    max_seq_len=config.data.text_length,
                    device=device,
                    positions=torch.arange(config.data.text_length, device=device).unsqueeze(0).expand(text_inputs.shape[0], -1),
                    padding_mask=encoder_mask,
                    attn_mask=create_attn_mask(encoder_mask, encoder_mask, device, is_causal=False)
                )
                encoder_out = model.encoder(text_inputs, enc_state) # [B, T_text, D_enc]

                # 2. Prepare Decoder State for Full Sequence
                # Precompute cross-attn cache (static for the batch)
                cross_attn_cache = model.decoder.precompute_cross_attn_cache(encoder_out, enc_state.positions)
                # Create decoder state for the full sequence length T_max
                dec_state = DecoderInferenceState(
                    device=device,
                    dtype=compute_dtype_torch,
                    enc_out=encoder_out,
                    enc_positions=enc_state.positions,
                    enc_padding_mask=enc_state.padding_mask,
                    self_attn_cache=[ # Initialize empty self-attn caches for training pass
                        KVCache(
                            num_heads=config.model.decoder.kv_heads, max_len=config.data.audio_length,
                            head_dim=config.model.decoder.gqa_head_dim, dtype=compute_dtype_torch,
                            device=device, batch_size=text_inputs.shape[0]
                        ) for _ in range(config.model.decoder.n_layer)
                    ],
                    cross_attn_cache=cross_attn_cache
                )
                # Set positions and masks for the full decoder sequence length
                dec_state.prepare_step(0, config.data.audio_length)

                # 3. Decoder Pass (Teacher Forcing)
                # Input to decoder is `decoder_inputs` [B, T_max, C]
                logits = model.decoder(decoder_inputs, dec_state) # [B, T_max, C, V]

                # 4. Calculate Loss
                # Reshape for CrossEntropyLoss: [B*T_max*C, V] and [B*T_max*C]
                logits_flat = logits.view(-1, logits.size(-1)) # [B*T_max*C, V]
                targets_flat = targets.view(-1) # [B*T_max*C]

                # Apply mask to targets (set padding tokens to ignore_index)
                # decoder_target_mask is [B, T_max, C], needs to be [B*T_max*C]
                mask_flat = decoder_target_mask.view(-1)
                targets_flat_masked = targets_flat.masked_fill(~mask_flat, loss_fct.ignore_index)

                loss = loss_fct(logits_flat, targets_flat_masked)


            if torch.isnan(loss):
                 print(f"Warning: NaN loss detected at step {global_step}. Skipping step.")
                 optimizer.zero_grad()
                 continue

            loss = loss / args.gradient_accumulation_steps
            epoch_loss += loss.item() * args.gradient_accumulation_steps # De-scale for logging
            steps_in_epoch += 1

            # --- Backward Pass & Optimization ---
            # TODO: Add gradient scaling if using mixed precision manually
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Gradient Clipping
                if args.max_grad_norm > 0:
                     # If using PEFT, unscale before clipping? Check PEFT docs.
                     # Clip norms for non-PEFT or after potential unscaling
                     torch.nn.utils.clip_grad_norm_(
                          model.parameters() if args.adapter_mode == 'none' else trainable_params, # Clip only trainable if LoRA
                          args.max_grad_norm
                     )

                optimizer.step()
                scheduler.step() # Step the scheduler
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss_interval = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0
                    print(f"Step: {global_step}, Epoch: {epoch+1}, LR: {scheduler.get_last_lr()[0]:.2e}, Loss: {avg_loss_interval:.4f}")
                    # Reset interval accumulators if needed, or keep running epoch loss

                # Saving Checkpoint
                if global_step % args.save_steps == 0:
                    print(f"Saving checkpoint at step {global_step}...")
                    save_path = output_path / f"checkpoint-{global_step}"
                    save_path.mkdir(exist_ok=True)

                    # Determine what to save based on mode
                    model_to_save = model.module if hasattr(model, 'module') else model # Handle DDP wrapping

                    if args.adapter_mode == "lora":
                        # Save LoRA adapters only
                        model_to_save.save_pretrained(str(save_path))
                        print(f"LoRA adapter saved to {save_path}")
                    else:
                        # Save full model state dict (including pruned weights)
                        torch.save(model_to_save.state_dict(), save_path / "pytorch_model.bin")
                        # Also save config
                        config.save(str(save_path / "config.json"))
                        print(f"Full model checkpoint saved to {save_path}")

        # End of Epoch
        avg_epoch_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0
        print(f"--- End of Epoch {epoch+1} --- Average Loss: {avg_epoch_loss:.4f} ---")


    # --- Final Save ---
    print("Training finished. Saving final model...")
    final_save_path = output_path / "final_model"
    final_save_path.mkdir(exist_ok=True)

    model_to_save = model.module if hasattr(model, 'module') else model # Handle DDP

    if args.prune_mode != "none":
        # Ensure model is on CPU and in float32 for making pruning permanent
        model_to_save.cpu().float()
        make_pruning_permanent(model_to_save)
        print("Pruning made permanent.")
        # Move back to original device/dtype if needed for saving? Or save CPU version? Save CPU float32.
        model_to_save.to(dtype=torch.float32) # Ensure saved state_dict is float32

    if args.adapter_mode == "lora":
        # Save final adapter
        model_to_save.save_pretrained(str(final_save_path))
        # Also save the base model config for reference
        try:
            # If PEFT modified the config, save the original one
            base_config = model_to_save.peft_config['default'].base_model_config
            # Need to instantiate DiaConfig from the base config dict
            DiaConfig.model_validate(base_config.to_dict()).save(str(final_save_path / "base_config.json"))
        except Exception as e:
             print(f"Could not save base config from PEFT model: {e}. Saving current config.")
             config.save(str(final_save_path / "base_config.json")) # Fallback
        print(f"Final LoRA adapter saved to {final_save_path}")
        print("To use: Load the base model, then load adapters using `PeftModel.from_pretrained(base_model, adapter_path)`")
    else:
        # Save the potentially pruned full model state dict (as float32 on CPU)
        torch.save(model_to_save.state_dict(), final_save_path / "pytorch_model.bin")
        config.save(str(final_save_path / "config.json"))
        print(f"Final full model saved to {final_save_path}")

if __name__ == "__main__":
    main()