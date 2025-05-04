# dia/model.py

import time
from enum import Enum
from typing import Optional
import os
from pathlib import Path

import dac
import numpy as np
import torch
import torchaudio
import soundfile as sf # Use soundfile for saving

from .audio import apply_audio_delay, build_delay_indices, build_revert_indices, decode, revert_audio_delay
from .config import DiaConfig
from .layers import DiaModel
from .state import DecoderInferenceState, DecoderOutput, EncoderInferenceState


DEFAULT_SAMPLE_RATE = 44100


def _get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sample_next_token(
    logits_BCxV: torch.Tensor,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int | None = None,
) -> torch.Tensor:
    if temperature == 0.0:
        # Argmax sampling
        return torch.argmax(logits_BCxV, dim=-1)

    # Temperature scaling
    logits_BCxV = logits_BCxV / temperature

    # Optional Top-K filtering for CFG
    if cfg_filter_top_k is not None and cfg_filter_top_k > 0:
        top_k_values, _ = torch.topk(logits_BCxV, k=cfg_filter_top_k, dim=-1)
        # Get the value of the k-th element
        kth_value = top_k_values[..., -1].unsqueeze(-1)
        # Mask elements smaller than the k-th value
        mask = logits_BCxV < kth_value
        logits_BCxV = logits_BCxV.masked_fill(mask, -torch.inf)


    # Top-P (Nucleus) sampling
    if top_p < 1.0:
        probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
        sorted_probs_BCxV, sorted_indices_BCxV = torch.sort(probs_BCxV, dim=-1, descending=True)
        cumulative_probs_BCxV = torch.cumsum(sorted_probs_BCxV, dim=-1)

        # Create mask for tokens to remove (cumulative prob > top_p)
        sorted_indices_to_remove_BCxV = cumulative_probs_BCxV > top_p
        # Shift mask to keep the first token that exceeds top_p
        sorted_indices_to_remove_BCxV = torch.roll(sorted_indices_to_remove_BCxV, shifts=1, dims=-1)
        sorted_indices_to_remove_BCxV[..., 0] = False # Always keep the highest probability token

        # Scatter mask back to original indices
        indices_to_remove_BCxV = torch.zeros_like(sorted_indices_to_remove_BCxV)
        indices_to_remove_BCxV = indices_to_remove_BCxV.scatter(dim=-1, index=sorted_indices_BCxV, src=sorted_indices_to_remove_BCxV)
        logits_BCxV = logits_BCxV.masked_fill(indices_to_remove_BCxV, -torch.inf)

    # Sample from the filtered distribution
    final_probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
    # Handle potential numerical issues where all probabilities become zero
    # If sum is zero, fall back to uniform distribution over valid tokens or argmax
    if torch.all(torch.isclose(final_probs_BCxV.sum(dim=-1), torch.tensor(0.0))):
         print("Warning: All probabilities became zero after filtering. Falling back to argmax.")
         return torch.argmax(logits_BCxV, dim=-1) # Fallback to argmax on the filtered logits

    sampled_indices_BC = torch.multinomial(final_probs_BCxV, num_samples=1)
    sampled_indices_C = sampled_indices_BC.squeeze(-1)
    return sampled_indices_C


class ComputeDtype(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    def to_dtype(self) -> torch.dtype:
        if self == ComputeDtype.FLOAT32:
            return torch.float32
        elif self == ComputeDtype.FLOAT16:
            return torch.float16
        elif self == ComputeDtype.BFLOAT16:
            return torch.bfloat16
        else:
            raise ValueError(f"Unsupported compute dtype: {self}")


class Dia:
    def __init__(
        self,
        config: DiaConfig,
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
    ):
        """Initializes the Dia model.

        Args:
            config: The configuration object for the model.
            compute_dtype: The computation dtype ('float32', 'float16', 'bfloat16').
            device: The device to load the model onto. If None, will automatically select the best available device.

        Raises:
            RuntimeError: If there is an error loading the DAC model.
        """
        super().__init__()
        self.config = config
        self.device = device if device is not None else _get_default_device()
        if isinstance(compute_dtype, str):
            compute_dtype = ComputeDtype(compute_dtype)
        self.compute_dtype = compute_dtype.to_dtype()
        # Ensure compute_dtype is compatible with device
        if self.device.type == 'cpu' and self.compute_dtype != torch.float32:
            print(f"Warning: CPU device selected, overriding compute_dtype to float32 (was {compute_dtype.value}).")
            self.compute_dtype = torch.float32
        elif self.device.type == 'mps' and self.compute_dtype == torch.bfloat16:
             print(f"Warning: MPS device does not support bfloat16, overriding compute_dtype to float16.")
             self.compute_dtype = torch.float16


        self.model: DiaModel = DiaModel(config, self.compute_dtype)
        self.dac_model = None

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

    @classmethod
    def from_local(
        cls,
        config_path: str,
        checkpoint_path: str,
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
    ) -> "Dia":
        """Loads the Dia model from local configuration and checkpoint files.

        Args:
            config_path: Path to the configuration JSON file.
            checkpoint_path: Path to the model checkpoint (.bin or .pth) file.
            compute_dtype: The computation dtype ('float32', 'float16', 'bfloat16').
            device: The device to load the model onto. If None, will automatically select the best available device.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If the config or checkpoint file is not found.
            RuntimeError: If there is an error loading the checkpoint.
        """
        config = DiaConfig.load(config_path)
        if config is None:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        dia = cls(config, compute_dtype, device)

        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu") # Load to CPU first
            # Handle potential PEFT adapter keys if loading a merged model accidentally
            # (This is basic, a more robust solution might be needed)
            filtered_state_dict = {k: v for k, v in state_dict.items() if 'lora_' not in k}
            missing_keys, unexpected_keys = dia.model.load_state_dict(filtered_state_dict, strict=False)
            if unexpected_keys:
                 print(f"Warning: Unexpected keys found in checkpoint: {unexpected_keys}")
            if missing_keys:
                 print(f"Warning: Missing keys in checkpoint: {missing_keys}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint from {checkpoint_path}") from e

        dia.model.to(dia.device) # Move model to target device
        dia.model.eval()
        dia._load_dac_model()
        return dia

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "nari-labs/Dia-1.6B",
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
        **kwargs, # Allow passing kwargs to underlying from_pretrained
    ) -> "Dia":
        """Loads the Dia model from a Hugging Face Hub repository or local path.

        Downloads the configuration and checkpoint files from the specified
        repository ID or loads from a local directory containing config.json and pytorch_model.bin.

        Args:
            model_name: The Hugging Face Hub repository ID or local directory path.
            compute_dtype: The computation dtype ('float32', 'float16', 'bfloat16').
            device: The device to load the model onto. If None, will automatically select the best available device.
            **kwargs: Additional arguments passed to `DiaModel.from_pretrained`.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If config or checkpoint download/loading fails.
            RuntimeError: If there is an error loading the checkpoint.
        """
        if isinstance(compute_dtype, str):
            compute_dtype = ComputeDtype(compute_dtype)

        # Determine device early to pass to cls constructor
        resolved_device = device if device is not None else _get_default_device()

        # Load the underlying nn.Module (DiaModel)
        # Pass compute_dtype explicitly if supported by the underlying loader
        # Note: DiaModel's from_pretrained doesn't explicitly take compute_dtype yet,
        # it's handled in the Dia wrapper's __init__.
        loaded_model = DiaModel.from_pretrained(model_name, **kwargs)
        config = loaded_model.config # Get config from loaded model

        # Create the Dia wrapper instance
        dia = cls(config, compute_dtype, resolved_device)

        # Assign the loaded nn.Module and move to device
        dia.model = loaded_model
        dia.model.to(dia.device)
        dia.model.eval()
        dia._load_dac_model() # Load DAC after model is on device
        return dia

    def _load_dac_model(self):
        """Loads the Descript Audio Codec model."""
        try:
            print("Loading DAC model...")
            # Ensure cache directory exists and is writable if needed
            # dac_cache_dir = Path.home() / ".cache" / "dac"
            # dac_cache_dir.mkdir(parents=True, exist_ok=True)
            dac_model_path = dac.utils.download() # Let DAC handle download/cache
            # Load DAC model to the same device as the main model
            dac_model = dac.DAC.load(dac_model_path).to(self.device)
            dac_model.eval() # Set DAC to eval mode
            print("DAC model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load DAC model: {e}") from e
        self.dac_model = dac_model

    def _prepare_text_input(self, text: str) -> torch.Tensor:
        """Encodes text prompt, pads, and creates attention mask and positions."""
        text_pad_value = self.config.data.text_pad_value
        max_len = self.config.data.text_length

        # Simple byte encoding, replace special tokens
        try:
            byte_text = text.encode("utf-8")
            # Replace S1/S2 tags with single low-value bytes (e.g., 1 and 2)
            # Ensure these bytes don't conflict with actual UTF-8 bytes if possible
            # Using values outside typical ASCII/UTF-8 multi-byte ranges
            replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
            text_tokens = list(replaced_bytes) # List of integer byte values
        except Exception as e:
            print(f"Error encoding text: {e}")
            # Fallback or raise error
            text_tokens = [ord('?')] * len(text) # Simple fallback

        current_len = len(text_tokens)
        if current_len > max_len:
            print(f"Warning: Input text truncated from {current_len} to {max_len} bytes.")
            text_tokens = text_tokens[:max_len]
            current_len = max_len

        src_tokens = torch.full(
            (1, max_len),
            fill_value=text_pad_value,
            dtype=torch.long,
            device=self.device,
        )
        src_tokens[0, :current_len] = torch.tensor(
            text_tokens,
            dtype=torch.long,
            device=self.device,
        )
        return src_tokens

    def _prepare_audio_prompt(self, audio_prompt: torch.Tensor | None) -> tuple[torch.Tensor, int]:
        """Prepares audio prompt tensor with BOS and delay padding."""
        num_channels = self.config.data.channels
        audio_bos_value = self.config.data.audio_bos_value
        audio_pad_value = self.config.data.audio_pad_value
        delay_pattern = self.config.data.delay_pattern
        max_delay_pattern = max(delay_pattern) if delay_pattern else 0

        # Start with BOS token(s)
        parts = [
            torch.full(
                (1, num_channels), # Single BOS step
                fill_value=audio_bos_value,
                dtype=torch.int, # Use int, will be cast later if needed
                device=self.device,
            )
        ]
        prefill_step = 1 # Account for BOS

        # Append actual audio prompt if provided
        if audio_prompt is not None:
            if audio_prompt.ndim == 2: # T, C
                 audio_prompt = audio_prompt.to(self.device)
            elif audio_prompt.ndim == 3 and audio_prompt.shape[0] == 1: # B, T, C
                 audio_prompt = audio_prompt.squeeze(0).to(self.device)
            else:
                 raise ValueError(f"Unexpected audio_prompt shape: {audio_prompt.shape}. Expected [T, C] or [1, T, C].")

            prefill_step += audio_prompt.shape[0]
            parts.append(audio_prompt)

        # Concatenate BOS and audio prompt
        prefill_no_delay = torch.cat(parts, dim=0) # Shape [T_total, C]

        # Add padding required for the maximum delay
        # The amount of padding needed depends on the *final* length after delay is applied
        # We need enough padding so that accessing `t + max_delay` is valid
        # Let T = prefill_no_delay.shape[0]. The delayed tensor needs indices up to T + max_delay - 1.
        # So, the input tensor needs to have length T + max_delay.
        delay_pad_tensor = torch.full(
            (max_delay_pattern, num_channels), fill_value=audio_pad_value, dtype=torch.int, device=self.device
        )
        prefill_padded = torch.cat([prefill_no_delay, delay_pad_tensor], dim=0) # Shape [T_total + max_delay, C]

        # Apply delay pattern
        # The build_delay_indices expects B, T, C input
        delay_precomp = build_delay_indices(
            B=1,
            T=prefill_padded.shape[0], # Use padded length for T
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        # Apply delay expects B, T, C
        prefill_delayed = apply_audio_delay(
            audio_BxTxC=prefill_padded.unsqueeze(0),
            pad_value=audio_pad_value,
            bos_value=audio_bos_value,
            precomp=delay_precomp,
        ).squeeze(0) # Back to [T_padded, C]

        # Return the delayed tensor and the number of *actual* content steps (BOS + audio)
        return prefill_delayed, prefill_step

    def _prepare_generation(self, text: str, audio_prompt: str | torch.Tensor | None, verbose: bool):
        """Prepares encoder/decoder states and initial output buffer."""
        # Prepare text inputs (conditional and unconditional)
        enc_input_cond = self._prepare_text_input(text)
        # Unconditional input is typically all padding tokens or a zero tensor
        enc_input_uncond = torch.full_like(enc_input_cond, fill_value=self.config.data.text_pad_value)
        # Batch conditional and unconditional inputs
        enc_input = torch.cat([enc_input_uncond, enc_input_cond], dim=0) # Shape [2, T_text]

        # Prepare audio prompt (encode if path, then add BOS/padding/delay)
        if isinstance(audio_prompt, str):
            # Load and encode audio file path
            audio_prompt_tensor = self.load_audio(audio_prompt) # Returns [T_codes, C]
        elif isinstance(audio_prompt, torch.Tensor):
            audio_prompt_tensor = audio_prompt # Assume already encoded [T_codes, C]
        else:
            audio_prompt_tensor = None

        prefill_tokens_delayed, prefill_step_count = self._prepare_audio_prompt(audio_prompt_tensor)
        # prefill_tokens_delayed has shape [T_padded, C]

        if verbose:
            print(f"generate: Text tokens shape: {enc_input.shape}")
            print(f"generate: Prefill audio steps: {prefill_step_count}")
            print(f"generate: Delayed prefill tokens shape: {prefill_tokens_delayed.shape}")

        # --- Encoder Pass ---
        enc_state = EncoderInferenceState.new(self.config, enc_input) # Uses enc_input shape/device
        # Ensure model is in eval mode and correct dtype
        self.model.eval()
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type, dtype=self.compute_dtype, enabled=self.device.type != 'cpu'):
             encoder_out = self.model.encoder(enc_input, enc_state) # Shape [2, T_text, D_enc]

        if verbose:
            print(f"generate: Encoder output shape: {encoder_out.shape}")

        # --- Prepare Decoder State ---
        # Precompute cross-attention K/V cache from encoder output
        dec_cross_attn_cache = self.model.decoder.precompute_cross_attn_cache(encoder_out, enc_state.positions)
        # Initialize decoder state
        dec_state = DecoderInferenceState.new(
            self.config, enc_state, encoder_out, dec_cross_attn_cache, self.compute_dtype
        )
        # Initialize output buffer
        dec_output = DecoderOutput.new(self.config, self.device)
        # Fill the output buffer with the prepared (delayed) prefill tokens
        dec_output.prefill(prefill_tokens_delayed, prefill_step_count)

        # --- Decoder Prefill Pass (Warm-up KV Cache) ---
        # Process the actual prefill steps (excluding padding) to populate KV cache
        # The prefill_step_count includes BOS + actual audio prompt length
        if prefill_step_count > 1: # Only run if there's more than just BOS
            # We need to process steps 0 to prefill_step_count - 1
            # The corresponding tokens are already in dec_output buffer
            dec_state.prepare_step(0, prefill_step_count - 1) # Set positions for the prefill range
            # Get the delayed tokens corresponding to these steps
            tokens_for_prefill_BxTxC = dec_output.get_tokens_at(0, prefill_step_count - 1).unsqueeze(0).expand(2, -1, -1)

            if verbose:
                print(f"generate: Running decoder prefill for {prefill_step_count - 1} steps...")
                print(f"generate: Decoder prefill input tokens shape: {tokens_for_prefill_BxTxC.shape}")

            with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type, dtype=self.compute_dtype, enabled=self.device.type != 'cpu'):
                # Run decoder forward pass just to update KV caches (output logits ignored)
                _ = self.model.decoder.forward(tokens_for_prefill_BxTxC, dec_state)

            if verbose:
                print("generate: Decoder prefill complete.")

        # The next step to generate is prefill_step_count - 1
        # (e.g., if only BOS, prefill_step=1, next step is 0)
        # (e.g., if BOS + 10 audio steps, prefill_step=11, next step is 10)
        return dec_state, dec_output

    def _decoder_step(
        self,
        tokens_Bx1xC: torch.Tensor, # Input token for the current step [2, 1, C]
        dec_state: DecoderInferenceState, # Contains KV caches, encoder output etc.
        cfg_scale: float,
        temperature: float,
        top_p: float,
        cfg_filter_top_k: int,
    ) -> torch.Tensor:
        """Performs one step of decoder generation with CFG."""
        audio_eos_value = self.config.data.audio_eos_value

        # Run the decoder step
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type, dtype=self.compute_dtype, enabled=self.device.type != 'cpu'):
            logits_Bx1xCxV = self.model.decoder.decode_step(tokens_Bx1xC, dec_state)
            # logits_Bx1xCxV shape: [2, 1, C, V]

        # Extract logits for the last step
        logits_last_BxCxV = logits_Bx1xCxV[:, -1, :, :] # Shape [2, C, V]

        # Apply Classifier-Free Guidance (CFG)
        uncond_logits_CxV = logits_last_BxCxV[0] # Shape [C, V]
        cond_logits_CxV = logits_last_BxCxV[1]   # Shape [C, V]
        # CFG formula: guided_logits = uncond + scale * (cond - uncond)
        # Simplified: guided_logits = cond + (scale - 1) * (cond - uncond) -- less common
        # Common: guided_logits = cond + scale * (cond - uncond) -- let's use this
        # Or: guided_logits = (1 - scale) * uncond + scale * cond -- equivalent if scale is weight on cond
        # Using: logits = cond + scale * (cond - uncond)
        logits_CxV = cond_logits_CxV + cfg_scale * (cond_logits_CxV - uncond_logits_CxV) # Shape [C, V]

        # --- Apply constraints ---
        # 1. Prevent EOS token (except in the first channel)
        #    Make EOS probability -inf for channels C > 0
        if logits_CxV.shape[0] > 1: # If more than one channel
             logits_CxV[1:, audio_eos_value] = -torch.inf

        # 2. Prevent padding/BOS tokens from being generated
        #    Make PAD/BOS probability -inf for all channels
        #    audio_pad_value = 1025, audio_bos_value = 1026
        #    Need to check exact values from config
        pad_val = self.config.data.audio_pad_value
        bos_val = self.config.data.audio_bos_value
        logits_CxV[:, pad_val] = -torch.inf
        logits_CxV[:, bos_val] = -torch.inf

        # 3. Prevent tokens > vocab size (already handled by vocab size limit in layers)
        #    But ensure EOS+1 etc are masked if vocab size is tight
        vocab_size = self.config.model.tgt_vocab_size
        if vocab_size <= audio_eos_value + 1: # Check if indices beyond EOS exist
             logits_CxV[:, vocab_size:] = -torch.inf


        # Sample the next token for each channel
        pred_C = _sample_next_token(
            logits_CxV.to(dtype=torch.float32), # Sampling often more stable in float32
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
        ) # Returns shape [C]
        return pred_C

    def _generate_output(self, generated_codes: torch.Tensor) -> np.ndarray | None:
        """Decodes generated audio codes back to waveform using DAC."""
        if self.dac_model is None:
            raise RuntimeError("DAC model not loaded. Cannot decode audio.")
        if generated_codes is None or generated_codes.numel() == 0:
             print("Warning: No generated codes to decode.")
             return None

        num_channels = self.config.data.channels
        seq_length = generated_codes.shape[0]
        delay_pattern = self.config.data.delay_pattern
        audio_pad_value = self.config.data.audio_pad_value
        max_delay_pattern = max(delay_pattern) if delay_pattern else 0

        # Revert the delay pattern to align codes across channels
        revert_precomp = build_revert_indices(
            B=1,
            T=seq_length, # Use the actual length of generated codes
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        # Revert delay expects [B, T, C]
        codebook_aligned_BxTxC = revert_audio_delay(
            audio_BxTxC=generated_codes.unsqueeze(0),
            pad_value=audio_pad_value,
            precomp=revert_precomp,
            T=seq_length, # Original length before potential padding in revert
        ) # Shape [1, T, C]

        # Remove the padding introduced by the maximum delay during generation
        # The effective length is seq_length - max_delay_pattern
        codebook_aligned_BxTxC = codebook_aligned_BxTxC[:, :seq_length - max_delay_pattern, :]

        # Clamp values to valid DAC code range [0, Vq-1] (typically 0-1023)
        # DAC vocab size Vq is usually 1024
        min_valid_index = 0
        max_valid_index = self.dac_model.vq_config.codebook_size - 1 # Typically 1024 - 1 = 1023
        invalid_mask = (codebook_aligned_BxTxC < min_valid_index) | (codebook_aligned_BxTxC > max_valid_index)
        # Replace invalid codes with a neutral code, e.g., 0
        codebook_aligned_BxTxC[invalid_mask] = 0

        # Transpose for DAC: [B, C, T]
        codebook_for_dac = codebook_aligned_BxTxC.transpose(1, 2)

        # Decode using DAC
        try:
            with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type, dtype=self.compute_dtype, enabled=self.device.type != 'cpu'):
                audio_values = decode(self.dac_model, codebook_for_dac) # Returns [B, 1, T_audio]
        except Exception as e:
            print(f"Error during DAC decoding: {e}")
            return None

        # Return as numpy array [T_audio]
        return audio_values.squeeze().cpu().numpy()

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Loads and encodes an audio file using the DAC model."""
        if self.dac_model is None:
            raise RuntimeError("DAC model not loaded. Cannot encode audio.")
        try:
            audio, sr = torchaudio.load(audio_path) # Loads to CPU, shape [C, T]
            # Ensure mono or handle multi-channel appropriately (DAC expects mono?)
            # Let's assume we take the mean if stereo
            if audio.shape[0] > 1:
                 audio = torch.mean(audio, dim=0, keepdim=True)

            # Resample if necessary
            if sr != DEFAULT_SAMPLE_RATE:
                audio = torchaudio.functional.resample(audio, sr, DEFAULT_SAMPLE_RATE)

            # Move to device and add batch dim: [1, 1, T]
            audio = audio.to(self.device).unsqueeze(0)

            # Preprocess and encode
            with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type, dtype=self.compute_dtype, enabled=self.device.type != 'cpu'):
                audio_data = self.dac_model.preprocess(audio, DEFAULT_SAMPLE_RATE) # Normalizes etc.
                # z, codes, latents, _, _ = self.dac_model.encode(audio_data)
                _, encoded_frame, _, _, _ = self.dac_model.encode(audio_data) # Shape [B, C, T_codes]

            # Return codes as [T_codes, C]
            return encoded_frame.squeeze(0).transpose(0, 1)

        except FileNotFoundError:
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading or encoding audio file {audio_path}: {e}") from e

    def save_audio(self, path: str, audio: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE):
        """Saves audio numpy array to a file using soundfile."""
        if audio is None:
            print("Warning: Cannot save None audio.")
            return
        try:
            # Ensure output directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            # Ensure audio is float32 for saving, normalize if needed
            if not np.issubdtype(audio.dtype, np.floating):
                 audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max # Example for int16
            # Clamp values to [-1, 1] to prevent clipping
            audio = np.clip(audio, -1.0, 1.0)

            sf.write(path, audio, sample_rate)
            # print(f"Audio saved to {path}")
        except Exception as e:
            print(f"Error saving audio to {path}: {e}")


    def load_adapter_weights(self, adapter_path: str, adapter_name: str = "default"):
         """Loads PEFT adapter weights into the model."""
         if not PEFT_AVAILABLE:
              raise ImportError("PEFT library is required to load adapters. Install with `pip install peft`.")
         from peft import PeftModel

         if not hasattr(self.model, 'load_adapter'):
              print("Trying to load adapters onto a non-PEFT model. Applying PEFT wrapper first.")
              # This assumes a default LoRA config if none was applied before. Risky.
              # It's better if the model was already prepared with PEFT during fine-tuning.
              # For now, we'll assume the user loads a base model THEN calls this.
              # We need to wrap the base model *before* loading adapters.
              # This requires knowing the original LoRA config, which isn't stored here.
              # --> Modification: Assume self.model IS ALREADY a PeftModel if adapters are being loaded.
              # --> Or, modify from_pretrained/from_local to handle adapter loading directly.

              # Let's try checking if it's already a PeftModel
              if not isinstance(self.model, PeftModel):
                   raise RuntimeError("Model is not a PEFT model. Load adapters onto a model previously configured with PEFT (e.g., during fine-tuning).")

         print(f"Loading adapter weights from: {adapter_path}")
         try:
              # Load adapter into the existing PeftModel instance
              self.model.load_adapter(adapter_path, adapter_name=adapter_name)
              print(f"Adapter '{adapter_name}' loaded successfully.")
              # Optionally set the loaded adapter as active
              self.model.set_adapter(adapter_name)
              print(f"Adapter '{adapter_name}' set as active.")
         except Exception as e:
              print(f"Error loading adapter weights from {adapter_path}: {e}")
              raise


    @torch.inference_mode()
    def generate(
        self,
        text: str,
        max_tokens: int | None = None,
        cfg_scale: float = 3.0,
        temperature: float = 1.3,
        top_p: float = 0.95,
        use_torch_compile: bool = False,
        cfg_filter_top_k: int = 35,
        audio_prompt: str | torch.Tensor | None = None,
        # audio_prompt_path: str | None = None, # Deprecated, use audio_prompt
        audio_prompt_text: Optional[str] = None, # Required if audio_prompt is used
        # use_cfg_filter: bool | None = None, # Deprecated
        seed: Optional[int] = None, # Add seed for reproducibility
        verbose: bool = False,
    ) -> np.ndarray | None:
        """Generates audio from text, optionally conditioned on an audio prompt.

        Args:
            text (str): Input text. If using audio_prompt, this should be the *combined*
                        transcript (prompt transcript + generation text).
            max_tokens (int, optional): Maximum number of audio tokens to generate.
                                        Defaults to config.data.audio_length.
            cfg_scale (float, optional): Classifier-Free Guidance scale. Defaults to 3.0.
            temperature (float, optional): Sampling temperature. Defaults to 1.3.
            top_p (float, optional): Nucleus sampling probability. Defaults to 0.95.
            use_torch_compile (bool, optional): Whether to use torch.compile (GPU only). Defaults to False.
            cfg_filter_top_k (int, optional): Top-K filtering applied *before* Top-P during CFG. Defaults to 35. Set to 0 to disable.
            audio_prompt (str | torch.Tensor | None, optional): Path to audio file (wav/mp3) or pre-encoded
                                                                audio tensor [T_codes, C] for voice cloning. Defaults to None.
            audio_prompt_text (Optional[str], optional): The exact transcript of the audio_prompt.
                                                        **Required** if audio_prompt is provided. Defaults to None.
            seed (Optional[int], optional): Random seed for generation. Defaults to None (random).
            verbose (bool, optional): Print progress information. Defaults to False.

        Returns:
            np.ndarray | None: Generated audio waveform as a NumPy array, or None if generation fails.
        """
        # --- Input Validation and Setup ---
        if audio_prompt is not None and not audio_prompt_text:
            raise ValueError("`audio_prompt_text` is required when `audio_prompt` is provided.")
        # if audio_prompt_path:
        #     print("Warning: audio_prompt_path is deprecated. Use audio_prompt instead.")
        #     audio_prompt = audio_prompt_path
        # if use_cfg_filter is not None:
        #     print("Warning: use_cfg_filter is deprecated. Control filtering with cfg_filter_top_k.")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if verbose: print(f"Using seed: {seed}")

        # Determine effective text input (combine if prompt text is given)
        effective_text = audio_prompt_text.strip() + " " + text.strip() if audio_prompt_text else text.strip()
        # Ensure text ends with the previous speaker tag (heuristic for better ending)
        # Find last tag
        last_s1 = effective_text.rfind("[S1]")
        last_s2 = effective_text.rfind("[S2]")
        if last_s1 > last_s2 and not effective_text.endswith("[S2]"):
             effective_text += " [S2]"
        elif last_s2 > last_s1 and not effective_text.endswith("[S1]"):
             effective_text += " [S1]"
        elif last_s1 == -1 and last_s2 == -1 and effective_text: # No tags found, assume S1 start
             effective_text += " [S2]" # Add opposite tag


        audio_eos_value = self.config.data.audio_eos_value
        audio_pad_value = self.config.data.audio_pad_value
        delay_pattern = self.config.data.delay_pattern
        max_tokens = self.config.data.audio_length if max_tokens is None else max_tokens
        max_delay_pattern = max(delay_pattern) if delay_pattern else 0

        self.model.eval() # Ensure model is in eval mode

        if verbose:
            total_start_time = time.time()
            print(f"Generating with parameters: cfg_scale={cfg_scale}, temp={temperature}, top_p={top_p}, top_k={cfg_filter_top_k}")
            print(f"Max tokens: {max_tokens}")
            print(f"Effective text input length: {len(effective_text)}")

        # --- Compilation (Optional) ---
        # Note: Compilation needs to happen outside the main generate call ideally
        # or be handled carefully to avoid recompilation on every call.
        # This simple check is not robust.
        if use_torch_compile and self.device.type == 'cuda' and not hasattr(self, "_compiled_generate"):
             print("Applying torch.compile to generation functions (first call will be slow)...")
             # Compile the core parts
             self._prepare_generation = torch.compile(self._prepare_generation, dynamic=True) # Dynamic shapes
             self._decoder_step = torch.compile(self._decoder_step, fullgraph=True, mode="max-autotune") # Can be static
             self._generate_output = torch.compile(self._generate_output, dynamic=True) # Might have dynamic shapes
             self._compiled_generate = True


        # --- Prepare Initial States ---
        try:
            dec_state, dec_output = self._prepare_generation(effective_text, audio_prompt, verbose)
        except Exception as e:
            print(f"Error during preparation: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Start decoding from the step after the prefill
        dec_step = dec_output.prefill_step - 1

        # --- Generation Loop ---
        bos_countdown = max_delay_pattern # Counter to handle BOS insertion during delay application
        eos_detected = False
        eos_countdown = -1 # Countdown after EOS is detected in channel 0

        if verbose:
            print(f"generate: Starting generation loop from step {dec_step + 1}")
            loop_start_time = time.time()

        try:
            while dec_step < max_tokens - 1: # Stop one step before max_tokens to avoid overflow
                current_step_idx = dec_step + 1
                if current_step_idx >= max_tokens: # Safety break
                     print(f"Warning: Reached max_tokens ({max_tokens}) limit.")
                     break

                # Prepare decoder state for the current step
                dec_state.prepare_step(current_step_idx) # Sets dec_positions correctly

                # Get the input token for this step from the output buffer (previous prediction)
                # Shape [C] -> [1, C] -> [2, 1, C]
                tokens_Bx1xC = dec_output.get_tokens_at(current_step_idx -1).unsqueeze(0).unsqueeze(0).expand(2, 1, -1)

                # Perform one decoder step
                pred_C = self._decoder_step(
                    tokens_Bx1xC,
                    dec_state,
                    cfg_scale,
                    temperature,
                    top_p,
                    cfg_filter_top_k,
                ) # Returns shape [C]

                # --- EOS Handling ---
                # Check if EOS is generated in the first channel
                if not eos_detected and pred_C[0] == audio_eos_value:
                    eos_detected = True
                    eos_countdown = max_delay_pattern # Start countdown
                    if verbose: print(f"EOS detected at step {current_step_idx}. Starting countdown ({eos_countdown} steps).")

                # If EOS countdown is active, force EOS/PAD based on delay pattern
                if eos_countdown > 0:
                    step_after_eos = max_delay_pattern - eos_countdown
                    for i, d in enumerate(delay_pattern):
                        if step_after_eos == d:
                            pred_C[i] = audio_eos_value
                        elif step_after_eos > d:
                            # Only overwrite if not already EOS (avoid overwriting forced EOS)
                            if pred_C[i] != audio_eos_value:
                                 pred_C[i] = audio_pad_value
                    eos_countdown -= 1

                # Update the output buffer with the prediction for the current step
                bos_countdown = max(0, bos_countdown - 1)
                dec_output.update_one(pred_C, current_step_idx, bos_countdown > 0)

                # Check termination conditions
                if eos_countdown == 0:
                    if verbose: print(f"EOS countdown finished at step {current_step_idx}. Stopping generation.")
                    break # Stop generation after countdown finishes

                # Check if max_tokens limit is effectively reached considering delay
                if current_step_idx >= max_tokens - max_delay_pattern -1 and not eos_detected:
                     # Trigger EOS if near the end and EOS hasn't naturally occurred
                     eos_detected = True
                     eos_countdown = max_delay_pattern
                     if verbose: print(f"Nearing max_tokens ({max_tokens}) at step {current_step_idx}. Triggering EOS countdown.")


                dec_step += 1 # Move to the next step

                # Progress Logging
                if verbose and (current_step_idx + 1) % 100 == 0:
                    current_time = time.time()
                    steps_done = current_step_idx - (dec_output.prefill_step - 1)
                    duration = current_time - loop_start_time
                    tokens_per_sec = steps_done / duration if duration > 0 else 0
                    print(f"generate step {current_step_idx+1}/{max_tokens}: speed={tokens_per_sec:.2f} tokens/s")

        except Exception as e:
            print(f"Error during generation loop at step {dec_step}: {e}")
            import traceback
            traceback.print_exc()
            return None # Indicate failure

        # --- Finalize Output ---
        # Extract generated codes (excluding prefill, including the last generated step)
        # Final step index is dec_step
        if dec_output.prefill_step > dec_step + 1:
            print("Warning: No new tokens were generated after prefill.")
            return None

        # +1 because slicing is exclusive end; includes the last `dec_step` prediction
        generated_codes = dec_output.generated_tokens[dec_output.prefill_step : dec_step + 1, :]

        if verbose:
            total_steps_generated = generated_codes.shape[0]
            total_duration = time.time() - total_start_time
            print(f"generate: Total steps generated={total_steps_generated}, total duration={total_duration:.3f}s")
            if total_duration > 0:
                 print(f"generate: Average speed={total_steps_generated / total_duration:.2f} tokens/s")

        # Decode the generated codes to audio waveform
        try:
            output_waveform = self._generate_output(generated_codes)
        except Exception as e:
            print(f"Error during final decoding: {e}")
            return None

        return output_waveform