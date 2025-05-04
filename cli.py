# cli.py

import argparse
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dia.model import Dia, DiaConfig

# PEFT integration (optional, for loading adapters)
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN (if used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Generate audio using the Dia model via CLI.")

    parser.add_argument("text", type=str, help="Input text for speech generation. If using --audio-prompt, this should be the text to *generate*, not the prompt transcript.")
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the generated audio file (e.g., output.wav)."
    )

    model_group = parser.add_argument_group("Model Loading")
    model_group.add_argument(
        "--model-path",
        type=str,
        default="nari-labs/Dia-1.6B",
        help="Hugging Face repository ID or local path to model directory (containing config.json and pytorch_model.bin).",
    )
    # Arguments for loading potentially pruned models from local paths
    model_group.add_argument(
        "--config", type=str, default=None, help="Path to local config.json file (overrides config found in --model-path if provided).")
    model_group.add_argument(
        "--pruned-checkpoint", type=str, default=None, help="Path to a specific (potentially pruned) model checkpoint .bin file (overrides checkpoint found in --model-path). Requires --config if model-path is not a local dir.")
    model_group.add_argument(
        "--adapter-path", type=str, default=None, help="Path to LoRA adapter directory (requires PEFT installed). Loads adapters on top of the base model specified by --model-path.")


    prompt_group = parser.add_argument_group("Audio Prompting (Voice Cloning)")
    prompt_group.add_argument(
        "--audio-prompt", type=str, default=None, help="Path to an audio prompt WAV/MP3 file for voice cloning (5-10 seconds recommended)."
    )
    prompt_group.add_argument(
        "--audio-prompt-text", type=str, default=None, help="Required: Exact transcript of the --audio-prompt file."
    )


    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of audio tokens to generate (defaults to config value).",
    )
    gen_group.add_argument(
        "--cfg-scale", type=float, default=3.0, help="Classifier-Free Guidance scale (default: 3.0)."
    )
    gen_group.add_argument(
        "--temperature", type=float, default=1.3, help="Sampling temperature (higher is more random, default: 1.3)."
    )
    gen_group.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling probability (default: 0.95).")
    gen_group.add_argument("--cfg-filter-top-k", type=int, default=35, help="Top-K filter for CFG (0 to disable, default: 35).")
    gen_group.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")


    infra_group = parser.add_argument_group("Infrastructure")
    infra_group.add_argument(
        "--device",
        type=str,
        default=None, # Auto-detect
        help="Device to run inference on (e.g., 'cuda', 'cpu', 'mps', default: auto).",
    )
    infra_group.add_argument(
        "--compute-dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Compute dtype for inference (default: float16). Will be overridden to float32 on CPU.")
    infra_group.add_argument("--verbose", action="store_true", help="Print verbose generation progress.")


    args = parser.parse_args()

    # --- Validations ---
    if args.audio_prompt and not args.audio_prompt_text:
        parser.error("--audio-prompt-text is required when using --audio-prompt.")
    if args.pruned_checkpoint and not args.config and not Path(args.model_path).is_dir():
         parser.error("--config is required when using --pruned-checkpoint with a non-local --model-path (e.g., HF repo ID).")
    if args.adapter_path and not PEFT_AVAILABLE:
         parser.error("--adapter-path requires the PEFT library. Install with `pip install peft`.")

    # --- Setup ---
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Using seed: {args.seed}")

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    # Adjust compute dtype for CPU
    compute_dtype = args.compute_dtype
    if device.type == 'cpu' and compute_dtype != 'float32':
        print("Device is CPU, setting compute_dtype to float32.")
        compute_dtype = 'float32'

    # --- Load Model ---
    print("Loading model...")
    model_load_path = args.model_path
    config_load_path = args.config
    checkpoint_load_path = args.pruned_checkpoint

    try:
        if checkpoint_load_path:
            # Loading a specific (potentially pruned) checkpoint
            print(f"Loading specific checkpoint: {checkpoint_load_path}")
            if not config_load_path:
                 # Try to find config in model_path if it's a directory
                 if Path(model_load_path).is_dir():
                      config_load_path = Path(model_load_path) / "config.json"
                      if not config_load_path.exists():
                           parser.error(f"Config file not found in {model_load_path} and --config not provided.")
                 else:
                      parser.error("--config path is required with --pruned-checkpoint when --model-path is not a local directory.")
            print(f"Using config: {config_load_path}")
            dia_wrapper = Dia.from_local(
                config_path=str(config_load_path),
                checkpoint_path=checkpoint_load_path,
                compute_dtype=compute_dtype,
                device=device
            )
        else:
            # Loading from HF repo or local directory specified by model_path
            print(f"Loading model from: {model_load_path}")
            dia_wrapper = Dia.from_pretrained(
                model_name=model_load_path,
                compute_dtype=compute_dtype,
                device=device
            )

        # --- Load Adapters (Optional) ---
        if args.adapter_path:
            if not PEFT_AVAILABLE:
                 raise ImportError("PEFT not installed, cannot load adapters.")
            print(f"Loading LoRA adapters from: {args.adapter_path}")
            # Wrap the base model with PeftModel
            # Ensure the underlying nn.Module is passed
            dia_wrapper.model = PeftModel.from_pretrained(dia_wrapper.model, args.adapter_path)
            dia_wrapper.model.eval() # Ensure adapter model is in eval mode
            print("LoRA adapters loaded successfully.")
            # Note: The Dia wrapper's generate method might need adjustments
            # if the PEFT model changes input/output structure significantly,
            # but usually it wraps transparently for inference.

        print("Model loaded successfully.")

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # --- Prepare Text ---
    # If using audio prompt, combine transcript with generation text
    if args.audio_prompt:
        full_text = args.audio_prompt_text.strip() + " " + args.text.strip()
    else:
        full_text = args.text.strip()

    # --- Generate Audio ---
    print("Generating audio...")
    try:
        output_audio = dia_wrapper.generate(
            text=full_text, # Pass the potentially combined text
            audio_prompt=args.audio_prompt, # Pass the audio prompt path/tensor
            audio_prompt_text=args.audio_prompt_text, # Pass the transcript (used internally by generate)
            max_tokens=args.max_tokens,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            top_p=args.top_p,
            cfg_filter_top_k=args.cfg_filter_top_k,
            seed=args.seed, # Pass seed again for internal reproducibility if needed
            verbose=args.verbose,
            # use_torch_compile=False # Compile usually handled outside or via env var
        )
        print("Audio generation complete.")

        if output_audio is None:
             print("Generation failed to produce audio.")
             exit(1)

        # --- Save Audio ---
        print(f"Saving audio to {args.output}...")
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use the wrapper's save method which uses soundfile
        dia_wrapper.save_audio(args.output, output_audio)
        print(f"Audio successfully saved to {args.output}")

    except Exception as e:
        print(f"Error during audio generation or saving: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()