import argparse
import torch
from pathlib import Path
import os
import traceback # Import traceback

from dia.model import Dia, DiaConfig
from dia.pruning_utils import (
    apply_unstructured_pruning,
    apply_structured_pruning,
    make_pruning_permanent,
    check_pruning_sparsity
)

# --- Add this function ---
def get_memory_usage():
    """Returns current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024) # Resident Set Size in MB
    except ImportError:
        return "N/A (psutil not installed)"
    except Exception:
        return "N/A (error getting memory)"

def main():
    parser = argparse.ArgumentParser(description="Offline Pruning Script for Dia TTS Model")

    # --- Model Arguments ---
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the pretrained Dia model directory (containing config.json and pytorch_model.bin) or HF repo ID.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the pruned model checkpoint and config.")

    # --- Pruning Arguments ---
    parser.add_argument("--prune-mode", type=str, required=True, choices=["unstructured", "structured"],
                        help="Pruning mode.")
    parser.add_argument("--prune-amount", type=float, required=True,
                        help="Fraction of weights/structures to prune (0.0-1.0). E.g., 0.6 for 60% sparsity.")
    # Structured pruning specific args
    parser.add_argument("--prune-dim", type=int, default=0,
                        help="Dimension for structured pruning (e.g., 0 for output channels/neurons). Only used if --prune-mode=structured.")
    parser.add_argument("--prune-norm", type=int, default=2, choices=[1, 2],
                        help="Norm (1 for L1, 2 for L2) for structured pruning. Only used if --prune-mode=structured.")

    # --- Infrastructure Arguments ---
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to perform pruning on ('cpu' recommended for stability, 'cuda' possible).")
    parser.add_argument("--compute-dtype", type=str, default="float32", choices=["float32"],
                        help="Compute dtype for loading and pruning (float32 recommended for stability).")


    args = parser.parse_args()

    # --- Validate Args ---
    if not (0.0 < args.prune_amount < 1.0):
        parser.error("--prune-amount must be between 0.0 and 1.0 (exclusive).")

    # --- Setup ---
    device = torch.device(args.device)
    # Force float32 for pruning stability, ignore args.compute_dtype for this script
    compute_dtype_torch = torch.float32
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"--- Starting Offline Pruning ---")
    print(f"Initial Memory Usage: {get_memory_usage()} MB")
    print(f"Model Path: {args.model_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Pruning Mode: {args.prune_mode}")
    print(f"Target Sparsity Amount: {args.prune_amount:.2f}")
    if args.prune_mode == "structured":
        print(f"Structured Pruning Dim: {args.prune_dim}")
        print(f"Structured Pruning Norm: L{args.prune_norm}")
    print(f"Device: {device}")
    print(f"Compute Dtype: float32 (forced for pruning)") # Indicate float32 is used
    print("-" * 30)


    # --- Load Model ---
    print(f"Loading base model from {args.model_path}...")
    try:
        # Load model onto specified device but ensure it's float32 for pruning
        dia_wrapper = Dia.from_pretrained(args.model_path, compute_dtype="float32", device=device)
        model = dia_wrapper.model # Get the underlying DiaModel (nn.Module)
        config = dia_wrapper.config
        model.float() # Ensure it's float32
        print(f"Base model loaded successfully. Memory Usage: {get_memory_usage()} MB")
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        exit(1)

    # --- Apply Pruning ---
    print(f"\nApplying {args.prune_mode} pruning...")
    # Pruning utilities work correctly in eval mode too, keep it simple
    model.eval()

    try:
        if args.prune_mode == "unstructured":
            apply_unstructured_pruning(model, amount=args.prune_amount)
        elif args.prune_mode == "structured":
            apply_structured_pruning(model, amount=args.prune_amount, dim=args.prune_dim, n=args.prune_norm)
        # Add a print *after* the call succeeds
        print("Pruning function call completed.")
        print(f"Memory Usage after pruning call: {get_memory_usage()} MB")

    except Exception as e:
        print("\n--- ERROR DURING PRUNING ---")
        print(f"An error occurred: {e}")
        traceback.print_exc()
        print("-----------------------------")
        exit(1)


    # --- Check Sparsity ---
    print("\nChecking sparsity...")
    try:
        achieved_sparsity = check_pruning_sparsity(model)
        print(f"Achieved sparsity: {achieved_sparsity:.4f}")
        print(f"Memory Usage after sparsity check: {get_memory_usage()} MB")
    except Exception as e:
        print("\n--- ERROR DURING SPARSITY CHECK ---")
        print(f"An error occurred: {e}")
        traceback.print_exc()
        print("-----------------------------")
        exit(1)


    # --- Make Pruning Permanent ---
    print("\nMaking pruning permanent (removing masks)...")
    try:
        # Ensure model is on CPU for making permanent? Sometimes safer.
        model.cpu()
        print(f"Moved model to CPU. Memory Usage: {get_memory_usage()} MB")
        make_pruning_permanent(model)
        print("Pruning made permanent.")
        print(f"Memory Usage after making permanent: {get_memory_usage()} MB")
    except Exception as e:
        print("\n--- ERROR MAKING PRUNING PERMANENT ---")
        print(f"An error occurred: {e}")
        traceback.print_exc()
        print("-----------------------------")
        exit(1)


    # --- Save Pruned Model ---
    print(f"\nSaving pruned model to {args.output_dir}...")
    try:
        # Save state dict (now with permanent zeros, on CPU, float32)
        torch.save(model.state_dict(), output_path / "pytorch_model.bin")
        # Save the original config file
        config.save(output_path / "config.json")
        print("Pruned model state dict and config saved.")
    except Exception as e:
        print("\n--- ERROR SAVING MODEL ---")
        print(f"An error occurred: {e}")
        traceback.print_exc()
        print("-----------------------------")
        exit(1)


    print("\n--- Pruning Complete ---")
    print(f"Final Memory Usage: {get_memory_usage()} MB")
    print(f"Pruned model saved in: {args.output_dir}")
    print("RECOMMENDATION: Consider fine-tuning this pruned model (using finetune.py)")
    print("  - For general quality recovery: Use full fine-tuning on a general dataset.")
    print("  - For speaker adaptation: Use LoRA fine-tuning on target speaker data.")

if __name__ == "__main__":
    # Install psutil if needed for memory monitoring
    try:
        import psutil
    except ImportError:
        print("Optional: Install 'psutil' (pip install psutil) for memory usage monitoring.")
    main()
