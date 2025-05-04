# generate_finetune_data.py

import argparse
import os
import random
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

from dia.model import Dia


# Harvard Sentences (Phonetically Balanced) - Sample Set
# Source: http://www.cs.columbia.edu/~hgs/audio/harvard.html (and others)
# Using a subset for brevity, expand as needed.
HARVARD_SENTENCES = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.",
    "Large size in stockings is hard to sell.",
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
    "The source of the huge river is the clear spring.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
    "The salt breeze came across the sea.",
    "The girl at the booth sold fifty bonds.",
    # Add more sentences for better coverage
]

def format_dialogue_prompts(sentences: list[str], max_len_sec: int = 15, min_len_sec: int = 5) -> list[str]:
    """
    Formats sentences into dialogue prompts respecting Dia constraints.
    Groups sentences to meet length constraints and alternates [S1]/[S2].

    Args:
        sentences: List of sentences.
        max_len_sec: Maximum approximate length in seconds for a prompt.
        min_len_sec: Minimum approximate length in seconds for a prompt.

    Returns:
        List of formatted dialogue prompts.
    """
    prompts = []
    current_prompt_sentences = []
    current_speaker_idx = 1  # Start with [S1]
    estimated_duration = 0
    words_per_sec_approx = 2.5 # Rough estimate

    shuffled_sentences = random.sample(sentences, len(sentences))

    for sentence in shuffled_sentences:
        sentence = sentence.strip().rstrip('.') # Clean up sentence

        # Estimate duration increase
        duration_increase = len(sentence.split()) / words_per_sec_approx

        # Check if adding the sentence exceeds max length
        if estimated_duration + duration_increase > max_len_sec and len(current_prompt_sentences) > 0:
            # Finalize the current prompt if it meets min length
            if estimated_duration >= min_len_sec:
                # Add the tag of the *previous* speaker at the end
                final_tag = f"[S{2 if current_speaker_idx == 1 else 1}]"
                prompt_str = " ".join(current_prompt_sentences) + f" {final_tag}"
                prompts.append(prompt_str.strip())

            # Start a new prompt
            current_prompt_sentences = []
            current_speaker_idx = 1
            estimated_duration = 0

        # Add the sentence to the current prompt
        tag = f"[S{current_speaker_idx}]"
        current_prompt_sentences.append(f"{tag} {sentence}.") # Add period back
        estimated_duration += duration_increase
        current_speaker_idx = 2 if current_speaker_idx == 1 else 1 # Alternate speaker

    # Add the last prompt if it meets min length
    if current_prompt_sentences and estimated_duration >= min_len_sec:
        final_tag = f"[S{2 if current_speaker_idx == 1 else 1}]"
        prompt_str = " ".join(current_prompt_sentences) + f" {final_tag}"
        prompts.append(prompt_str.strip())

    print(f"Generated {len(prompts)} dialogue prompts from {len(sentences)} sentences.")
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Generate Fine-tuning Data using Dia TTS and Harvard Sentences")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save generated wavs and metadata.")
    parser.add_argument("--model-name", type=str, default="nari-labs/Dia-1.6B", help="Pretrained Dia model name or path.")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of dialogue prompts to generate.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, cpu, mps). Auto-detects if None.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--min-len-sec", type=int, default=6, help="Min duration target per sample.")
    parser.add_argument("--max-len-sec", type=int, default=18, help="Max duration target per sample.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation (currently only 1 supported).")
    parser.add_argument("--compute-dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Compute dtype for the model.")
    parser.add_argument("--voice-prompt-audio", type=str, default=None, help="Optional path to an audio file (wav/mp3) to use as a voice prompt for all generations.")
    parser.add_argument("--voice-prompt-text", type=str, default=None, help="Required transcript if --voice-prompt-audio is used.")


    args = parser.parse_args()

    # --- Setup ---
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Output directory: {args.output_dir}")

    output_path = Path(args.output_dir)
    wavs_path = output_path / "wavs"
    wavs_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "metadata.csv"

    if args.voice_prompt_audio and not args.voice_prompt_text:
        parser.error("--voice-prompt-text is required when using --voice-prompt-audio.")

    # --- Load Model ---
    print(f"Loading model {args.model_name}...")
    try:
        model = Dia.from_pretrained(args.model_name, compute_dtype=args.compute_dtype, device=device)
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Format Prompts ---
    dialogue_prompts = format_dialogue_prompts(HARVARD_SENTENCES, max_len_sec=args.max_len_sec, min_len_sec=args.min_len_sec)
    if args.num_samples < len(dialogue_prompts):
        dialogue_prompts = random.sample(dialogue_prompts, args.num_samples)
    else:
        print(f"Warning: Requested {args.num_samples} samples, but only {len(dialogue_prompts)} unique prompts could be generated with current settings. Using all.")
        args.num_samples = len(dialogue_prompts)


    # --- Generate Data ---
    metadata = []
    print(f"Generating {args.num_samples} audio samples...")
    # Note: Batching > 1 is complex due to variable output lengths and current API
    if args.batch_size > 1:
        print("Warning: Batch size > 1 not fully supported yet, proceeding with batch size 1.")

    for i in tqdm(range(args.num_samples)):
        text_prompt = dialogue_prompts[i]
        basename = f"harvard_dia_{i:04d}"
        wav_filename = wavs_path / f"{basename}.wav"

        # Prepare generation args
        gen_kwargs = {
            "text": text_prompt,
            "verbose": False, # Keep console clean during batch generation
            # Add other generation params like temp, top_p if needed
        }
        if args.voice_prompt_audio:
            gen_kwargs["audio_prompt"] = args.voice_prompt_audio
            gen_kwargs["audio_prompt_text"] = args.voice_prompt_text
            # The text prompt already includes the generation part
            gen_kwargs["text"] = f"{args.voice_prompt_text.strip()} {text_prompt.strip()}"


        try:
            # Generate audio
            output_audio = model.generate(**gen_kwargs)

            if output_audio is not None and len(output_audio) > 0:
                # Save audio
                model.save_audio(str(wav_filename), output_audio)
                # Add to metadata (LJSpeech format: filename|transcript)
                # We save the *generated* part of the transcript, not the voice prompt part
                metadata.append(f"{wav_filename.name}|{text_prompt}")
            else:
                print(f"Warning: Generation failed or produced empty audio for prompt {i}. Skipping.")

        except Exception as e:
            print(f"Error generating sample {i}: {e}. Skipping.")
            import traceback
            traceback.print_exc() # More detailed error for debugging

    # --- Save Metadata ---
    if metadata:
        with open(metadata_path, "w", encoding="utf-8") as f:
            for line in metadata:
                f.write(line + "\n")
        print(f"Metadata saved to {metadata_path}")
    else:
        print("No metadata generated.")

    print("Dataset generation complete.")

if __name__ == "__main__":
    main()