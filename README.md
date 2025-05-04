<p align="center">
<a href="https://github.com/nari-labs/dia">
<img src="./dia/static/images/banner.png">
</a>
</p>
<p align="center">
<a href="https://tally.so/r/meokbo" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/Join-Waitlist-white?style=for-the-badge"></a>
<a href="https://discord.gg/pgdB5YRe" target="_blank"><img src="https://img.shields.io/badge/Discord-Join%20Chat-7289DA?logo=discord&style=for-the-badge"></a>
<a href="https://github.com/nari-labs/dia/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge" alt="LICENSE"></a>
</p>
<p align="center">
<a href="https://huggingface.co/nari-labs/Dia-1.6B"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-lg-dark.svg" alt="Dataset on HuggingFace" height=42 ></a>
<a href="https://huggingface.co/spaces/nari-labs/Dia-1.6B"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-lg-dark.svg" alt="Space on HuggingFace" height=38></a>
</p>

Dia is a 1.6B parameter text to speech model created by Nari Labs.

Dia **directly generates highly realistic dialogue from a transcript**. You can condition the output on audio, enabling emotion and tone control. The model can also produce nonverbal communications like laughter, coughing, clearing throat, etc.

To accelerate research, we are providing access to pretrained model checkpoints and inference code. The model weights are hosted on [Hugging Face](https://huggingface.co/nari-labs/Dia-1.6B). The model only supports English generation at the moment.

We also provide a [demo page](https://yummy-fir-7a4.notion.site/dia) comparing our model to [ElevenLabs Studio](https://elevenlabs.io/studio) and [Sesame CSM-1B](https://github.com/SesameAILabs/csm).

- (Update) We have a ZeroGPU Space running! Try it now [here](https://huggingface.co/spaces/nari-labs/Dia-1.6B). Thanks to the HF team for the support :)
- Join our [discord server](https://discord.gg/pgdB5YRe) for community support and access to new features.
- Play with a larger version of Dia: generate fun conversations, remix content, and share with friends. 🔮 Join the [waitlist](https://tally.so/r/meokbo) for early access.

## ⚡️ Quickstart

### Install via pip

```bash
# Install directly from GitHub
pip install git+https://github.com/nari-labs/dia.git

# For LoRA fine-tuning, also install peft:
# pip install peft
```

### Run the Gradio UI

This will open a Gradio UI that you can work on.

```bash
git clone https://github.com/nari-labs/dia.git
cd dia

# Using uv (recommended)
uv venv # Create virtual environment
source .venv/bin/activate
uv pip install -e .[dev] # Install editable with dev dependencies
python app.py

# Using pip/venv
# python -m venv .venv
# source .venv/bin/activate
# pip install -e .[dev]
# python app.py
```

Note that the model was not fine-tuned on a specific voice. Hence, you will get different voices every time you run the model.
You can keep speaker consistency by either adding an audio prompt (see Features and Caveats sections), or fixing the seed.

## Features

- Generate dialogue via `[S1]` and `[S2]` tag
- Generate non-verbal like `(laughs)`, `(coughs)`, etc.
  - Below verbal tags will be recognized, but might result in unexpected output.
  - `(laughs), (clears throat), (sighs), (gasps), (coughs), (singing), (sings), (mumbles), (beep), (groans), (sniffs), (claps), (screams), (inhales), (exhales), (applause), (burps), (humming), (sneezes), (chuckle), (whistles)`
- Voice cloning. See [`example/voice_clone.py`](example/voice_clone.py) for more information.
  - In the Hugging Face space, you can upload the audio you want to clone and place its transcript before your script. Make sure the transcript follows the required format. The model will then output only the content of your script.
- **Model Pruning**: Reduce model size and potentially speed up inference using structured or unstructured pruning. See the [Pruning](#-pruning) section.
- **Fine-tuning**: Adapt the model to specific voices or styles using full fine-tuning or LoRA adapters. See the [Fine-tuning](#-fine-tuning) section.

## ⚙️ Usage

### As a Python Library

```python
from dia.model import Dia

# Load the base model
model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16") # Or float32 for CPU

# --- Example Generation ---
text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face. [S1]" # End with previous speaker tag

output = model.generate(text, use_torch_compile=False, verbose=True) # use_torch_compile=True for GPU speedup

model.save_audio("simple.wav", output) # Save as WAV

# --- Example Voice Cloning ---
clone_from_text = "[S1] This is the transcript of the audio I want to clone. It should be about 5 to 10 seconds long. [S1]"
clone_from_audio = "path/to/your/audio_prompt.wav" # 5-10 seconds WAV/MP3
text_to_generate = "[S1] Now generate this text using the voice from the audio prompt. [S2] It should sound similar! [S1]"

# Combine prompt text and generation text
full_text = clone_from_text + " " + text_to_generate

cloned_output = model.generate(
    full_text,
    audio_prompt=clone_from_audio,
    verbose=True
)
model.save_audio("voice_clone.wav", cloned_output)

# --- Using a Pruned/Fine-tuned Model (Example) ---
# Assume you saved a pruned model to './pruned_model_dir'
# pruned_model = Dia.from_local("./pruned_model_dir/config.json", "./pruned_model_dir/pytorch_model.bin")
# output_pruned = pruned_model.generate(text)
# pruned_model.save_audio("pruned_output.wav", output_pruned)

# Assume you saved LoRA adapters to './lora_adapter_dir' on top of the base model
# from peft import PeftModel # Requires pip install peft
# base_model = Dia.from_pretrained("nari-labs/Dia-1.6B")
# lora_model = PeftModel.from_pretrained(base_model.model, './lora_adapter_dir')
# # Need to integrate PeftModel back into Dia wrapper or use it directly (more complex)
# # Direct usage example (conceptual):
# # lora_output = dia_wrapper_with_peft_model.generate(text) # Requires modification of Dia class
```

### Command Line Interface (CLI)

Generate audio directly from the command line:

```bash
python cli.py \
    "[S1] This is a test from the command line. [S2] It should generate an audio file. [S1]" \
    --output generated_cli.wav \
    --repo-id nari-labs/Dia-1.6B \
    --device cpu # Or cuda if available
    # --seed 1234 # Optional: for reproducibility
    # --audio-prompt path/to/prompt.wav # Optional: for voice cloning
    # --audio-prompt-text "[S1] Transcript of the audio prompt. [S1]" # Required if using audio-prompt
    # --pruned-checkpoint path/to/pruned/pytorch_model.bin # Load a pruned model
    # --adapter-path path/to/lora/adapter/dir # Load LoRA adapters (requires PEFT)
```

See `python cli.py --help` for all options.

## 💻 Hardware and Inference Speed

Dia has been tested on GPUs (PyTorch 2.0+, CUDA 11.8+) and CPU. CPU inference will be significantly slower.
The initial run will take longer as the Descript Audio Codec (DAC) model also needs to be downloaded (~500MB).

**GPU Benchmarks (RTX 4090):**

| precision | realtime factor w/ compile | realtime factor w/o compile | VRAM |
|:---------:|:--------------------------:|:---------------------------:|:----:|
| `bfloat16`| x2.1                       | x1.5                        | ~10GB|
| `float16` | x2.2                       | x1.3                        | ~10GB|
| `float32` | x1                         | x0.9                        | ~13GB|

(`use_torch_compile=True` enables PyTorch 2 compilation for potential speedup on GPUs, requires compatible hardware/drivers).

**CPU Usage:**

- Pruning the model (see [Pruning](#-pruning)) can significantly reduce the parameter count and memory footprint, making CPU inference more feasible.
- Using `compute_dtype="float32"` is recommended for CPU.
- The Gradio app (`app.py`) uses `torch.quantization.quantize_dynamic` for CPU execution, which can further reduce latency and memory usage. Expect slower-than-realtime performance on most CPUs.
- **Recommended Batch Size for CPU:** Start with a batch size of 1 for generation or fine-tuning on CPU to avoid memory issues.

If you don't have hardware available or if you want to play with bigger versions of our models, join the waitlist [here](https://tally.so/r/meokbo).

## ✂️ Pruning

Pruning removes redundant weights from the model, reducing its size and potentially speeding up inference, especially on CPU or resource-constrained devices. We provide utilities for both unstructured and structured pruning.

**Important:** Pruning is typically done *offline*. You prune the model once and save the smaller checkpoint.

**Steps:**

1.  **Load the Base Model:** Start with the pretrained Dia model.
2.  **Apply Pruning:** Use functions from `dia.pruning_utils` to apply the desired pruning method.
3.  **Make Pruning Permanent:** Remove the pruning masks to get a standard model with zeroed weights.
4.  **Save the Pruned Model:** Save the `state_dict` and `config.json`.
5.  **(Optional but Recommended) Fine-tune:** Fine-tune the pruned model for a few epochs on relevant data to recover potential quality loss (see [Fine-tuning](#-fine-tuning)).

**Example Script (Conceptual - adapt as needed):**

```python
# prune_model.py (Example)
import torch
from dia.model import Dia, DiaConfig
from dia.pruning_utils import apply_unstructured_pruning, apply_structured_pruning, make_pruning_permanent, check_pruning_sparsity
from pathlib import Path

# --- Configuration ---
MODEL_ID = "nari-labs/Dia-1.6B"
OUTPUT_DIR = "./pruned_dia_50_unstructured"
PRUNE_MODE = "unstructured" # "unstructured" or "structured"
PRUNE_AMOUNT = 0.5 # 50% sparsity
STRUCTURED_DIM = 0 # For structured: 0 for output neurons/channels
STRUCTURED_NORM = 2 # For structured: L2 norm

# --- Load Model ---
print(f"Loading base model: {MODEL_ID}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dia_wrapper = Dia.from_pretrained(MODEL_ID, compute_dtype="float32", device=device) # Use float32 for pruning stability
model = dia_wrapper.model
config = dia_wrapper.config

# --- Apply Pruning ---
if PRUNE_MODE == "unstructured":
    print(f"Applying unstructured pruning ({PRUNE_AMOUNT * 100}%)")
    apply_unstructured_pruning(model, amount=PRUNE_AMOUNT)
elif PRUNE_MODE == "structured":
    print(f"Applying structured pruning ({PRUNE_AMOUNT * 100}%, dim={STRUCTURED_DIM}, norm=L{STRUCTURED_NORM})")
    apply_structured_pruning(model, amount=PRUNE_AMOUNT, dim=STRUCTURED_DIM, n=STRUCTURED_NORM)
else:
    print("No pruning applied.")

# --- Check Sparsity ---
check_pruning_sparsity(model)

# --- Make Permanent & Save ---
make_pruning_permanent(model)
print(f"Saving pruned model to {OUTPUT_DIR}")
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), output_path / "pytorch_model.bin")
config.save(str(output_path / "config.json"))

print("Pruning complete. Model saved.")
print("RECOMMENDATION: Fine-tune this pruned model to potentially recover quality.")

```

**Loading a Pruned Model:**

Use the `--pruned-checkpoint` argument in `cli.py` or load manually using `Dia.from_local`.

```bash
# Example using CLI
python cli.py "[S1] Testing the pruned model. [S1]" --output pruned_test.wav --pruned-checkpoint ./pruned_dia_50_unstructured/pytorch_model.bin --config ./pruned_dia_50_unstructured/config.json --device cpu
```

**Choosing Pruning Parameters:**

-   **Unstructured:** Often achieves higher sparsity (e.g., 50-70%) with less quality impact initially. Good for general size reduction.
-   **Structured (dim=0):** Prunes entire output neurons (Linear) or filters (Conv). Can lead to better speedups on some hardware but might impact quality more heavily. Start with lower amounts (e.g., 20-40%).
-   **Amount:** Start low (e.g., 0.2) and increase gradually, fine-tuning in between if possible (iterative pruning). High amounts (e.g., > 0.8) will likely require significant fine-tuning.

## ✨ Fine-tuning

Fine-tuning adapts the Dia model to specific voices, speaking styles, or domains. You can either fine-tune the entire (potentially pruned) model or use parameter-efficient fine-tuning (PEFT) methods like LoRA.

**Workflow:**

1.  **Prepare Dataset:**
    *   **Option A (Generate with Dia):** Use the `generate_finetune_data.py` script to create a dataset using Harvard sentences spoken by the base Dia model (or a prompted voice). This is useful for general adaptation or style transfer.
        ```bash
        python generate_finetune_data.py \
            --output-dir ./harvard_dataset \
            --num-samples 100 \
            --device cuda # Or cpu
            # --voice-prompt-audio path/to/voice.wav # Optional: Generate data with a specific voice
            # --voice-prompt-text "[S1] Transcript for voice prompt. [S1]" # Required if using voice prompt
        ```
    *   **Option B (Your Own Data):** Prepare your data in LJSpeech format (a `metadata.csv` file mapping `wav_filename|transcript` and a `wavs` folder). Ensure transcripts follow Dia's `[S1]`/`[S2]` format and meet the length/content caveats (see [Caveats](#-caveats)). 5-15 minutes of high-quality audio per speaker is a good starting point for LoRA. More data is needed for full fine-tuning.

2.  **Run Fine-tuning Script:** Use `finetune.py`.

    *   **Full Fine-tuning (potentially on a pruned model):**
        ```bash
        python finetune.py \
            --model-path nari-labs/Dia-1.6B # Or path to your pruned model dir
            --dataset-dir ./harvard_dataset # Or path to your custom dataset
            --output-dir ./finetuned_full_dia \
            --epochs 5 \
            --batch-size 2 \
            --learning-rate 5e-5 \
            --device cuda \
            # --prune-mode structured --prune-amount 0.3 # Example: Fine-tune a 30% structured pruned model
        ```

    *   **LoRA Fine-tuning (Parameter-Efficient):** Requires `pip install peft`.
        ```bash
        python finetune.py \
            --model-path nari-labs/Dia-1.6B \
            --dataset-dir ./your_speaker_dataset \
            --output-dir ./finetuned_lora_speakerX \
            --adapter-mode lora \
            --lora-rank 8 \
            --lora-alpha 16 \
            --epochs 10 \
            --batch-size 4 \
            --learning-rate 1e-4 \
            --device cuda
        ```

    **Note on Loss Calculation:** The provided `finetune.py` script includes the structure for fine-tuning but uses a *placeholder loss function*. The core Dia model class does not expose its internal training loss calculation. You will need to adapt the script to compute a suitable TTS loss (e.g., based on predicted token probabilities vs. target audio tokens) by modifying the model's forward pass or integrating with the original training logic if available.

3.  **Use the Fine-tuned Model:**
    *   **Full Model:** Load using `Dia.from_local` pointing to the `output-dir/final_model`.
    *   **LoRA Adapters:** Load the base model first, then apply the adapters using the PEFT library. See `cli.py` for an example using `--adapter-path` or adapt the Python usage example.

**Fine-tuning Batch Sizes:**

-   **GPU (e.g., RTX 4090):** Start with batch sizes like 2-8 for full fine-tuning, potentially higher (4-16) for LoRA, depending on VRAM. Use gradient accumulation if needed.
-   **CPU:** Use a batch size of 1. Training will be very slow.

## ⚠️ Caveats & Best Practices

Please follow these guidelines for best results:

1.  **Input Text Length:** Keep input text moderate.
    *   Short input (< 5s audio equivalent) can sound unnatural.
    *   Very long input (> 20s audio equivalent) can make speech unnaturally fast. Aim for segments of 5-15 seconds.
2.  **Non-Verbal Tags:** Use tags like `(laughs)` sparingly from the [supported list](#features). Overusing or using unlisted tags may cause artifacts.
3.  **Speaker Tags:**
    *   **Always** start input with `[S1]`.
    *   **Always** alternate between `[S1]` and `[S2]` (e.g., `[S1] ... [S2] ... [S1] ...`). Do not repeat the same tag consecutively (`[S1] ... [S1] ...`).
    *   **End** the input text with the tag of the *second-to-last* speaker (e.g., if the last utterance is `[S2]`, end the whole text string with `[S1]`). This improves audio quality at the end.
4.  **Audio Prompts (Voice Cloning):**
    *   Provide the **exact transcript** of the audio prompt *before* the text you want to generate.
    *   The transcript must use `[S1]`, `[S2]` correctly. For a single speaker in the prompt audio, use only `[S1]`.
    *   Ideal audio prompt duration is **5-10 seconds**. Shorter may not capture the voice well, longer may introduce instability. (1 second ≈ 86 DAC tokens).
5.  **Pruning & Fine-tuning:** Aggressive pruning often requires fine-tuning to restore quality. LoRA is efficient for adapting to new speakers without retraining the entire model.

## 🪪 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This project offers a high-fidelity speech generation model intended for research and educational use. The following uses are **strictly forbidden**:

- **Identity Misuse**: Do not produce audio resembling real individuals without permission.
- **Deceptive Content**: Do not use this model to generate misleading content (e.g. fake news)
- **Illegal or Malicious Use**: Do not use this model for activities that are illegal or intended to cause harm.

By using this model, you agree to uphold relevant legal standards and ethical responsibilities. We **are not responsible** for any misuse and firmly oppose any unethical usage of this technology.

## 🔭 TODO / Future Work

- Integrate proper loss calculation into `finetune.py`.
- Docker support for ARM architecture and MacOS.
- Optimize inference speed further (e.g., Flash Attention, better quantization).
- Add quantization-aware training.
- Explore advanced PEFT techniques (e.g., AdaLoRA).

## 🤝 Contributing

We are a tiny team of 1 full-time and 1 part-time research-engineers. We are extra-welcome to any contributions!
Join our [Discord Server](https://discord.gg/pgdB5YRe) for discussions.

## 🤗 Acknowledgements

- We thank the [Google TPU Research Cloud program](https://sites.research.google/trc/about/) for providing computation resources.
- Our work was heavily inspired by [SoundStorm](https://arxiv.org/abs/2305.09636), [Parakeet](https://jordandarefsky.com/blog/2024/parakeet/), and [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec).
- Hugging Face for providing the ZeroGPU Grant.
- "Nari" is a pure Korean word for lily.
- We thank Jason Y. for providing help with data filtering.


## ⭐ Star History

<a href="https://www.star-history.com/#nari-labs/dia&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date" />
 </picture>
</a>