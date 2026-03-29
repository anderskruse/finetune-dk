"""
Export fine-tuned model to GGUF format for use with llama.cpp, Ollama, etc.

Unsloth's save_pretrained_gguf does not support Qwen3.5 (VL architecture),
so this script merges via Unsloth then converts with llama.cpp directly.
"""

import os
import sys
import subprocess
import argparse
import gc
from unsloth import FastLanguageModel
import config


QUANTIZATION_METHODS = ["q4_k_m", "q5_k_m", "q8_0", "f16"]
LLAMA_CPP_PATH = "/workspace/llama.cpp"


def setup_llama_cpp():
    """Clone and build llama.cpp if not already present."""
    if not os.path.exists(LLAMA_CPP_PATH):
        print("Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp", LLAMA_CPP_PATH],
            check=True
        )

    quantize_bin = os.path.join(LLAMA_CPP_PATH, "build", "bin", "llama-quantize")
    if not os.path.exists(quantize_bin):
        print("Building llama.cpp with CUDA...")
        subprocess.run(
            ["cmake", "-B", "build", "-DGGML_CUDA=ON"],
            cwd=LLAMA_CPP_PATH, check=True
        )
        subprocess.run(
            ["cmake", "--build", "build", "--config", "Release", "-j4"],
            cwd=LLAMA_CPP_PATH, check=True
        )

    # Skip llama.cpp requirements — they conflict with the training venv.
    # The convert script only needs numpy + torch which are already installed.


def merge_model(model_path, merged_path):
    """Load LoRA adapter, merge into base model, save as safetensors."""
    import json

    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_8bit=True,
        dtype=None,
    )

    # Scope the callable-patching workaround to only the save call, then restore.
    _orig = json.JSONEncoder.default
    def _patched(self, obj):
        if callable(obj):
            return None
        return _orig(self, obj)
    json.JSONEncoder.default = _patched
    try:
        print(f"Saving merged model to {merged_path}...")
        model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    finally:
        json.JSONEncoder.default = _orig

    del model
    gc.collect()

    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # Verify the chat template survived the merge; restore from the adapter if not.
    _ensure_chat_template(merged_path, model_path)


def _ensure_chat_template(merged_path, fallback_path):
    """Check that tokenizer_config.json in merged_path has a chat_template.
    If not, copy it from fallback_path (the LoRA adapter directory)."""
    import json

    config_path = os.path.join(merged_path, "tokenizer_config.json")
    if not os.path.exists(config_path):
        print("Warning: tokenizer_config.json not found in merged model directory.")
        return

    with open(config_path, encoding="utf-8") as f:
        tok_config = json.load(f)

    if tok_config.get("chat_template"):
        return  # all good

    print("Warning: chat_template missing from merged tokenizer config — restoring from adapter...")
    fallback_config_path = os.path.join(fallback_path, "tokenizer_config.json")
    if not os.path.exists(fallback_config_path):
        print("Warning: adapter tokenizer_config.json not found; chat template cannot be restored.")
        return

    with open(fallback_config_path, encoding="utf-8") as f:
        fallback_config = json.load(f)

    template = fallback_config.get("chat_template")
    if not template:
        print("Warning: adapter tokenizer also lacks chat_template; GGUF may not behave as instruct model.")
        return

    tok_config["chat_template"] = template
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(tok_config, f, indent=2, ensure_ascii=False)
    print("chat_template restored successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./outputs/lora_adapter",
                        help="Path to LoRA adapter (default: %(default)s)")
    parser.add_argument("--merged-path", default="./outputs/merged_model",
                        help="Intermediate path for merged safetensors (default: %(default)s)")
    parser.add_argument("--output-dir", default="./outputs/gguf",
                        help="Output directory for GGUF files (default: %(default)s)")
    parser.add_argument("--quant", default="q4_k_m", choices=QUANTIZATION_METHODS,
                        help="Quantization method (default: %(default)s)")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merge step if merged model already exists")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.merged_path, exist_ok=True)

    # Step 1: Merge LoRA into base model
    already_merged = any(f.endswith(".safetensors") for f in os.listdir(args.merged_path))
    if args.skip_merge or already_merged:
        print(f"Using existing merged model at {args.merged_path}")
    else:
        merge_model(args.model_path, args.merged_path)

    # Step 2: Set up llama.cpp
    setup_llama_cpp()

    # Step 3: Convert merged safetensors → GGUF f16
    f16_path = os.path.join(args.output_dir, "model-f16.gguf")
    convert_script = os.path.join(LLAMA_CPP_PATH, "convert_hf_to_gguf.py")
    print("Converting to GGUF (f16)...")
    subprocess.run(
        [sys.executable, convert_script, args.merged_path,
         "--outfile", f16_path, "--outtype", "f16"],
        check=True
    )

    if args.quant == "f16":
        print(f"Done! GGUF saved to {f16_path}")
        return

    # Step 4: Quantize
    quant_path = os.path.join(args.output_dir, f"model-{args.quant}.gguf")
    quantize_bin = os.path.join(LLAMA_CPP_PATH, "build", "bin", "llama-quantize")
    print(f"Quantizing to {args.quant}...")
    subprocess.run([quantize_bin, f16_path, quant_path, args.quant.upper()], check=True)

    os.remove(f16_path)
    print(f"Done! GGUF saved to {quant_path}")


if __name__ == "__main__":
    main()
