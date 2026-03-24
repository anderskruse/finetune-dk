"""
Export fine-tuned model to GGUF format for use with llama.cpp, Ollama, etc.
"""

import os
import argparse
from unsloth import FastLanguageModel
import config


QUANTIZATION_METHODS = [
    "q4_k_m",
    "q5_k_m",
    "q8_0",
    "f16",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./outputs/lora_adapter",
                        help="Path to LoRA adapter or merged model (default: %(default)s)")
    parser.add_argument("--output-dir", default="./outputs/gguf",
                        help="Output directory for GGUF files (default: %(default)s)")
    parser.add_argument("--quant", default="q4_k_m", choices=QUANTIZATION_METHODS,
                        help="Quantization method (default: %(default)s)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_8bit=True,
        dtype=None,
    )

    # Patch: Qwen3.5 VL config contains function objects buried in nested sub-configs
    # that break JSON serialization during the merge step.
    # Monkey-patch the JSON encoder to silently drop any non-serializable callables.
    import json, types
    _original_default = json.JSONEncoder.default
    def _patched_default(self, obj):
        if callable(obj):
            return None
        return _original_default(self, obj)
    json.JSONEncoder.default = _patched_default

    print(f"Exporting to GGUF with {args.quant} quantization...")
    model.save_pretrained_gguf(
        args.output_dir,
        tokenizer,
        quantization_method=args.quant,
    )

    print(f"Done! GGUF files saved to {args.output_dir}")


if __name__ == "__main__":
    main()
