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

    # Patch: Qwen3.5 VL config contains function objects that break JSON serialization
    # during the merge step — replace any callable config values with None
    import types
    cfg = model.config
    for key in list(vars(cfg).keys()):
        val = getattr(cfg, key, None)
        if isinstance(val, types.FunctionType):
            setattr(cfg, key, None)

    print(f"Exporting to GGUF with {args.quant} quantization...")
    model.save_pretrained_gguf(
        args.output_dir,
        tokenizer,
        quantization_method=args.quant,
    )

    print(f"Done! GGUF files saved to {args.output_dir}")


if __name__ == "__main__":
    main()
