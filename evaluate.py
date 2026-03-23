"""
Evaluate a fine-tuned model on Danish prompts.
"""

import argparse
from unsloth import FastLanguageModel
import config

TEST_PROMPTS = [
    "Hvad er koldskål?",
    "Forklar hygge på dansk.",
    "Hvem var Grundtvig?",
    "Skriv en kort opskrift på smørrebrød.",
    "Hvad er forskellen mellem rødgrød og risalamande?",
    "Forklar kvantemekanik på simpelt dansk.",
]

def load_model(model_path):
    """Load model for inference."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_8bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate(model, tokenizer, prompt, no_thinking=False):
    """Generate response for a prompt."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": config.SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    kwargs = {"enable_thinking": False} if no_thinking else {}
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        **kwargs,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./outputs/lora_adapter", help="Path to model")
    parser.add_argument("--base", action="store_true", help="Test base model instead")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode (for Qwen3+)")
    args = parser.parse_args()

    model_path = config.MODEL_NAME if args.base else args.model
    print(f"Loading: {model_path}")
    model, tokenizer = load_model(model_path)

    print("\n" + "="*60)
    for prompt in TEST_PROMPTS:
        print(f"\nQ: {prompt}")
        response = generate(model, tokenizer, prompt, no_thinking=args.no_thinking)
        print(f"A: {response}")
        print("-"*60)

if __name__ == "__main__":
    main()
