"""
Evaluate the fine-tuned Mistral-Nemo model on Danish prompts.
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

def generate(model, tokenizer, prompt):
    """Generate response for a prompt."""
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant response
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./outputs/merged_model", help="Path to model")
    parser.add_argument("--base", action="store_true", help="Test base model instead")
    args = parser.parse_args()

    model_path = config.MODEL_NAME if args.base else args.model
    print(f"Loading: {model_path}")
    model, tokenizer = load_model(model_path)

    print("\n" + "="*60)
    for prompt in TEST_PROMPTS:
        print(f"\nQ: {prompt}")
        response = generate(model, tokenizer, prompt)
        print(f"A: {response}")
        print("-"*60)

if __name__ == "__main__":
    main()
