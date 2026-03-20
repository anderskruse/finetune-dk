"""
Pre-finetune evaluation of Qwen3.5-9B via HuggingFace Inference API.
Runs the same Danish prompts used in evaluate.py for baseline comparison.
"""

import os
from openai import OpenAI
import config

TEST_PROMPTS = [
    "Hvad er koldskål?",
    "Forklar hygge på dansk.",
    "Hvem var Grundtvig?",
    "Skriv en kort opskrift på smørrebrød.",
    "Hvad er forskellen mellem rødgrød og risalamande?",
    "Forklar kvantemekanik på simpelt dansk.",
]

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

MODEL = "Qwen/Qwen3.5-9B:together"

def ask(prompt):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return completion.choices[0].message.content

def main():
    print(f"Model: {MODEL}")
    print(f"Thinking: disabled")
    print("=" * 60)

    for prompt in TEST_PROMPTS:
        print(f"\nQ: {prompt}")
        response = ask(prompt)
        print(f"A: {response}")
        print("-" * 60)

if __name__ == "__main__":
    main()
