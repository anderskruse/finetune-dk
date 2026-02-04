"""
Upload fine-tuned Mistral-Nemo model to Hugging Face Hub.
"""

import argparse
from huggingface_hub import HfApi, login

MODEL_CARD = """---
license: apache-2.0
language:
- da
- en
base_model: mistralai/Mistral-Nemo-Instruct-2407
tags:
- danish
- mistral-nemo
- instruction-tuning
- lora
---

# Mistral-Nemo-12B Danish Instruct

Danish instruction-tuned version of Mistral-Nemo-12B, fine-tuned on the skolegpt-instruct dataset.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("YOUR_USERNAME/Mistral-Nemo-12B-Danish-Instruct")
tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/Mistral-Nemo-12B-Danish-Instruct")

messages = [
    {"role": "system", "content": "Du er en hjælpsom assistent."},
    {"role": "user", "content": "Hvad er koldskål?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training

- **Base model:** mistralai/Mistral-Nemo-Instruct-2407
- **Dataset:** kobprof/skolegpt-instruct
- **Method:** LoRA fine-tuning with Unsloth
- **Hardware:** RTX 4090

## Acknowledgments

- Mistral AI for the base model
- KobProf for the skolegpt-instruct dataset
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HF repo name (username/model-name)")
    parser.add_argument("--model-path", default="./outputs/merged_model", help="Local model path")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()

    print("Logging in to Hugging Face...")
    login()

    api = HfApi()

    print(f"Creating repo: {args.repo}")
    api.create_repo(args.repo, private=args.private, exist_ok=True)

    print(f"Uploading model from {args.model_path}...")
    api.upload_folder(
        folder_path=args.model_path,
        repo_id=args.repo,
        commit_message="Upload fine-tuned Mistral-Nemo-12B Danish model"
    )

    # Upload model card
    print("Uploading model card...")
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        commit_message="Add model card"
    )

    print(f"Done! Model available at: https://huggingface.co/{args.repo}")

if __name__ == "__main__":
    main()
