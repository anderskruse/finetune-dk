"""
Upload fine-tuned model to Hugging Face Hub.
"""

import argparse
from huggingface_hub import HfApi, login


def make_model_card(repo, model_name, dataset_name):
    return f"""---
license: apache-2.0
language:
- da
- en
base_model: {model_name}
tags:
- danish
- instruction-tuning
- lora
---

# {repo.split("/")[-1]}

Danish instruction-tuned version of {model_name}, fine-tuned on the {dataset_name} dataset.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo}")
tokenizer = AutoTokenizer.from_pretrained("{repo}")

messages = [
    {{"role": "system", "content": "Du er en hjælpsom assistent."}},
    {{"role": "user", "content": "Hvad er koldskål?"}}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training

- **Base model:** {model_name}
- **Dataset:** {dataset_name}
- **Method:** LoRA fine-tuning with Unsloth

## Acknowledgments

- Base model authors
- Dataset authors
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HF repo name (username/model-name)")
    parser.add_argument("--model-name", default="unsloth/Mistral-Nemo-Instruct-2407",
                        help="Base model name for model card (default: %(default)s)")
    parser.add_argument("--dataset", default="kobprof/skolegpt-instruct",
                        help="Dataset name for model card (default: %(default)s)")
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
        commit_message=f"Upload fine-tuned {args.model_name} model"
    )

    # Upload model card
    print("Uploading model card...")
    card = make_model_card(args.repo, args.model_name, args.dataset)
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        commit_message="Add model card"
    )

    print(f"Done! Model available at: https://huggingface.co/{args.repo}")

if __name__ == "__main__":
    main()
