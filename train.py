"""
Fine-tune a language model on instruction data using LoRA + Unsloth.
"""

import os
import argparse
# Limit multiprocessing before importing libraries that auto-detect CPU count
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["UNSLOTH_NUM_PROC"] = "4"

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import config


def load_model(model_name, lora_r=config.LORA_R, lora_alpha=config.LORA_ALPHA,
               max_seq_length=config.MAX_SEQ_LENGTH):
    """Load model with 8-bit quantization and apply LoRA."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_8bit=True,
        dtype=None,  # auto-detect
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    return model, tokenizer

def format_example(example, tokenizer):
    """Format a single example using the model's chat template."""
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["response"]}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.MODEL_NAME, help="HF model name (default: %(default)s)")
    parser.add_argument("--dataset", default=config.DATASET_NAME, help="HF dataset name (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Per-device batch size (default: %(default)s)")
    parser.add_argument("--grad-accum", type=int, default=config.GRADIENT_ACCUMULATION, help="Gradient accumulation steps (default: %(default)s)")
    parser.add_argument("--lora-r", type=int, default=config.LORA_R, help="LoRA rank (default: %(default)s)")
    parser.add_argument("--lora-alpha", type=int, default=config.LORA_ALPHA, help="LoRA alpha (default: %(default)s)")
    parser.add_argument("--max-seq-length", type=int, default=config.MAX_SEQ_LENGTH, help="Max sequence length (default: %(default)s)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                                  max_seq_length=args.max_seq_length)

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")

    print(f"Formatting {len(dataset)} examples...")
    dataset = dataset.map(
        lambda x: format_example(x, tokenizer),
        remove_columns=dataset.column_names
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=config.WARMUP_STEPS,
        save_steps=config.SAVE_STEPS,
        logging_steps=config.LOGGING_STEPS,
        fp16=False,
        bf16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=4,
        args=training_args,
    )

    print("Starting training...")
    # Resume from latest checkpoint if --resume and checkpoints exist
    resume_checkpoint = None
    if args.resume:
        checkpoints = [
            os.path.join(config.OUTPUT_DIR, d)
            for d in os.listdir(config.OUTPUT_DIR)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            resume_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"Resuming from checkpoint: {resume_checkpoint}")
        else:
            print("No checkpoints found, starting from scratch.")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save
    print("Saving model...")
    model.save_pretrained(os.path.join(config.OUTPUT_DIR, "lora_adapter"))
    tokenizer.save_pretrained(os.path.join(config.OUTPUT_DIR, "lora_adapter"))

    # Also save merged model for easier inference
    print("Saving merged model...")
    model.save_pretrained_merged(
        os.path.join(config.OUTPUT_DIR, "merged_model"),
        tokenizer,
        save_method="merged_16bit"
    )

    print("Done!")

if __name__ == "__main__":
    main()
