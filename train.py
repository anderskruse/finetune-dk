"""
Fine-tune Qwen3-8B on Danish instruction data.
"""

import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import config

def load_model():
    """Load Qwen3 with 4-bit quantization and apply LoRA."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,  # auto-detect
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    return model, tokenizer

def format_example(example, tokenizer):
    """Format a single example for Qwen3 chat template."""
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["response"]}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

def main():
    print("Loading model...")
    model, tokenizer = load_model()

    print("Loading dataset...")
    dataset = load_dataset(config.DATASET_NAME, split="train")

    print(f"Formatting {len(dataset)} examples...")
    dataset = dataset.map(
        lambda x: format_example(x, tokenizer),
        remove_columns=dataset.column_names
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.NUM_EPOCHS,
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
        max_seq_length=config.MAX_SEQ_LENGTH,
        args=training_args,
        dataset_num_proc=4,
    )

    print("Starting training...")
    trainer.train()

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
