# Training configuration for Danish fine-tuning

# Model
MODEL_NAME = "unsloth/Qwen3.5-9B"
OUTPUT_DIR = "./outputs"

# Dataset
DATASET_NAME = "kobprof/skolegpt-instruct"
MAX_SEQ_LENGTH = 1024

# LoRA
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

# Training
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 100
SAVE_STEPS = 500
LOGGING_STEPS = 50

# System prompt for Danish
SYSTEM_PROMPT = "Du er en hjælpsom assistent."
