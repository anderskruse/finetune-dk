# Training configuration for Qwen3-8B Danish fine-tuning

# Model
MODEL_NAME = "Qwen/Qwen3-8B"
OUTPUT_DIR = "./outputs"

# Dataset
DATASET_NAME = "kobprof/skolegpt-instruct"
MAX_SEQ_LENGTH = 1024

# LoRA
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Training
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 100
SAVE_STEPS = 500
LOGGING_STEPS = 50

# System prompt for Danish
SYSTEM_PROMPT = "Du er en hjælpsom assistent."
