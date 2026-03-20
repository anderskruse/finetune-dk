# Danish LLM Fine-tuning

Fine-tune instruction-following LLMs on Danish data using LoRA + [Unsloth](https://unsloth.ai). Supports any Unsloth-compatible model.

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Install dependencies
uv pip install -e .  # or: pip install -e .

# Login to Hugging Face (need Write token from https://huggingface.co/settings/tokens)
hf auth login

# Train with default model (Qwen3.5-9B)
python train.py --load-in-16bit --no-thinking

# Evaluate
python evaluate.py --model ./outputs/merged_model

# Upload to Hugging Face
python upload_model.py --repo yourusername/My-Danish-Model

# Export to GGUF (for llama.cpp, Ollama, etc.)
python export_gguf.py
```

## Supported Models

Different model families have different requirements:

| Model | Command |
|---|---|
| **Qwen3.5-9B** (default) | `python train.py --load-in-16bit --no-thinking` |
| **Qwen3.5-27B** | `python train.py --model unsloth/Qwen3.5-27B --load-in-16bit --no-thinking` |
| **Qwen3-14B** | `python train.py --model unsloth/Qwen3-14B --load-in-16bit --no-thinking` |
| **Mistral-Nemo-12B** | `python train.py --model unsloth/Mistral-Nemo-Instruct-2407` |
| **Llama-3.1-8B** | `python train.py --model unsloth/Meta-Llama-3.1-8B-Instruct` |

**`--load-in-16bit`** — required for Qwen3.5 models (quantization causes accuracy issues). Other models default to 8-bit.

**`--no-thinking`** — disables `<think>` reasoning blocks in the chat template. Use for Qwen3 and Qwen3.5 models when you want direct answers without chain-of-thought output.

## Files

- `config.py` - Training hyperparameters and defaults
- `train.py` - Main training script
- `evaluate.py` - Test model on Danish prompts
- `upload_model.py` - Upload to Hugging Face Hub
- `export_gguf.py` - Export model to GGUF format

## Training

Uses LoRA fine-tuning with [Unsloth](https://unsloth.ai) on the [kobprof/skolegpt-instruct](https://huggingface.co/datasets/kobprof/skolegpt-instruct) Danish instruction dataset.

Default config:
- LoRA r=16, alpha=16
- Learning rate: 5e-5
- Batch size: 2 (with 8x gradient accumulation)
- Epochs: 3

All settings can be overridden via CLI args:

```bash
python train.py --model unsloth/Qwen3.5-9B --dataset your/dataset
python train.py --epochs 5 --lr 1e-4 --batch-size 4 --grad-accum 4
python train.py --lora-r 32 --lora-alpha 32 --max-seq-length 2048
python train.py --resume  # Resume from latest checkpoint
```

## GGUF Export

Export the fine-tuned model to GGUF format for use with llama.cpp, Ollama, LM Studio, etc.

```bash
# Default: q4_k_m quantization
python export_gguf.py

# Choose quantization method
python export_gguf.py --quant q8_0

# Custom paths
python export_gguf.py --model-path ./outputs/lora_adapter --output-dir ./outputs/gguf
```

Available quantization methods: `q4_k_m`, `q5_k_m`, `q8_0`, `f16`

## Upload

```bash
# Basic upload
python upload_model.py --repo yourusername/My-Model

# Specify base model name for the model card
python upload_model.py --repo yourusername/My-Model --model-name unsloth/Qwen3.5-9B --dataset kobprof/skolegpt-instruct
```

## RunPod

### 1. Create a pod
- GPU: RTX 4090 or A5000
- Template: PyTorch (runpod/pytorch)
- Volume: 50GB at `/workspace`

### 2. Setup and train
```bash
cd /workspace

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone and install
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME
cd YOUR_REPO_NAME
uv pip install --system -e .

# Login to HF
hf auth login

# Start 6h safety timer + train + auto-stop
bash -c "nohup sleep 6h; runpodctl stop pod $RUNPOD_POD_ID" &
python train.py --load-in-16bit --no-thinking && runpodctl stop pod $RUNPOD_POD_ID
```

### 3. Upload (after pod stops)
```bash
# Start the pod again from dashboard, then:
cd /workspace/YOUR_REPO_NAME
python upload_model.py --repo YOUR_USERNAME/My-Danish-Model

# Terminate pod after upload is done
```

Weights are saved to `/workspace/YOUR_REPO_NAME/outputs/merged_model/`.

## Serverless Endpoint

The `endpoint/` folder contains a RunPod serverless endpoint with an OpenAI-compatible chat completions API, powered by vLLM.

### Build and push
```bash
cd endpoint
docker build -t your-registry/llm-endpoint:latest .
docker push your-registry/llm-endpoint:latest
```

### Environment variables
| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `unsloth/Qwen3.5-9B` | HF model or path to upload |
| `HF_TOKEN` | | Hugging Face token (for gated models) |
| `MAX_MODEL_LEN` | `4096` | Max context length |
| `GPU_MEMORY_UTILIZATION` | `0.9` | vLLM GPU memory fraction |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism |

### Request format (OpenAI-compatible)
```json
{
  "input": {
    "messages": [
      {"role": "system", "content": "Du er en hjælpsom assistent."},
      {"role": "user", "content": "Hvad er koldskål?"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }
}
```

## License

Apache 2.0
