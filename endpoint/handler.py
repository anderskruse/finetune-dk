"""
RunPod serverless handler with OpenAI-compatible chat completions API.
Uses vLLM for fast inference.
"""

import os
import runpod
from vllm import LLM, SamplingParams
from vllm.entrypoints.chat_utils import apply_chat_template

MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/Mistral-Nemo-Instruct-2407")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"))
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))

print(f"Loading model: {MODEL_NAME}")
llm = LLM(
    model=MODEL_NAME,
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    trust_remote_code=True,
)
tokenizer = llm.get_tokenizer()
print("Model loaded.")


def handler(job):
    """
    OpenAI-compatible chat completions handler.

    Expected input format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": false
    }
    """
    input_data = job["input"]
    messages = input_data.get("messages", [])

    if not messages:
        return {"error": "No messages provided"}

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Sampling parameters
    sampling_params = SamplingParams(
        max_tokens=input_data.get("max_tokens", 512),
        temperature=input_data.get("temperature", 0.7),
        top_p=input_data.get("top_p", 0.9),
        stop=input_data.get("stop", None),
        presence_penalty=input_data.get("presence_penalty", 0.0),
        frequency_penalty=input_data.get("frequency_penalty", 0.0),
    )

    outputs = llm.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text

    # Return OpenAI-compatible response format
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text,
                },
                "finish_reason": "stop",
            }
        ],
        "model": MODEL_NAME,
        "usage": {
            "prompt_tokens": len(outputs[0].prompt_token_ids),
            "completion_tokens": len(outputs[0].outputs[0].token_ids),
            "total_tokens": len(outputs[0].prompt_token_ids) + len(outputs[0].outputs[0].token_ids),
        },
    }


runpod.serverless.start({"handler": handler})
