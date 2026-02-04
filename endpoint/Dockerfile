FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL_NAME="unsloth/Mistral-Nemo-Instruct-2407"
ENV HF_TOKEN=""
ENV MAX_MODEL_LEN=4096
ENV GPU_MEMORY_UTILIZATION=0.9
ENV TENSOR_PARALLEL_SIZE=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    vllm \
    runpod \
    huggingface_hub

COPY handler.py /handler.py

CMD ["python3", "/handler.py"]
