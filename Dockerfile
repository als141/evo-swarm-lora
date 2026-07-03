FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates build-essential python3 python3-venv python3-pip python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN uv python install 3.12

WORKDIR /workspace
COPY pyproject.toml README.md ./

RUN uv venv -p 3.12 && . .venv/bin/activate && uv sync --no-dev
ENV PATH="/workspace/.venv/bin:${PATH}"

# Vertex AI の GPU ホストはドライバが CUDA 12.2 相当（535系）のため、
# cu124 ではなく cu118 ビルドを使う（12.2 >= 11.8 でネイティブ動作）
RUN . .venv/bin/activate && pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118

RUN mkdir -p $HF_HOME

COPY src ./src
COPY data ./data
COPY scripts ./scripts

ENV PYTHONPATH=/workspace
CMD ["/bin/bash"]
