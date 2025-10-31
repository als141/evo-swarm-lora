FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates build-essential python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN uv python install 3.12

WORKDIR /workspace
COPY pyproject.toml README.md ./

RUN uv venv -p 3.12 && . .venv/bin/activate && uv sync --no-dev

RUN . .venv/bin/activate && pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124

RUN mkdir -p $HF_HOME

COPY src ./src
COPY data ./data
COPY scripts ./scripts

ENV PYTHONPATH=/workspace
CMD ["/bin/bash"]
