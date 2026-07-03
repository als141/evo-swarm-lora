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
# cu124 ではなく cu118 ビルドを使う（12.2 >= 11.8 でネイティブ動作）。
# 注意: uv の venv には pip が無く、素の `pip install` はシステム Python 3.10 に
# 入ってしまう（過去の事故原因）。必ず `uv pip` で venv を対象にする。
RUN uv pip install --python .venv/bin/python --no-cache \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118

# ビルド時にジョブ実行系（venv）の torch が cu118 であることを保証する
RUN .venv/bin/python -c "import torch; v=torch.__version__; print('venv torch:', v); assert '+cu118' in v, v"

RUN mkdir -p $HF_HOME

COPY src ./src
COPY data ./data
COPY scripts ./scripts

ENV PYTHONPATH=/workspace
CMD ["/bin/bash"]
