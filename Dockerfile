FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates build-essential python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace
COPY pyproject.toml README.md ./

RUN uv venv && . .venv/bin/activate && uv sync --no-dev

ENV HF_HOME=/workspace/.cache/huggingface
RUN mkdir -p $HF_HOME

COPY src ./src
COPY data ./data
COPY scripts ./scripts

ENV PYTHONPATH=/workspace
CMD ["/bin/bash"]
