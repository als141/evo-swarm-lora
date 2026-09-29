# evo-swarm-lora v4 実験環境イメージ（環境のみ。コードはジョブ起動時に GCS から取得する）
# - 推論: vLLM v0.8.5（Qwen3-4B-Instruct-2507 対応の既定版）。推論経路パッケージは constraints.txt で固定。
# - 学習: bf16 LoRA（peft のみ追加。QLoRA/bitsandbytes/TRL は使わない）。
# - 実験期間中はこのイメージを digest 固定で使い、再ビルドしない（2026-07 の +6pt 系統差の教訓）。
FROM vllm/vllm-openai:v0.8.5

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY cloud/v4/constraints.txt /opt/evo/constraints.txt
RUN pip install --no-cache-dir -c /opt/evo/constraints.txt "peft==0.15.2" "datasets==3.6.0" \
    && pip freeze > /opt/evo/pip_freeze.txt \
    && python3 -c "import torch, transformers, vllm, peft, datasets; print('torch', torch.__version__, 'transformers', transformers.__version__, 'vllm', vllm.__version__, 'peft', peft.__version__, 'datasets', datasets.__version__)"

# Vertex ホストの GPU ドライバが古い場合（CUDA 12.2 相当）に備えた forward-compat ライブラリ
RUN apt-get update && apt-get install -y --no-install-recommends cuda-compat-12-4 \
    && dpkg -l | grep cuda-compat > /opt/evo/cuda_compat_version.txt \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY cloud/v4/entrypoint.sh /opt/evo/entrypoint.sh
RUN chmod +x /opt/evo/entrypoint.sh
ENTRYPOINT ["/opt/evo/entrypoint.sh"]
