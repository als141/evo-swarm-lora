#!/usr/bin/env bash
# v4 ジョブの入口（イメージの ENTRYPOINT を置き換える。コードスナップショットの一部）。
# - CODE_DIR（GCS FUSE）からコードを取得
# - CUDA は互換ライブラリなしでまず試し、使えない場合だけ forward-compat を有効化する
#   （2026-09 時点の Vertex ホストはドライバ 580 系。7 月（535 系）向けの compat 12.4 を強制すると
#     Error 803 で CUDA が使えなくなるため）
set -euo pipefail
rm -rf /workspace/code && mkdir -p /workspace/code
cp -r "${CODE_DIR:?CODE_DIR required}/." /workspace/code/
cd /workspace/code
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
cat CODE_VERSION 2>/dev/null || true
head -1 /proc/driver/nvidia/version 2>/dev/null || echo "[job_entry] no nvidia driver"
if python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[job_entry] CUDA available without compat libs"
else
  for d in /usr/local/cuda/compat /usr/local/cuda-*/compat; do
    if [ -d "$d" ]; then export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; echo "[job_entry] enabling compat $d"; break; fi
  done
  python3 -c "import torch; print('[job_entry] CUDA with compat:', torch.cuda.is_available())" || true
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || true
echo "[job_entry] running: $*"
exec "$@"
