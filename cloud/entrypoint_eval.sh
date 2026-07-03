#!/usr/bin/env bash
# vLLM サーバをバックグラウンド起動し、ヘルスチェック通過後に引数のドライバを実行する。
# 例: entrypoint_eval.sh python scripts/run_evolution.py --base-url http://localhost:8000/v1 ...
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-32}"
MAX_LORAS="${VLLM_MAX_LORAS:-8}"
DTYPE="${VLLM_DTYPE:-bfloat16}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

echo "[entrypoint] starting vLLM: model=${MODEL} port=${PORT} dtype=${DTYPE}"
python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --enable-lora \
  --max-loras "${MAX_LORAS}" \
  --max-lora-rank "${MAX_LORA_RANK}" \
  ${VLLM_EXTRA_ARGS:-} \
  > /tmp/vllm.log 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "[entrypoint] stopping vLLM (pid=${VLLM_PID})"
  kill "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[entrypoint] waiting for vLLM health..."
for i in $(seq 1 180); do
  if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "[entrypoint] vLLM is healthy (after ${i}0s)"
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[entrypoint] vLLM process died. tail of log:"
    tail -50 /tmp/vllm.log
    exit 1
  fi
  if [ "$i" -eq 180 ]; then
    echo "[entrypoint] vLLM did not become healthy in 30min. tail of log:"
    tail -50 /tmp/vllm.log
    exit 1
  fi
  sleep 10
done

echo "[entrypoint] running driver: $*"
"$@"
DRIVER_EXIT=$?
echo "[entrypoint] driver finished with exit code ${DRIVER_EXIT}"
exit "${DRIVER_EXIT}"
