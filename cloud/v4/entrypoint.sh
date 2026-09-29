#!/usr/bin/env bash
# v4 共通エントリポイント。
#   1) CODE_DIR（例: /gcs/<bucket>/code/<git-sha>）からコードを取得（イメージを再ビルドせずにコードだけ差し替える）
#   2) CUDA forward-compat を有効化し、GPU とイメージ情報をログに残す
#   3) START_VLLM=1 のとき vLLM サーバを起動してヘルスチェックを待つ
#   4) 引数のドライバを実行する
set -euo pipefail

if [ -n "${CODE_DIR:-}" ]; then
  echo "[entrypoint] fetching code from ${CODE_DIR}"
  rm -rf /workspace/code && mkdir -p /workspace/code
  cp -r "${CODE_DIR}/." /workspace/code/
  cat /workspace/code/CODE_VERSION 2>/dev/null || true
fi
cd /workspace/code 2>/dev/null || cd /workspace
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

for compat_dir in /usr/local/cuda/compat /usr/local/cuda-*/compat; do
  if [ -d "${compat_dir}" ]; then
    export LD_LIBRARY_PATH="${compat_dir}:${LD_LIBRARY_PATH:-}"
    echo "[entrypoint] enabled CUDA compat libs: ${compat_dir}"
    break
  fi
done
echo "[entrypoint] nvidia-smi:"; nvidia-smi || echo "[entrypoint] WARNING: nvidia-smi failed"
echo "[entrypoint] image pip freeze sha256: $(sha256sum /opt/evo/pip_freeze.txt | cut -c1-16)"

VLLM_PID=""
cleanup() {
  if [ -n "${VLLM_PID}" ]; then
    echo "[entrypoint] stopping vLLM (pid=${VLLM_PID})"; kill "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [ "${START_VLLM:-0}" = "1" ]; then
  MODEL="${VLLM_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
  PORT="${VLLM_PORT:-8000}"
  export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
  echo "[entrypoint] starting vLLM: model=${MODEL}"
  python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" \
    --dtype "${VLLM_DTYPE:-bfloat16}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS:-256}" \
    --enable-lora \
    --max-loras "${VLLM_MAX_LORAS:-8}" \
    --max-lora-rank "${VLLM_MAX_LORA_RANK:-32}" \
    ${VLLM_EXTRA_ARGS:-} \
    > /tmp/vllm.log 2>&1 &
  VLLM_PID=$!
  for i in $(seq 1 180); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
      echo "[entrypoint] vLLM healthy after ~${i}0s"; break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "[entrypoint] vLLM died:"; tail -80 /tmp/vllm.log; exit 1
    fi
    if [ "$i" -eq 180 ]; then echo "[entrypoint] vLLM not healthy in 30min"; tail -80 /tmp/vllm.log; exit 1; fi
    sleep 10
  done
  # vLLM ログを出力先へ定期退避（障害解析用）
  if [ -n "${VLLM_LOG_COPY_TO:-}" ]; then
    ( while kill -0 "${VLLM_PID}" 2>/dev/null; do cp /tmp/vllm.log "${VLLM_LOG_COPY_TO}" 2>/dev/null || true; sleep 300; done ) &
  fi
fi

echo "[entrypoint] running: $*"
set +e
"$@"
RC=$?
set -e
if [ -n "${VLLM_LOG_COPY_TO:-}" ] && [ -f /tmp/vllm.log ]; then cp /tmp/vllm.log "${VLLM_LOG_COPY_TO}" || true; fi
echo "[entrypoint] driver exit code ${RC}"
exit "${RC}"
