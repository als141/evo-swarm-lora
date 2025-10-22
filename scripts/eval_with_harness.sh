#!/usr/bin/env bash
set -euo pipefail

export HF_ALLOW_CODE_EVAL=1
TASKS="arc_easy,winogrande,hellaswag"
MODEL="openai-chat-completions"
MODEL_ARGS="model=Qwen/Qwen3-4B-Instruct-2507:persona_a,base_url=http://localhost:8000/v1/chat/completions"

python -m lm_eval \
  --model "${MODEL}" \
  --model_args "${MODEL_ARGS}" \
  --tasks "${TASKS}" \
  --batch_size auto \
  --apply_chat_template \
  --output_path out/persona_a_eval.json
