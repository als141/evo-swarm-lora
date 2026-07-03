#!/usr/bin/env bash
# Vertex AI Custom Job (Spot) を投入する共通スクリプト。
#
# 使用例:
#   学習 (T4):
#     scripts/cloud/submit_job.sh train run001
#   進化 (A100):
#     scripts/cloud/submit_job.sh evolution run001 [追加ドライバ引数...]
#   評価 (A100):
#     scripts/cloud/submit_job.sh eval run001 --task mmlu_pro --n 500 --mode team \
#       --agents critic=gen6_critic pragmatist=gen6_pragmatist explorer=gen6_explorer ...
#
# 前提: CLOUDSDK_ACTIVE_CONFIG_NAME=evo-swarm、イメージは cloud/cloudbuild.yaml でビルド済み。
set -euo pipefail

PROJECT="research-501308"
REGION="us-central1"
# ジョブと同リージョン（us-central1）のバケットが必須（Vertexの制約）
BUCKET="gs://evo-swarm-lora-usc1-research-501308"
IMAGE_PREFIX="us-central1-docker.pkg.dev/${PROJECT}/evo-swarm"

JOB_TYPE="${1:?usage: submit_job.sh <train|evolution|eval> <run_id> [driver args...]}"
RUN_ID="${2:?run_id required}"
shift 2
EXTRA_ARGS=("$@")

RUN_URI="${BUCKET}/experiments/${RUN_ID}"
GCS_MOUNT="/gcs/evo-swarm-lora-usc1-research-501308/experiments/${RUN_ID}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
CONFIG_FILE=$(mktemp /tmp/vertex-job-XXXX.yaml)

case "${JOB_TYPE}" in
  train)
    DISPLAY_NAME="train-personas-${RUN_ID}-${TIMESTAMP}"
    cat > "${CONFIG_FILE}" <<EOF
workerPoolSpecs:
  - machineSpec:
      machineType: n1-highmem-8
      acceleratorType: NVIDIA_TESLA_T4
      acceleratorCount: 1
    replicaCount: 1
    diskSpec:
      bootDiskSizeGb: 200
    containerSpec:
      imageUri: ${IMAGE_PREFIX}/trainer:latest
      command: [python, scripts/train_all_personas_vertex.py]
      args:
        - --seed=1234
        - --rank=32
        - --epochs=3
        - --upload-uri=
        - --notes=run ${RUN_ID}
$(for arg in "${EXTRA_ARGS[@]}"; do echo "        - ${arg}"; done)
scheduling:
  strategy: SPOT
baseOutputDirectory:
  outputUriPrefix: ${RUN_URI}/training
EOF
    ;;
  evolution)
    DISPLAY_NAME="evolution-${RUN_ID}-${TIMESTAMP}"
    cat > "${CONFIG_FILE}" <<EOF
workerPoolSpecs:
  - machineSpec:
      machineType: a2-highgpu-1g
      acceleratorType: NVIDIA_TESLA_A100
      acceleratorCount: 1
    replicaCount: 1
    diskSpec:
      bootDiskSizeGb: 200
    containerSpec:
      imageUri: ${IMAGE_PREFIX}/eval:latest
      args:
        - python3
        - scripts/run_evolution.py
        - --base-url=http://localhost:8000/v1
        - --gen0
        - critic=${GCS_MOUNT}/training/model/adapters/persona_a
        - pragmatist=${GCS_MOUNT}/training/model/adapters/persona_b
        - explorer=${GCS_MOUNT}/training/model/adapters/persona_c
        - --out-root=${GCS_MOUNT}/evolution/adapters
        - --log=${GCS_MOUNT}/evolution/run_log.json
$(for arg in "${EXTRA_ARGS[@]}"; do echo "        - ${arg}"; done)
scheduling:
  strategy: SPOT
  restartJobOnWorkerRestart: true
baseOutputDirectory:
  outputUriPrefix: ${RUN_URI}/evolution
EOF
    ;;
  battery)
    # 使用例: submit_job.sh battery run001 <バッテリー設定のGCSマウントパス>
    CONFIG_PATH="${EXTRA_ARGS[0]:?battery config path (GCS mount) required}"
    DISPLAY_NAME="battery-${RUN_ID}-${TIMESTAMP}"
    cat > "${CONFIG_FILE}" <<EOF
workerPoolSpecs:
  - machineSpec:
      machineType: a2-highgpu-1g
      acceleratorType: NVIDIA_TESLA_A100
      acceleratorCount: 1
    replicaCount: 1
    diskSpec:
      bootDiskSizeGb: 200
    containerSpec:
      imageUri: ${IMAGE_PREFIX}/eval:latest
      args:
        - python3
        - scripts/run_eval_battery.py
        - --base-url=http://localhost:8000/v1
        - --config=${CONFIG_PATH}
        - --out-dir=${GCS_MOUNT}/final_eval
scheduling:
  strategy: SPOT
baseOutputDirectory:
  outputUriPrefix: ${RUN_URI}/final_eval_job
EOF
    ;;
  eval)
    DISPLAY_NAME="eval-${RUN_ID}-${TIMESTAMP}"
    cat > "${CONFIG_FILE}" <<EOF
workerPoolSpecs:
  - machineSpec:
      machineType: a2-highgpu-1g
      acceleratorType: NVIDIA_TESLA_A100
      acceleratorCount: 1
    replicaCount: 1
    diskSpec:
      bootDiskSizeGb: 200
    containerSpec:
      imageUri: ${IMAGE_PREFIX}/eval:latest
      args:
        - python3
        - scripts/run_eval.py
        - --base-url=http://localhost:8000/v1
$(for arg in "${EXTRA_ARGS[@]}"; do echo "        - ${arg}"; done)
scheduling:
  strategy: SPOT
baseOutputDirectory:
  outputUriPrefix: ${RUN_URI}/eval
EOF
    ;;
  *)
    echo "Unknown job type: ${JOB_TYPE}" >&2
    exit 1
    ;;
esac

echo "=== job config (${CONFIG_FILE}) ==="
cat "${CONFIG_FILE}"
gcloud ai custom-jobs create \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --display-name="${DISPLAY_NAME}" \
  --config="${CONFIG_FILE}"
