#!/usr/bin/env bash
# v4 の Vertex AI カスタムジョブ（Spot A100）を投入する。イメージは digest 固定。
# 使い方: scripts/v4/submit.sh <display-name> <code-version> <python-script> [args...]
set -euo pipefail
PROJECT="pro-plasma-510112-m7"
REGION="${EVO4_REGION:-us-central1}"
BUCKET="evo-swarm-lora-v4-pro-plasma-510112-m7"
IMAGE="us-central1-docker.pkg.dev/${PROJECT}/evo-swarm/env-v4@sha256:41ab70107a39096377da1fef113d9ef9fbd5f1ce3b016cabfc979a2d13da8d71"
MACHINE="${EVO4_MACHINE:-a2-highgpu-1g}"
ACCEL="${EVO4_ACCEL:-NVIDIA_TESLA_A100}"
NAME="${1:?display name}"; VERSION="${2:?code version}"; SCRIPT="${3:?script}"; shift 3
CFG=$(mktemp /tmp/evo4-job-XXXX.yaml)
{
  echo "workerPoolSpecs:"
  echo "  - machineSpec:"
  echo "      machineType: ${MACHINE}"
  echo "      acceleratorType: ${ACCEL}"
  echo "      acceleratorCount: 1"
  echo "    replicaCount: 1"
  echo "    diskSpec:"
  echo "      bootDiskSizeGb: 200"
  echo "    containerSpec:"
  echo "      imageUri: ${IMAGE}"
  echo "      env:"
  echo "        - name: CODE_DIR"
  echo "          value: /gcs/${BUCKET}/code/${VERSION}"
  echo "        - name: HF_HOME"
  echo "          value: /tmp/hf"
  echo "        - name: EVO4_BUCKET_MOUNT"
  echo "          value: /gcs/${BUCKET}"
  echo "      command:"
  echo "        - bash"
  echo "        - /gcs/${BUCKET}/code/${VERSION}/scripts/v4/job_entry.sh"
  echo "      args:"
  echo "        - python3"
  echo "        - ${SCRIPT}"
  for a in "$@"; do echo "        - \"${a}\""; done
  echo "scheduling:"
  if [ "${EVO4_SCHEDULING:-SPOT}" = "FLEX_START" ]; then
    echo "  strategy: FLEX_START"
    echo "  maxWaitDuration: ${EVO4_MAX_WAIT:-43200s}"
  else
    echo "  strategy: SPOT"
  fi
  echo "  restartJobOnWorkerRestart: true"
} > "${CFG}"
cat "${CFG}"
gcloud ai custom-jobs create --project="${PROJECT}" --region="${REGION}" --display-name="${NAME}" --config="${CFG}"
