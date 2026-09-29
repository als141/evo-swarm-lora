#!/usr/bin/env bash
# 同じジョブを4経路（us-central1 の Flex-start と Spot、europe-west4 と asia-southeast1 の Spot）に投入する。
# 最初に起動した1本以外は race_monitor.sh が取り消す。標準出力の最終行は "region:id ..."（監視に渡す）。
# 使い方: scripts/v4/launch_race.sh <name> <code-version> <script> [args...]
set -euo pipefail
NAME="$1"; VER="$2"; SCRIPT="$3"; shift 3
DIR="$(cd "$(dirname "$0")" && pwd)"
IDS=""
for spec in "us-central1:FLEX_START" "us-central1:SPOT" "europe-west4:SPOT" "asia-southeast1:SPOT"; do
  R=${spec%%:*}; S=${spec##*:}
  out=$(EVO4_REGION=$R EVO4_SCHEDULING=$S "$DIR/submit.sh" "${NAME}-$(echo $S | tr 'A-Z_' 'a-z-')-$R" "$VER" "$SCRIPT" "$@" 2>&1 \
        | grep -oE "customJobs/[0-9]+" | head -1)
  IDS="$IDS $R:${out##*/}"
done
echo "$IDS"
