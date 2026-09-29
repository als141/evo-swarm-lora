#!/usr/bin/env bash
# v4 のコードスナップショットを GCS に置く（イメージを再ビルドせずにコードだけ差し替えるため）。
# 出力: 標準出力の最終行にバージョン名（ジョブの CODE_DIR=/gcs/<bucket>/code/<version> に使う）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUCKET="${EVO4_BUCKET:-evo-swarm-lora-v4-pro-plasma-510112-m7}"
cd "$ROOT"
HASH=$( (git rev-parse --short HEAD; git diff HEAD -- src scripts; git ls-files --others --exclude-standard src scripts | xargs -r cat) | sha256sum | cut -c1-10)
VERSION="$(date +%Y%m%d-%H%M%S)-${HASH}"
TMP=$(mktemp -d)
mkdir -p "$TMP/snap"
rsync -a --exclude "__pycache__" --exclude "*.pyc" src scripts configs "$TMP/snap/"
mkdir -p "$TMP/snap/data/v4" && cp -r data/v4/items "$TMP/snap/data/v4/"
# 採点用の純 Python 依存（sympy はイメージ同梱の 1.13.1 を使うため入れない）
uv pip install --quiet --python 3.12 --target "$TMP/snap/vendor" --no-deps \
  "math-verify==0.8.0" "latex2sympy2-extended==1.10.2" "antlr4-python3-runtime==4.13.2" >/dev/null
{
  echo "version=${VERSION}"
  echo "git_head=$(git rev-parse HEAD)"
  echo "dirty_files=$(git status --porcelain | wc -l)"
  echo "created=$(date -Iseconds)"
} > "$TMP/snap/CODE_VERSION"
(cd "$TMP/snap" && gcloud storage cp -r ./* "gs://${BUCKET}/code/${VERSION}/" --quiet >/dev/null)
gcloud storage cp "$TMP/snap/CODE_VERSION" "gs://${BUCKET}/code/${VERSION}/CODE_VERSION" --quiet >/dev/null
rm -rf "$TMP"
echo "${VERSION}"
