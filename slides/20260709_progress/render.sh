#!/bin/bash
# pptx -> PDF -> ページ別PNG。引数: pptxパス, 出力プレフィックス
set -e
export HOME=/home/als0028
SOFFICE=~/apps/squashfs-root/opt/libreoffice26.2/program/soffice
PPTX="$1"
OUTDIR="${2:-/tmp/claude-1000/-home-als0028-study-research-evo-swarm-lora/132de56c-1ddc-424a-86ce-ae486acc7b2e/scratchpad/render}"
mkdir -p "$OUTDIR"
DIR=$(dirname "$PPTX")
BASE=$(basename "$PPTX" .pptx)
timeout 180 $SOFFICE --headless --convert-to pdf --outdir "$DIR" "$PPTX" >/dev/null 2>&1
PDF="$DIR/$BASE.pdf"
# pdftoppm があればページ別PNG
if command -v pdftoppm >/dev/null 2>&1; then
  pdftoppm -r 90 -png "$PDF" "$OUTDIR/page" >/dev/null 2>&1
else
  echo "pdftoppm not found; using soffice per-page is unavailable. PDF at $PDF"
fi
echo "PDF: $PDF"
ls "$OUTDIR"/page*.png 2>/dev/null | sort -V
