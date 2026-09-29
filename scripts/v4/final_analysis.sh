#!/usr/bin/env bash
# v4 の最終解析を一括で行う: GCS からストア・系統の state.json・パイロットの集計を取得し、
# 事前登録の検定（stats.py）、副次解析（secondary.py）、第5章の図（figures.py）と表（tables.py）を作る。
# 使い方: scripts/v4/final_analysis.sh [--match-k 4,7]
# 出力: results/v4/{stats_final,secondary}.json、configs/v4/stats_final.json、thesis/fig/*.pdf、thesis/tab/*.tex
set -euo pipefail
cd "$(dirname "$0")/../.."
export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-$HOME/.config/gcloud-evo-swarm-lora}"
G=gs://evo-swarm-lora-v4-pro-plasma-510112-m7
MATCH_K=""
if [ "${1:-}" = "--match-k" ]; then MATCH_K="$2"; fi

mkdir -p results/v4/store_mirror results/v4/lineages
# ストア: 書き込み中のファイルは cp だと世代の不一致で止まるため、1ファイルずつ cat で取る
for f in $(gcloud storage ls "$G/v4/store/calls_*.jsonl"); do
  b=$(basename "$f")
  timeout 600 gcloud storage cat "$f" > "results/v4/store_mirror/$b.part" && mv "results/v4/store_mirror/$b.part" "results/v4/store_mirror/$b"
done
STATES=()
for L in S N A1; do
  if gcloud storage cat "$G/v4/lineages/$L/state.json" > "results/v4/lineages/$L.state.json" 2>/dev/null; then
    STATES+=(--state "$L=results/v4/lineages/$L.state.json")
  else
    rm -f "results/v4/lineages/$L.state.json"
  fi
done
gcloud storage cat "$G/v4/pilot2/pilot_summary.json" > results/v4/pilot2_summary.json
gcloud storage cat "$G/v4/pilot2/personas_selected.json" > results/v4/personas_selected.json

python3 scripts/v4/make_stats_spec.py --personas results/v4/personas_selected.json "${STATES[@]}" \
  ${MATCH_K:+--match-k "$MATCH_K"} --out configs/v4/stats_final.json > /dev/null
uv run python scripts/v4/stats.py --spec configs/v4/stats_final.json --store results/v4/store_mirror \
  --out results/v4/stats_final.json > /dev/null
uv run python scripts/v4/secondary.py --spec configs/v4/stats_final.json --store results/v4/store_mirror \
  "${STATES[@]}" --out results/v4/secondary.json > /dev/null
uv run --with matplotlib python scripts/v4/figures.py --stats results/v4/stats_final.json \
  --secondary results/v4/secondary.json --pilot results/v4/pilot2_summary.json "${STATES[@]}" --out thesis/fig
python3 scripts/v4/tables.py --pilot results/v4/pilot2_summary.json --stats results/v4/stats_final.json --out thesis/tab
python3 - <<'EOF'
import json
d = json.load(open("results/v4/stats_final.json"))
for k, v in d["accuracy"].items():
    print(f"{k:14s} macro={100*v['macro']:.2f} n={v['n']}")
for k, v in d["comparisons"].items():
    h = d.get("holm_adjusted", {}).get(k)
    print(f"{k:28s} {100*v['diff_macro']:+.2f}pt [{100*v['ci95'][0]:+.2f}, {100*v['ci95'][1]:+.2f}] p={v['p']:.4g}"
          + (f" holm={h:.4g}" if h is not None else ""))
EOF
