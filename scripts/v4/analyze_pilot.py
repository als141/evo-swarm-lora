"""パイロットの補足解析（保存済みの生出力をローカルで採点する。GPU・課金なし）。

ジョブ内の要約（pilot_summary.json）に含まれない次を計算する:
- 実効スループット（同時実行が最大の 80% 以上だった区間の出力 tok/s）、呼び出し種別ごとの出力長
- 反復ループの早期打ち切り率と、打ち切られた出力の内訳
- 議論プロトコルごとの同調（round0 で少数派だったエージェントが round1 で多数派に合わせた率。自分が正解の場合と誤答の場合）

使い方:
  gcloud storage cp -r gs://<bucket>/v4/store results/v4/store_mirror
  gcloud storage cp gs://<bucket>/v4/pilot/pilot_summary.json results/v4/
  python scripts/v4/analyze_pilot.py --store results/v4/store_mirror --summary results/v4/pilot_summary.json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evo4.items import index_items  # noqa: E402
from src.evo4.scoring import extract_answer, is_correct, require_math_verify  # noqa: E402
from src.evo4.store import CallStore  # noqa: E402


def steady_throughput(records: list) -> float:
    events = [(r["t_start"], r["t_start"] + r["elapsed"], r.get("n_out") or 0) for r in records if r.get("t_start")]
    if not events:
        return 0.0
    t0 = min(e[0] for e in events)
    bins = collections.defaultdict(lambda: [0, 0.0])
    for start, end, n in events:
        dur = max(end - start, 1e-3)
        for sec in range(int(start - t0), int(end - t0) + 1):
            bins[sec][0] += 1
            bins[sec][1] += n / dur
    peak = max(v[0] for v in bins.values())
    steady = [v[1] for v in bins.values() if v[0] >= 0.8 * peak]
    return statistics.mean(steady) if steady else 0.0


def main() -> None:
    require_math_verify()  # 手元で採点するとき math-verify が無いと MATH を過小評価する
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--items", default=str(ROOT / "data/v4/items/dev.jsonl"))
    args = parser.parse_args()
    items = index_items([args.items])
    summary = json.loads(Path(args.summary).read_text())
    store = CallStore(tempfile.mkdtemp(prefix="evo4_analyze_"), args.store, resolve="earliest")
    recs = [r for r in store.records() if r.get("item") in items]
    out = {"n_records": len(recs)}

    kinds = collections.defaultdict(list)
    for r in recs:
        kinds[r["kind"]].append(r)
    out["finish_counts"] = dict(collections.Counter(r.get("finish") for r in recs))
    out["loop_abort_rate"] = round(sum(r.get("finish") == "loop_abort" for r in recs) / max(1, len(recs)), 4)
    out["length_hit_rate"] = round(sum(r.get("finish") == "length" for r in recs) / max(1, len(recs)), 4)
    out["steady_out_tok_per_s"] = {k: round(steady_throughput(v), 1) for k, v in kinds.items()}
    out["mean_out_tokens"] = {k: round(statistics.mean((r.get("n_out") or 0) for r in v), 1) for k, v in kinds.items()}
    out["sweep"] = summary.get("sweep")

    def answer(r):
        return extract_answer(r["text"], items[r["item"]].answer_type)

    # 同調率（既定トリオ、プロトコル別）
    trio = [f"p.{n}" for n in json.loads((ROOT / "configs/v4/persona_pool.json").read_text())["default_trio"]]
    r0 = {(r["agent"], r["item"]): r for r in kinds["r0"]}
    conform = {}
    for protocol in ("v4", "v4c", "july"):
        table = {}
        for r in kinds["r1"]:
            if r.get("protocol") == protocol and r["agent"] in trio and sorted(r["others"]) == sorted(
                    a for a in trio if a != r["agent"]):
                table[(r["agent"], r["item"])] = r
        right, wrong, keep_majority = [0, 0], [0, 0], [0, 0]
        for item_id in {i for (_, i) in table}:
            if not all((a, item_id) in table and (a, item_id) in r0 for a in trio):
                continue
            item = items[item_id]
            a0 = [answer(r0[(a, item_id)]) for a in trio]
            a1 = [answer(table[(a, item_id)]) for a in trio]
            for k in range(3):
                others = [a0[j] for j in range(3) if j != k]
                if others[0] is not None and others[0] == others[1] and a0[k] != others[0]:
                    bucket = right if is_correct(a0[k], item.gold, item.answer_type) else wrong
                    bucket[0] += a1[k] == others[0]
                    bucket[1] += 1
                elif others[0] is not None and others[0] == a0[k]:
                    keep_majority[0] += a1[k] == a0[k]
                    keep_majority[1] += 1
        conform[protocol] = {
            "minority_correct_switch_rate": round(right[0] / max(1, right[1]), 3), "n_minority_correct": right[1],
            "minority_wrong_switch_rate": round(wrong[0] / max(1, wrong[1]), 3), "n_minority_wrong": wrong[1],
            "majority_keep_rate": round(keep_majority[0] / max(1, keep_majority[1]), 3),
        }
    out["conformity_default_trio"] = conform
    print(json.dumps(out, ensure_ascii=False, indent=1))
    target = ROOT / "results/v4/pilot_analysis.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
