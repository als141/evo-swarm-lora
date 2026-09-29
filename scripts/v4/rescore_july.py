"""2026-07 の新環境データ（run002 系）を v4 採点器で採点し直し、修論の予備実験の数値を確定する。

対象: c1（ベース単体）、c2（SC@9 の各サンプル → SC@k 曲線）、c5（旧チーム）、c7（再学習チーム）。
チームの最終回答は 7 月と同じ集約で再現する（c7: tail logprob 重み付き投票、c5: 多数決）。
7 月の集約の同数決着は乱数だったため、ここでは確信度→役割順で決定的に決める（差は同数の問題だけ）。
比較は問題 ID クラスタの符号反転置換検定とブートストラップ（macro＝ベンチ等重み）。
"""

from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts/v4"))

from src.evo4.scoring import extract_answer, is_correct  # noqa: E402
from stats import paired_test  # noqa: E402

INDEX = ROOT / "results/reanalysis_2026-09/cache/calls_index.jsonl.gz"
RAW = ROOT / "results/gcs/run002"
OUT = ROOT / "results/v4/rescore_july.json"
NEW_ENV_DIRS = {"remeasure_v1", "robust_c2", "robust_c7", "final_c7", "g3_team_check", "recheck_c2s1"}
BENCH = {"mmlu_pro": "mmlu_pro", "supergpqa": "supergpqa", "math500": "math"}


def raw_math_gold() -> dict:
    from datasets import load_dataset

    return {f"math500-test-{i}": r["answer"] for i, r in enumerate(load_dataset("HuggingFaceH4/MATH-500", split="test"))}


def main() -> None:
    gold_math = raw_math_gold()
    by_file = collections.defaultdict(list)
    for line in gzip.open(INDEX, "rt"):
        row = json.loads(line)
        if row["dir"] in NEW_ENV_DIRS and row.get("kind") in ("sc", "solo", "team"):
            by_file[(row["dir"], row["file"])].append(row)
    sc = collections.defaultdict(dict)       # (seed, item) -> {k: (ans, ok)}
    solo = {}                                 # (seed, item) -> ok
    team = collections.defaultdict(lambda: collections.defaultdict(dict))  # fam -> (seed,item) -> agent -> (ans, conf, round)
    gold_of, bench_of = {}, {}
    for (dirname, filename), rows in by_file.items():
        lines = gzip.open(RAW / dirname / "llm_calls" / filename, "rt").read().splitlines()
        for row in rows:
            rec = json.loads(lines[row["line"]])
            bench = BENCH[row["bench"]]
            atype = "math" if bench == "math" else "letter"
            gold = gold_math[row["item_id"]] if bench == "math" else row["gold"]
            gold_of[row["item_id"]] = (gold, atype)
            bench_of[row["item_id"]] = bench
            ans = extract_answer(rec["response"], atype)
            ok = is_correct(ans, gold, atype)
            key = (row["run_seed"], row["item_id"])
            if row["kind"] == "sc" and row["family"] == "base":
                sc[key][row["sample_idx"]] = (ans, ok)
            elif row["kind"] == "solo" and row["family"] == "base":
                solo[key] = ok
            elif row["kind"] == "team" and row["family"] in ("c5", "c7") and row["round"] == 1:
                team[row["family"]][key][row["agent_idx"]] = (ans, rec.get("tail_confidence"))

    def vote(votes, weighted):
        w = collections.defaultdict(float)
        best = {}
        for ans, conf in votes:
            if ans is None:
                continue
            w[ans] += (conf if (weighted and conf is not None) else 1.0)
            best[ans] = max(best.get(ans, -1.0), conf if conf is not None else -1.0)
        if not w:
            return None
        top = max(w.values())
        tied = [a for a in w if abs(w[a] - top) < 1e-12]
        return sorted(tied, key=lambda a: (-best[a], a))[0]

    # 問題ごと（seed 平均）の正答
    def per_item(cells: dict) -> dict:
        acc = collections.defaultdict(list)
        for (seed, item), ok in cells.items():
            acc[item].append(float(ok))
        return {i: float(np.mean(v)) for i, v in acc.items()}

    conditions = {}
    conditions["c1_base_solo"] = per_item(solo)
    sc_cells = {}
    for k in (1, 3, 6, 9):
        cells = {}
        for key, samples in sc.items():
            if len(samples) < 9:
                continue
            ordered = [samples[j] for j in range(9)]
            if k == 1:
                cells[key] = float(np.mean([o for _, o in ordered]))  # 単発の期待正答（9サンプル平均）
            else:
                final = vote([(a, None) for a, _ in ordered[:k]], weighted=False)
                cells[key] = is_correct(final, *gold_of[key[1]])
        sc_cells[k] = cells
        conditions[f"sc{k}" if k > 1 else "base_single_avg9"] = per_item(cells)
    for fam, weighted in (("c7", True), ("c5", False)):
        cells = {}
        for key, agents in team[fam].items():
            if len(agents) < 3:
                continue
            final = vote([agents[i] for i in sorted(agents)], weighted=weighted)
            cells[key] = is_correct(final, *gold_of[key[1]])
        conditions[f"{fam}_team"] = per_item(cells)

    def macro(values: dict) -> dict:
        by = collections.defaultdict(list)
        for i, v in values.items():
            by[bench_of[i]].append(v)
        res = {b: round(float(np.mean(v)), 4) for b, v in by.items()}
        res["macro"] = round(float(np.mean(list(res.values()))), 4)
        res["n_items"] = len(values)
        return res

    out = {"accuracy": {name: macro(v) for name, v in conditions.items()}, "comparisons": {}}
    pairs = [("c7_team", "sc9"), ("c7_team", "sc6"), ("c7_team", "sc3"), ("c7_team", "c1_base_solo"),
             ("c7_team", "c5_team"), ("c5_team", "c1_base_solo"), ("sc9", "c1_base_solo"),
             ("c7_team", "base_single_avg9")]
    for a, b in pairs:
        out["comparisons"][f"{a} vs {b}"] = paired_test(conditions[a], conditions[b], bench_of)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for name, acc in out["accuracy"].items():
        print(f"{name:18s} {acc}")
    for name, r in out["comparisons"].items():
        print(f"{name:32s} diff={r['diff_macro']*100:+.2f}pt CI95=[{r['ci95'][0]*100:+.2f},{r['ci95'][1]*100:+.2f}] "
              f"p={r['p']:.4f} n={r['n']} per_bench={ {k: round(v*100, 2) for k, v in r['per_bench'].items()} }")


if __name__ == "__main__":
    main()
