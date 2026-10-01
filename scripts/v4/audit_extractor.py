"""v4 採点器の監査: 新環境の全 LLM 呼び出し（生出力）に v4 抽出器を適用し、
2026-07 修正版抽出器（calls_index の ans）との差分と精度への影響を集計する。

入力: results/reanalysis_2026-09/cache/calls_index.jsonl.gz（再解析で作成した呼び出し索引）
      results/gcs/run002/*/llm_calls/*.jsonl.gz（生出力）
出力: results/v4/audit_extractor.json（集計）と、差分の実例（人手確認用）
"""

from __future__ import annotations

import collections
import gzip
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evo4.scoring import extract_answer, is_correct, scorer_info, require_math_verify  # noqa: E402

INDEX = ROOT / "results/reanalysis_2026-09/cache/calls_index.jsonl.gz"
RAW_ROOT = ROOT / "results/gcs/run002"
OUT = ROOT / "results/v4/audit_extractor.json"
ANSWER_TYPE = {"mmlu_pro": "letter", "supergpqa": "letter", "math500": "math"}


def _raw_math_gold() -> dict:
    """MATH-500 の元の LaTeX 正解（calls_index の gold は 2026-07 の正規化済み形式のため）。"""
    from datasets import load_dataset

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return {f"math500-test-{idx}": row["answer"] for idx, row in enumerate(dataset)}


def main() -> None:
    require_math_verify()  # 手元で採点するとき math-verify が無いと MATH を過小評価する
    rows = [json.loads(line) for line in gzip.open(INDEX, "rt")]
    raw_gold = _raw_math_gold()
    by_file = collections.defaultdict(list)
    for row in rows:
        if row.get("kind") == "judge":  # GenSelect 裁定呼び出しは回答抽出の対象外
            continue
        by_file[(row["dir"], row["file"])].append(row)

    stats = collections.defaultdict(lambda: collections.Counter())
    examples = collections.defaultdict(list)
    for (dirname, filename), group in sorted(by_file.items()):
        path = RAW_ROOT / dirname / "llm_calls" / filename
        lines = gzip.open(path, "rt").read().splitlines()
        for row in group:
            record = json.loads(lines[row["line"]])
            bench = row["bench"]
            answer_type = ANSWER_TYPE[bench]
            new = extract_answer(record["response"], answer_type)
            old = row["ans"]
            gold = raw_gold[row["item_id"]] if bench == "math500" else row["gold"]
            ok_new = is_correct(new, gold, answer_type)
            ok_old = bool(row["ok"])
            key = f"{bench}|{row['family']}|{row['kind']}|r{row['round']}"
            c = stats[key]
            c["n"] += 1
            c["ok_old"] += ok_old
            c["ok_new"] += ok_new
            c["none_old"] += old is None
            c["none_new"] += new is None
            if (old or "") != (new or ""):
                c["diff"] += 1
                c["flip_to_correct"] += (not ok_old) and ok_new
                c["flip_to_wrong"] += ok_old and (not ok_new)
                if len(examples[key]) < 400:
                    examples[key].append(
                        {"item_id": row["item_id"], "gold": gold, "old": old, "new": new,
                         "tail": record["response"][-300:]}
                    )

    summary = {}
    for key, c in sorted(stats.items()):
        n = c["n"]
        summary[key] = {
            "n": n,
            "acc_old": round(c["ok_old"] / n, 4),
            "acc_new": round(c["ok_new"] / n, 4),
            "diff_rate": round(c["diff"] / n, 4),
            "none_old": round(c["none_old"] / n, 4),
            "none_new": round(c["none_new"] / n, 4),
            "flip_to_correct": c["flip_to_correct"],
            "flip_to_wrong": c["flip_to_wrong"],
        }
    random.seed(0)
    sampled = {k: random.sample(v, min(12, len(v))) for k, v in examples.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"scorer": scorer_info(), "summary": summary, "examples": sampled},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    for key, s in summary.items():
        print(f"{key:40s} n={s['n']:6d} acc {s['acc_old']:.4f}->{s['acc_new']:.4f} "
              f"diff={s['diff_rate']:.4f} none {s['none_old']:.3f}->{s['none_new']:.3f} "
              f"+{s['flip_to_correct']} -{s['flip_to_wrong']}")


if __name__ == "__main__":
    main()
