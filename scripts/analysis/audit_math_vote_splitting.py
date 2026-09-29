"""MATH-500 の SC@9 多数決における「同値だが表記違いの回答による票割れ」の監査。

保存済みの SC@9 は正規化文字列の完全一致で票を数えるため、`120` と `120^circ`、
`11sqrt(2)` と `11sqrt2` 等が別票になる。LLM 完全ログ（run002）の 9 サンプルから
  (a) 現行: 正規化文字列で多数決 → 現行採点
  (b) 現行多数決 → 改良採点（audit_math_grading の同値判定）
  (c) 改良: 改良正規化（canon）で票を束ねて多数決 → 改良採点
を比べ、(c)−(b) を「票割れによる損失」として出す。

実行: uv run --with sympy python scripts/analysis/audit_math_vote_splitting.py [--math500-jsonl PATH]
"""
from __future__ import annotations

import collections
import glob
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/analysis"))

import audit_math_grading as G  # noqa: E402
from src.evalx.debate import majority_vote  # noqa: E402
from src.evalx.tasks import extract_answer  # noqa: E402

R2 = ROOT / "results/gcs/run002"


def main() -> None:
    path = sys.argv[sys.argv.index("--math500-jsonl") + 1] if "--math500-jsonl" in sys.argv else None
    rows = G.load_math500(path)
    by_problem = {r["problem"]: i for i, r in enumerate(rows)}
    samples = collections.defaultdict(dict)
    for d in ("remeasure_v1", "robust_c2"):
        for f in glob.glob(str(R2 / d / "llm_calls" / "*.jsonl.gz")):
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if "ANSWER: <final simplified answer>" not in line:
                        continue
                    rec = json.loads(line)
                    if rec["seed"] is None or rec["seed"] < 1000 or len(rec["messages"]) != 2:
                        continue
                    idx = by_problem.get(rec["messages"][1]["content"])
                    if idx is None:
                        continue
                    seed, k = divmod(rec["seed"], 1000)
                    samples[(idx, seed)].setdefault(k, extract_answer(rec["response"], "math"))
    n = a = b = c = 0
    for (idx, seed), slot in samples.items():
        if len(slot) < 9:
            continue
        answers = [slot[k] for k in range(9)]
        gold_c = G.gold_canon(rows[idx]["answer"])
        gold_proj = G.project_normalize(rows[idx]["answer"])
        maj = majority_vote(answers, seed)
        canon_answers = [None if x is None else G.canon(x) for x in answers]
        maj_c = majority_vote(canon_answers, seed)
        n += 1
        a += maj is not None and maj == gold_proj or (maj is not None and _float_eq(maj, gold_proj))
        ok_a = maj is not None and (maj == gold_proj or _float_eq(maj, gold_proj))
        b += ok_a or (maj is not None and G.equivalent(gold_c, G.canon(maj)))
        # 改良多数決の勝者が現行正規化で金と一致する場合も正解（\text{名前} 等）
        winner_raw = next((x for x in answers if x is not None and G.canon(x) == maj_c), None)
        c += (winner_raw is not None and (winner_raw == gold_proj or _float_eq(winner_raw, gold_proj))) or (
            maj_c is not None and G.equivalent(gold_c, maj_c))
    print(f"MATH-500 SC@9 (新環境, item×seed={n})")
    print(f"  (a) 現行多数決・現行採点  = {a/n:.4f}")
    print(f"  (b) 現行多数決・改良採点  = {b/n:.4f}   (偽陰性の修正分 {100*(b-a)/n:+.2f}pt)")
    print(f"  (c) 改良多数決・改良採点  = {c/n:.4f}   (票割れの解消分 {100*(c-b)/n:+.2f}pt)")


def _float_eq(x: str, y: str) -> bool:
    try:
        return abs(float(x) - float(y)) < 1e-6
    except ValueError:
        return False


if __name__ == "__main__":
    main()
