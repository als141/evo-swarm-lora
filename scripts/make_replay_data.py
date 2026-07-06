"""能力保持リプレイデータの自己生成（設計v2・修正2）。

ベースモデル自身に高難度問題を解かせ、正解した長いCoTのみを
SFTリプレイ例として収集する。ペルソナSFTによる能力破壊
（v3実測: MATH-500 Level5 で 0.82→0.55）への対策。

- MATH 訓練split Level4-5（MATH-500はtest split由来なので重複なし）
- MMLU-Pro validation split（最終評価はtest splitなので重複なし）
- 出力は system なしの {user, assistant} ペア。ペルソナ SFT ファイルへの
  マージ時に各ペルソナの system prompt を付与する。

使用例（vLLM 稼働中のホストで）:
  python3 scripts/make_replay_data.py --base-url http://localhost:8000/v1 \
    --out /gcs/<bucket>/experiments/run001/replay/replay_pool.jsonl \
    --n-math 24 --n-mmlu 12
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from src.evalx.client import ChatClient, GenerationConfig
from src.evalx.debate import ANSWER_FORMAT_LETTER, ANSWER_FORMAT_MATH
from src.evalx.parallel import parallel_map
from src.evalx.tasks import (
    CHOICE_LETTERS,
    extract_answer,
    is_correct,
    normalize_math_answer,
)

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MATH_SUBJECTS = [
    "algebra",
    "intermediate_algebra",
    "precalculus",
    "geometry",
    "counting_and_probability",
    "number_theory",
    "prealgebra",
]
# v3実測で劣化が大きかった分野を優先的に厚くする
PRIORITY_SUBJECTS = {"intermediate_algebra", "precalculus", "geometry"}


def load_math_train(seed: int, per_subject: int = 40) -> list[dict]:
    items = []
    for subject in MATH_SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        rows = [r for r in ds if r["level"] in ("Level 4", "Level 5")]
        random.Random(seed).shuffle(rows)
        quota = per_subject * (2 if subject in PRIORITY_SUBJECTS else 1)
        for row in rows[:quota]:
            boxed = re.findall(r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", row["solution"])
            if not boxed:
                continue
            items.append(
                {
                    "kind": "math",
                    "question": row["problem"],
                    "gold": normalize_math_answer(boxed[-1]),
                    "subject": subject,
                    "level": row["level"],
                }
            )
    return items


def load_mmlu_val(seed: int, n: int = 60) -> list[dict]:
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    items = []
    for idx in indices[:n]:
        row = ds[idx]
        lines = [row["question"].strip(), ""]
        for letter, choice in zip(CHOICE_LETTERS, row["options"]):
            lines.append(f"{letter}. {choice}")
        items.append(
            {
                "kind": "letter",
                "question": "\n".join(lines),
                "gold": row["answer"].strip().upper(),
            }
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-math", type=int, default=24, help="採用するMATHリプレイ例数")
    parser.add_argument("--n-mmlu", type=int, default=12, help="採用するMMLU-Proリプレイ例数")
    parser.add_argument("--min-chars", type=int, default=500, help="CoTの最短文字数（短い解答は採らない）")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()

    client = ChatClient(base_url=args.base_url)
    config = GenerationConfig(temperature=0.7, max_tokens=args.max_tokens, seed=args.seed)

    candidates = load_math_train(args.seed) + load_mmlu_val(args.seed)
    print(f"[info] candidate problems: {len(candidates)}")

    results = []

    def process(item):
        fmt = ANSWER_FORMAT_MATH if item["kind"] == "math" else ANSWER_FORMAT_LETTER
        answer_type = item["kind"] if item["kind"] == "math" else "letter"
        messages = [
            {"role": "system", "content": fmt},
            {"role": "user", "content": item["question"]},
        ]
        utterance = client.chat(BASE_MODEL, messages, config)
        predicted = extract_answer(utterance, answer_type)
        if (
            predicted is not None
            and is_correct(predicted, item["gold"], answer_type)
            and len(utterance) >= args.min_chars
        ):
            results.append({**item, "cot": utterance})

    parallel_map(candidates, process, max_workers=24)
    print(f"[info] verified-correct long CoT: {len(results)}")

    rng = random.Random(args.seed)
    math_pool = [r for r in results if r["kind"] == "math"]
    mmlu_pool = [r for r in results if r["kind"] == "letter"]
    rng.shuffle(math_pool)
    rng.shuffle(mmlu_pool)
    # 劣化が大きかった分野を優先しつつ選抜
    math_pool.sort(key=lambda r: (r["subject"] not in PRIORITY_SUBJECTS, r["level"] != "Level 5"))
    picked = math_pool[: args.n_math] + mmlu_pool[: args.n_mmlu]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for r in picked:
            handle.write(
                json.dumps(
                    {
                        "user": r["question"],
                        "assistant": r["cot"],
                        "meta": {k: r[k] for k in ("kind", "subject", "level") if k in r},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    kinds = {}
    for r in picked:
        kinds[r.get("subject", r["kind"])] = kinds.get(r.get("subject", r["kind"]), 0) + 1
    print(f"[info] wrote {len(picked)} replay examples to {out_path}: {kinds}")


if __name__ == "__main__":
    main()
