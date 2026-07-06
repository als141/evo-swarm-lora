"""run002 用 SFT データセットの合成（設計v3 §3.1 G1）。

各ペルソナの既存 SFT 60例に、ベースモデル自己生成の能力保持リプレイ
（scripts/make_replay_data.py の出力）へ当該ペルソナの system prompt を
付与した例を追加し、新しい JSONL を書き出す。

- スタイルはペルソナ例が、能力（長CoT）はリプレイ例が担う分業。
- リプレイの user/assistant は改変しない（正解検証済みテキストを保つ）。

使用例:
  uv run python scripts/make_run002_datasets.py \
    --replay /path/to/replay_pool.jsonl --out-dir data/run002
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.personalities import PERSONAS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, help="replay_pool.jsonl のパス")
    parser.add_argument("--data-dir", default="data", help="既存ペルソナ SFT の場所")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-replay", type=int, default=36, help="混合するリプレイ例の上限")
    args = parser.parse_args()

    replay_rows = []
    with Path(args.replay).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                replay_rows.append(json.loads(line))
    if len(replay_rows) > args.max_replay:
        replay_rows = replay_rows[: args.max_replay]
    if not replay_rows:
        raise SystemExit("replay pool is empty")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for persona, system_prompt in PERSONAS.items():
        src = Path(args.data_dir) / f"sft_{persona}.jsonl"
        rows = []
        seen_users = set()
        with src.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows.append(row)
                seen_users.add(row["messages"][1]["content"])

        added = 0
        for rep in replay_rows:
            if rep["user"] in seen_users:  # 偶発重複は追加しない
                continue
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": rep["user"]},
                        {"role": "assistant", "content": rep["assistant"]},
                    ]
                }
            )
            added += 1

        out_path = out_dir / f"sft_{persona}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                # 構造検証: 3メッセージ・役割順序
                roles = [m["role"] for m in row["messages"]]
                assert roles == ["system", "user", "assistant"], roles
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[info] {out_path}: base={len(rows) - added} replay={added} total={len(rows)}")


if __name__ == "__main__":
    main()
