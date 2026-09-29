"""子個体の LoRA 学習を別プロセスで実行する（学習後に GPU メモリを完全に解放するため）。

入力: --examples（1 行 1 例の JSONL、各行は messages のリスト）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evo4.train import train_lora  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--parent", default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    args = parser.parse_args()
    examples = [json.loads(line) for line in open(args.examples, encoding="utf-8") if line.strip()]
    info = train_lora(examples, args.out, parent_adapter=args.parent or None, seed=args.seed, lr=args.lr,
                      log=lambda m: print(m, flush=True))
    print(json.dumps({k: v for k, v in info.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()
