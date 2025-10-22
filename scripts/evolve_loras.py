import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.lora_ops import alpha_blend_lora, mutate_lora


def main():
    parser = argparse.ArgumentParser(description="Blend and mutate LoRA adapters to produce a new generation.")
    parser.add_argument("--parents", nargs=2, required=True, help="Parent adapter directories.")
    parser.add_argument("--child", required=True, help="Output directory for the child adapter.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend ratio between parent adapters.")
    parser.add_argument("--mut", type=float, default=0.05, help="Mutation ratio applied to LoRA tensors.")
    parser.add_argument("--mut-std", type=float, default=0.01, help="Standard deviation of Gaussian noise for mutation.")
    args = parser.parse_args()

    tmp_dir = f"{args.child}_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    alpha_blend_lora(args.parents[0], args.parents[1], tmp_dir, args.alpha)

    if args.mut > 0:
        if os.path.exists(args.child):
            shutil.rmtree(args.child)
        mutate_lora(tmp_dir, args.child, ratio=args.mut, std=args.mut_std)
        shutil.rmtree(tmp_dir)
    else:
        if os.path.exists(args.child):
            shutil.rmtree(args.child)
        os.rename(tmp_dir, args.child)
    print(f"Generated child adapter at {args.child}")


if __name__ == "__main__":
    main()
