"""評価結果の統計分析 (docs/research_design.md §6.3)。

2 条件の paired 比較（McNemar + paired bootstrap CI）と、
複数比較の Holm-Bonferroni 補正、進化軌跡の要約を行う。

使用例:
  # 2 条件比較（run_eval.py の出力 JSON 同士）
  uv run python scripts/analyze_results.py compare \
    --a results/final/sc9.json --a-key sc.base \
    --b results/final/evolved_team.json --b-key team \
    --label "SC@9 vs evolved team"

  # 進化軌跡の要約
  uv run python scripts/analyze_results.py trajectory --log results/evolution_run.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evalx.stats import bootstrap_accuracy_diff, mcnemar_exact, paired_outcomes


def extract_per_item(payload: dict, key: str) -> dict:
    """'team' や 'sc.base' のようなドット区切りキーで per_item 辞書を取り出す。"""
    node = payload
    for part in key.split("."):
        if part not in node:
            raise KeyError(f"Key '{part}' not found. Available: {sorted(node.keys())}")
        node = node[part]
    if "per_item" not in node:
        raise KeyError(f"'{key}' does not contain per_item results.")
    return node["per_item"]


def holm_bonferroni(p_values: list[tuple[str, float]]) -> list[dict]:
    """(label, p) のリストに Holm-Bonferroni 補正を適用する。"""
    m = len(p_values)
    ordered = sorted(p_values, key=lambda item: item[1])
    results = []
    running_max = 0.0
    for rank, (label, p) in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * p)
        running_max = max(running_max, adjusted)
        results.append({"label": label, "p_raw": p, "p_holm": running_max})
    return results


def cmd_compare(args: argparse.Namespace) -> dict:
    payload_a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    payload_b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    per_item_a = extract_per_item(payload_a, args.a_key)
    per_item_b = extract_per_item(payload_b, args.b_key)
    pairs = paired_outcomes(per_item_a, per_item_b)

    acc_a = sum(a for a, _ in pairs) / len(pairs)
    acc_b = sum(b for _, b in pairs) / len(pairs)
    mcnemar = mcnemar_exact(pairs)
    bootstrap = bootstrap_accuracy_diff(pairs, n_resamples=args.bootstrap, seed=args.seed)

    result = {
        "label": args.label or f"{args.a_key} vs {args.b_key}",
        "n_common_items": len(pairs),
        "accuracy_a": acc_a,
        "accuracy_b": acc_b,
        "diff_b_minus_a": bootstrap["diff"],
        "bootstrap_ci95": [bootstrap["ci_lower"], bootstrap["ci_upper"]],
        "mcnemar": mcnemar,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def cmd_trajectory(args: argparse.Namespace) -> None:
    log = json.loads(Path(args.log).read_text(encoding="utf-8"))
    rows = []
    for gen_log in log["generations"]:
        gen = gen_log["generation"]
        for role, role_log in sorted(gen_log["roles"].items()):
            selected = role_log["selected"]
            record = role_log["candidates"][selected]
            rows.append(
                {
                    "generation": gen,
                    "role": role,
                    "selected": selected,
                    "fitness": record["fitness"],
                    "shapley": record["shapley"],
                    "solo_accuracy": record["solo_accuracy"],
                    "team_accuracy": record["team_accuracy"],
                    "sharing_penalty": record.get("sharing_penalty", 1.0),
                }
            )
    print(f"{'gen':>3} {'role':<12} {'fitness':>8} {'shapley':>8} {'solo':>6} {'team':>6}")
    for row in rows:
        print(
            f"{row['generation']:>3} {row['role']:<12} {row['fitness']:>8.4f} "
            f"{row['shapley']:>8.4f} {row['solo_accuracy']:>6.3f} {row['team_accuracy']:>6.3f}"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[info] wrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser("compare", help="2 条件の paired 比較")
    compare.add_argument("--a", required=True, help="条件 A の eval JSON")
    compare.add_argument("--a-key", required=True, help="per_item を含むノードへのドット区切りキー")
    compare.add_argument("--b", required=True, help="条件 B の eval JSON")
    compare.add_argument("--b-key", required=True)
    compare.add_argument("--label", default=None)
    compare.add_argument("--bootstrap", type=int, default=10000)
    compare.add_argument("--seed", type=int, default=0)
    compare.set_defaults(func=cmd_compare)

    trajectory = sub.add_parser("trajectory", help="進化軌跡の要約")
    trajectory.add_argument("--log", required=True, help="run_evolution.py の実行ログ JSON")
    trajectory.add_argument("--out", default=None)
    trajectory.set_defaults(func=cmd_trajectory)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
