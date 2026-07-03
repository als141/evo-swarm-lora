"""最終評価バッテリー設定の生成 (docs/research_design.md §6.2 の 7 条件)。

進化済みアダプタのパスを受け取り、全条件 × 全ベンチ × 全 seed の
run_eval エントリを含む JSON を出力する。

使用例:
  uv run python scripts/cloud/make_battery_config.py \
    --gen0-prefix /gcs/<bucket>/experiments/run001/training/model/adapters \
    --evolved critic=/gcs/.../gen_05/gen5_critic_child ... \
    --ablation-solo critic=/gcs/.../solo_run/gen_05/... ... \
    --seeds 1 2 3 --out cloud/eval_battery_final.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BENCHMARKS = [
    ("mmlu_pro", 500),
    ("math500", 200),
    ("supergpqa", 300),
]
ROLES = ["critic", "pragmatist", "explorer"]


def parse_team(specs: list[str], prefix: str) -> dict[str, str]:
    team = {}
    for spec in specs:
        role, path = spec.split("=", 1)
        team[f"{prefix}_{role}" if prefix else role] = path
    return team


def team_entry(name, task, n, seed, rounds, adapters: dict[str, str], agents: dict[str, str]):
    args = ["--task", task, "--n", str(n), "--seed", str(seed), "--mode", "team", "--rounds", str(rounds)]
    if adapters:
        args += ["--load-adapters"] + [f"{k}={v}" for k, v in adapters.items()]
    args += ["--agents"] + [f"{k}={v}" for k, v in agents.items()]
    return {"name": name, "args": args}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen0-prefix", required=True, help="gen0 アダプタ群の親ディレクトリ（persona_a/b/c を含む）")
    parser.add_argument("--evolved", nargs=3, required=True, help="role=path 形式（主要条件のチーム）")
    parser.add_argument("--ablation-solo", nargs=3, default=None, help="role=path 形式（solo適応度で進化したチーム A1）")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--sc-k", type=int, default=9)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    gen0 = {role: f"{args.gen0_prefix}/persona_{suffix}" for role, suffix in
            zip(ROLES, ["a", "b", "c"])}
    evolved = parse_team(args.evolved, "")
    ablation = parse_team(args.ablation_solo, "") if args.ablation_solo else None

    entries = []
    for task, n in BENCHMARKS:
        for seed in args.seeds:
            suffix = f"{task}_s{seed}"
            common = {"task": task, "n": n, "seed": seed, "rounds": args.rounds}

            # 条件1: ベース solo CoT
            entries.append({
                "name": f"c1_base_solo_{suffix}",
                "args": ["--task", task, "--n", str(n), "--seed", str(seed), "--mode", "solo",
                         "--agents", f"base={BASE_MODEL}", "--no-persona-prompt"],
            })
            # 条件2: Self-Consistency@k（計算量マッチ）
            entries.append({
                "name": f"c2_sc{args.sc_k}_{suffix}",
                "args": ["--task", task, "--n", str(n), "--seed", str(seed), "--mode", "sc",
                         "--sc-k", str(args.sc_k),
                         "--agents", f"base={BASE_MODEL}", "--no-persona-prompt"],
            })
            # 条件3: ベース×3 温度サンプリング debate（重みなしペルソナ対照）
            entries.append(team_entry(
                f"c3_base_team_{suffix}", **common,
                adapters={},
                agents={f"sampler_{i}": BASE_MODEL for i in range(3)},
            ))
            # 条件3': ベース×3 + プロンプトペルソナのみ debate
            entries.append(team_entry(
                f"c3p_prompt_persona_team_{suffix}", **common,
                adapters={},
                agents={role: BASE_MODEL for role in ROLES},
            ))
            # 条件4: gen0 ペルソナ LoRA チーム
            entries.append(team_entry(
                f"c4_gen0_team_{suffix}", **common,
                adapters={f"g0_{r}": p for r, p in gen0.items()},
                agents={r: f"g0_{r}" for r in ROLES},
            ))
            # 条件5: 進化後チーム（主要条件）
            entries.append(team_entry(
                f"c5_evolved_team_{suffix}", **common,
                adapters={f"ev_{r}": p for r, p in evolved.items()},
                agents={r: f"ev_{r}" for r in ROLES},
            ))
            # 条件6: 進化後 LoRA solo ×3
            entries.append({
                "name": f"c6_evolved_solo_{suffix}",
                "args": ["--task", task, "--n", str(n), "--seed", str(seed), "--mode", "solo",
                         "--load-adapters", *[f"ev_{r}={p}" for r, p in evolved.items()],
                         "--agents", *[f"{r}=ev_{r}" for r in ROLES]],
            })
            # 条件7: A1 solo 適応度で進化したチーム
            if ablation:
                entries.append(team_entry(
                    f"c7_solofit_team_{suffix}", **common,
                    adapters={f"sf_{r}": p for r, p in ablation.items()},
                    agents={r: f"sf_{r}" for r in ROLES},
                ))

    Path(args.out).write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] wrote {len(entries)} battery entries to {args.out}")


if __name__ == "__main__":
    main()
