"""進化ループのドライバ (docs/research_design.md §4)。

前提: vLLM が `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` かつ `--enable-lora` で
起動しており、動的 LoRA ロードが可能であること。

使用例:
  uv run python scripts/run_evolution.py \
    --base-url http://localhost:8000/v1 \
    --gen0 critic=adapters/persona_a pragmatist=adapters/persona_b explorer=adapters/persona_c \
    --generations 6 --fitness-task mmlu_pro --fitness-n 100 --fitness-seed 777 \
    --out-root adapters/evolution --log results/evolution_run.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evalx.client import ChatClient, GenerationConfig
from src.evalx.tasks import load_task
from src.evolve.loop import (
    CoalitionEvaluator,
    Individual,
    VllmLoraManager,
    breed_next_generation,
    make_child,
    run_generation,
)

ROLE_PERSONAS = {
    "critic": "あなたは厳密な検証を重視する批判的思考家。反証・例外・境界条件に敏感。",
    "pragmatist": "あなたは応用志向の実務家。意思決定に役立つ実装可能性とコストを重視。",
    "explorer": "あなたは創発を促す発想家。仮説生成と多角的比喩で発想を広げる。",
}


def parse_gen0(raw: list[str]) -> dict[str, str]:
    mapping = {}
    for item in raw:
        role, path = item.split("=", 1)
        if role not in ROLE_PERSONAS:
            raise ValueError(f"Unknown role '{role}'. Expected one of {sorted(ROLE_PERSONAS)}")
        mapping[role] = path
    if len(mapping) != 3:
        raise ValueError("Exactly three roles (critic/pragmatist/explorer) are required.")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--gen0", nargs=3, required=True, help="role=adapter_dir 形式で 3 役割")
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--fitness-task", default="mmlu_pro")
    parser.add_argument("--fitness-n", type=int, default=100)
    parser.add_argument("--fitness-seed", type=int, default=777, help="適応度セットのサンプリングシード（最終評価と変えること）")
    parser.add_argument("--rounds", type=int, default=1, help="debate ラウンド数（文献推奨: 1-2）")
    parser.add_argument("--fitness-mode", choices=["shapley", "solo"], default="shapley")
    parser.add_argument("--no-sharing", action="store_true", help="fitness sharing を無効化（アブレーション A2）")
    parser.add_argument("--crossover", choices=["delta", "naive"], default="delta")
    parser.add_argument("--sharing-sigma", type=float, default=0.3)
    parser.add_argument("--mut-ratio", type=float, default=0.3)
    parser.add_argument("--mut-std", type=float, default=0.02)
    parser.add_argument("--evolution-seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--out-root", default="adapters/evolution")
    parser.add_argument("--log", required=True, help="実行全体のログ JSON 出力先")
    args = parser.parse_args()

    gen0_map = parse_gen0(args.gen0)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.evolution_seed)

    client = ChatClient(base_url=args.base_url)
    config = GenerationConfig(temperature=args.temperature, max_tokens=args.max_tokens, seed=args.fitness_seed)
    task = load_task(args.fitness_task, n=args.fitness_n, seed=args.fitness_seed)
    lora_manager = VllmLoraManager(args.base_url)

    # gen 0: SFT 済み 3 個体 + 各役割の突然変異コピー（初期多様性の確保）
    subpopulations: dict[str, list[Individual]] = {}
    representatives: dict[str, Individual] = {}
    for role, adapter_dir in sorted(gen0_map.items()):
        original = Individual(
            name=f"gen0_{role}_base",
            role=role,
            adapter_dir=adapter_dir,
            persona_prompt=ROLE_PERSONAS[role],
        )
        mutant_dir = out_root / "gen_00" / f"gen0_{role}_mutant"
        if not mutant_dir.exists():
            make_child(
                original, original, f"gen0_{role}_mutant", str(mutant_dir),
                alpha=0.5, mut_ratio=1.0, mut_std=args.mut_std,
                seed=rng.randrange(2**31), crossover=args.crossover,
            )
        mutant = Individual(
            name=f"gen0_{role}_mutant",
            role=role,
            adapter_dir=str(mutant_dir),
            persona_prompt=ROLE_PERSONAS[role],
        )
        subpopulations[role] = [original, mutant]
        representatives[role] = original

    run_log = {
        "config": vars(args),
        "fitness_items": [item.item_id for item in task.items],
        "generations": [],
    }

    for generation in range(args.generations):
        print(f"[info] === generation {generation} ===")
        evaluator = CoalitionEvaluator(client, task, config, args.rounds, args.fitness_seed)
        result = run_generation(
            generation=generation,
            subpopulations=subpopulations,
            representatives=representatives,
            evaluator=evaluator,
            lora_manager=lora_manager,
            out_root=out_root,
            fitness_mode=args.fitness_mode,
            sharing_sigma=args.sharing_sigma,
            use_sharing=not args.no_sharing,
        )
        representatives = result["representatives"]
        run_log["generations"].append(result["log"])

        for role, rep in sorted(representatives.items()):
            record = result["log"]["roles"][role]["candidates"][rep.name]
            print(
                f"[info] {role}: selected={rep.name} fitness={record['fitness']:.4f} "
                f"(shapley={record['shapley']:.4f}, solo={record['solo_accuracy']:.4f}, "
                f"team={record['team_accuracy']:.4f})"
            )

        # 逐次保存（Spot プリエンプト対策）
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")

        if generation < args.generations - 1:
            fitness_by_name = {
                name: rec["fitness"]
                for role_log in result["log"]["roles"].values()
                for name, rec in role_log["candidates"].items()
            }
            subpopulations = breed_next_generation(
                generation=generation,
                representatives=representatives,
                subpopulations=subpopulations,
                fitness_by_name=fitness_by_name,
                out_root=out_root,
                rng=rng,
                mut_ratio=args.mut_ratio,
                mut_std=args.mut_std,
                crossover=args.crossover,
                persona_prompts=ROLE_PERSONAS,
            )

    final_team = {role: rep.adapter_dir for role, rep in sorted(representatives.items())}
    run_log["final_team"] = final_team
    Path(args.log).write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] evolution complete. final team: {final_team}")


if __name__ == "__main__":
    main()
