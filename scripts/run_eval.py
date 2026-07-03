"""ベンチマーク評価ランナー（vLLM OpenAI 互換 API 経由）。

モード:
  solo       各エージェント単独の精度
  team       全エージェントによる debate の精度
  coalitions 全ての非空連合で debate を実行し、厳密 Shapley 値まで算出
  sc         Self-Consistency@k（単一エージェントから k サンプル → 多数決。
             debate と計算量マッチのベースライン）

使用例:
  uv run python scripts/run_eval.py \
    --base-url http://localhost:8000/v1 \
    --task gsm8k --n 200 --seed 42 --rounds 2 \
    --agents persona_a=persona_a persona_b=persona_b persona_c=persona_c \
    --mode coalitions --out results/eval_gen0.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.personalities import PERSONAS
from src.evalx.client import ChatClient, GenerationConfig
from src.evalx.debate import AgentSpec, majority_vote, run_debate, solo_answer
from src.evalx.shapley import all_coalitions, shapley_values
from src.evalx.tasks import load_task, score_predictions


def parse_agents(raw_agents: list[str], persona_map: dict[str, str]) -> list[AgentSpec]:
    agents = []
    for raw in raw_agents:
        if "=" not in raw:
            raise ValueError(f"Invalid agent spec '{raw}'. Expected name=model.")
        name, model = raw.split("=", 1)
        agents.append(AgentSpec(name=name, model=model, persona_prompt=persona_map.get(name, "")))
    return agents


def evaluate_solo(client, agents, task, config) -> dict:
    results = {}
    for agent in agents:
        predictions = {}
        for item in task.items:
            result = solo_answer(client, agent, item, task.answer_type, config)
            predictions[item.item_id] = result["answer"]
        results[agent.name] = score_predictions(task.items, predictions, task.answer_type)
        print(f"[info] solo {agent.name}: acc={results[agent.name]['accuracy']:.4f}")
    return results


def evaluate_self_consistency(client, agent, task, config, k, tie_break_seed) -> dict:
    from dataclasses import replace

    predictions = {}
    for item in task.items:
        answers = []
        for sample_idx in range(k):
            # 同一 seed だと同一サンプルが k 回返るため、サンプルごとに seed をずらす
            sample_config = replace(
                config, seed=None if config.seed is None else config.seed * 1000 + sample_idx
            )
            result = solo_answer(client, agent, item, task.answer_type, sample_config)
            answers.append(result["answer"])
        predictions[item.item_id] = majority_vote(answers, tie_break_seed)
    return score_predictions(task.items, predictions, task.answer_type)


def evaluate_team(client, agents, task, config, rounds, tie_break_seed, transcript_path=None) -> dict:
    predictions = {}
    transcripts = []
    for item in task.items:
        record = run_debate(client, agents, item, task.answer_type, rounds, config, tie_break_seed)
        predictions[item.item_id] = record.majority_answer
        transcripts.append(
            {
                "item_id": item.item_id,
                "rounds": record.rounds,
                "final_answers": record.final_answers,
                "majority_answer": record.majority_answer,
            }
        )
    if transcript_path:
        Path(transcript_path).parent.mkdir(parents=True, exist_ok=True)
        Path(transcript_path).write_text(
            json.dumps(transcripts, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return score_predictions(task.items, predictions, task.answer_type)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--task", default="gsm8k", choices=["gsm8k", "mmlu", "arc_challenge"])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42, help="問題サンプリングと同点処理のシード")
    parser.add_argument("--rounds", type=int, default=2, help="独立回答後の debate ラウンド数")
    parser.add_argument("--agents", nargs="+", required=True, help="name=model 形式")
    parser.add_argument("--mode", choices=["solo", "team", "coalitions", "sc"], default="coalitions")
    parser.add_argument("--sc-k", type=int, default=9, help="Self-Consistency のサンプル数")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--out", required=True)
    parser.add_argument("--save-transcripts", action="store_true")
    args = parser.parse_args()

    client = ChatClient(base_url=args.base_url)
    config = GenerationConfig(temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed)
    agents = parse_agents(args.agents, PERSONAS)
    task = load_task(args.task, n=args.n, seed=args.seed)
    print(f"[info] task={task.name} n={len(task.items)} agents={[a.name for a in agents]} mode={args.mode}")

    started = time.time()
    payload = {
        "task": task.name,
        "n": len(task.items),
        "seed": args.seed,
        "rounds": args.rounds,
        "mode": args.mode,
        "agents": [{"name": a.name, "model": a.model, "persona": a.persona_prompt} for a in agents],
        "generation": {"temperature": args.temperature, "max_tokens": args.max_tokens},
    }
    out_dir = Path(args.out).parent

    if args.mode == "solo":
        payload["solo"] = evaluate_solo(client, agents, task, config)
    elif args.mode == "sc":
        payload["sc_k"] = args.sc_k
        payload["sc"] = {}
        for agent in agents:
            payload["sc"][agent.name] = evaluate_self_consistency(
                client, agent, task, config, args.sc_k, args.seed
            )
            print(f"[info] sc@{args.sc_k} {agent.name}: acc={payload['sc'][agent.name]['accuracy']:.4f}")
    elif args.mode == "team":
        transcript_path = out_dir / "transcripts_team.json" if args.save_transcripts else None
        payload["team"] = evaluate_team(client, agents, task, config, args.rounds, args.seed, transcript_path)
        print(f"[info] team: acc={payload['team']['accuracy']:.4f}")
    else:
        # 全連合を評価して厳密 Shapley 値を算出
        agent_by_name = {a.name: a for a in agents}
        names = [a.name for a in agents]
        coalition_results = {}
        for coalition in all_coalitions(names):
            members = [agent_by_name[n] for n in coalition]
            key = "+".join(coalition)
            if len(members) == 1:
                result = evaluate_solo(client, members, task, config)[members[0].name]
            else:
                transcript_path = (
                    out_dir / f"transcripts_{key}.json" if args.save_transcripts else None
                )
                result = evaluate_team(client, members, task, config, args.rounds, args.seed, transcript_path)
            coalition_results[key] = result
            print(f"[info] coalition {key}: acc={result['accuracy']:.4f}")

        values = shapley_values(
            names,
            {frozenset(c.split("+")): r["accuracy"] for c, r in coalition_results.items()},
        )
        payload["coalitions"] = coalition_results
        payload["shapley"] = values
        payload["solo"] = {n: coalition_results[n] for n in names}
        payload["team"] = coalition_results["+".join(names)]

    payload["elapsed_seconds"] = time.time() - started
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] wrote {out_path} in {payload['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
