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
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.personalities import PERSONAS, ROLE_PERSONAS
from src.evalx.client import ChatClient, GenerationConfig
from src.evalx.debate import AgentSpec, majority_vote, run_debate, solo_answer
from src.evalx.parallel import parallel_map
from src.evalx.shapley import all_coalitions, shapley_values
from src.evalx.tasks import load_task, score_predictions


class ProgressCache:
    """問題単位で予測を JSONL に逐次追記し、Spot プリエンプション後の再開を可能にする。"""

    def __init__(self, path: Path | None):
        self._path = path
        self._lock = threading.Lock()
        self._answers: dict[str, str | None] = {}
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                self._answers[record["item_id"]] = record["answer"]
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._answers

    def get(self, item_id: str):
        return self._answers.get(item_id)

    def put(self, item_id: str, answer) -> None:
        with self._lock:
            self._answers[item_id] = answer
            if self._path is not None:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps({"item_id": item_id, "answer": answer}, ensure_ascii=False) + "\n"
                    )


def _cache_for(progress_dir: Path | None, label: str) -> ProgressCache:
    safe = label.replace("/", "_").replace("+", "_")
    return ProgressCache(progress_dir / f"{safe}.jsonl" if progress_dir else None)


def parse_agents(raw_agents: list[str], persona_map: dict[str, str]) -> list[AgentSpec]:
    agents = []
    for raw in raw_agents:
        if "=" not in raw:
            raise ValueError(f"Invalid agent spec '{raw}'. Expected name=model.")
        name, model = raw.split("=", 1)
        agents.append(AgentSpec(name=name, model=model, persona_prompt=persona_map.get(name, "")))
    return agents


def evaluate_solo(client, agents, task, config, progress_dir=None, workers=16) -> dict:
    results = {}
    for agent in agents:
        cache = _cache_for(progress_dir, f"solo_{agent.name}")
        pending = [item for item in task.items if item.item_id not in cache]

        def process(item, agent=agent, cache=cache):
            result = solo_answer(client, agent, item, task.answer_type, config)
            cache.put(item.item_id, result["answer"])

        parallel_map(pending, process, max_workers=workers)
        predictions = {item.item_id: cache.get(item.item_id) for item in task.items}
        results[agent.name] = score_predictions(task.items, predictions, task.answer_type)
        print(f"[info] solo {agent.name}: acc={results[agent.name]['accuracy']:.4f}")
    return results


def evaluate_self_consistency(
    client, agent, task, config, k, tie_break_seed, progress_dir=None, workers=16
) -> dict:
    from dataclasses import replace

    cache = _cache_for(progress_dir, f"sc{k}_{agent.name}")
    pending = [item for item in task.items if item.item_id not in cache]

    def process(item):
        answers = []
        for sample_idx in range(k):
            # 同一 seed だと同一サンプルが k 回返るため、サンプルごとに seed をずらす
            sample_config = replace(
                config, seed=None if config.seed is None else config.seed * 1000 + sample_idx
            )
            result = solo_answer(client, agent, item, task.answer_type, sample_config)
            answers.append(result["answer"])
        cache.put(item.item_id, majority_vote(answers, tie_break_seed))

    parallel_map(pending, process, max_workers=workers)
    predictions = {item.item_id: cache.get(item.item_id) for item in task.items}
    return score_predictions(task.items, predictions, task.answer_type)


def evaluate_team(
    client, agents, task, config, rounds, tie_break_seed, transcript_path=None, progress_dir=None,
    workers=16,
) -> dict:
    label = "team_" + "_".join(a.name for a in agents)
    cache = _cache_for(progress_dir, label)
    pending = [item for item in task.items if item.item_id not in cache]
    transcripts = []
    transcripts_lock = threading.Lock()

    def process(item):
        record = run_debate(client, agents, item, task.answer_type, rounds, config, tie_break_seed)
        cache.put(item.item_id, record.majority_answer)
        with transcripts_lock:
            transcripts.append(
                {
                    "item_id": item.item_id,
                    "rounds": record.rounds,
                    "final_answers": record.final_answers,
                    "majority_answer": record.majority_answer,
                }
            )

    parallel_map(pending, process, max_workers=workers)
    predictions = {item.item_id: cache.get(item.item_id) for item in task.items}
    if transcript_path and transcripts:
        path = Path(transcript_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        path.write_text(
            json.dumps(existing + transcripts, ensure_ascii=False, indent=2), encoding="utf-8"
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
    parser.add_argument(
        "--load-adapters",
        nargs="*",
        default=[],
        help="name=path 形式。vLLM の動的 LoRA ロード API で評価前に登録する",
    )
    parser.add_argument(
        "--no-persona-prompt",
        action="store_true",
        help="ペルソナ system prompt を使わない（ベースモデル×温度サンプリング条件用）",
    )
    parser.add_argument("--mode", choices=["solo", "team", "coalitions", "sc"], default="coalitions")
    parser.add_argument("--sc-k", type=int, default=9, help="Self-Consistency のサンプル数")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=16, help="問題単位の並列リクエスト数")
    parser.add_argument("--out", required=True)
    parser.add_argument("--save-transcripts", action="store_true")
    parser.add_argument(
        "--progress-dir",
        default=None,
        help="問題単位の逐次キャッシュを置くディレクトリ（Spot プリエンプト後の再開用）",
    )
    args = parser.parse_args()

    client = ChatClient(base_url=args.base_url)
    config = GenerationConfig(temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed)

    if args.load_adapters:
        from src.evolve.loop import VllmLoraManager

        manager = VllmLoraManager(args.base_url)
        for spec in args.load_adapters:
            name, path = spec.split("=", 1)
            manager.load(name, path)
            print(f"[info] loaded adapter {name} from {path}")

    persona_map = {} if args.no_persona_prompt else {**PERSONAS, **ROLE_PERSONAS}
    agents = parse_agents(args.agents, persona_map)
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

    progress_dir = Path(args.progress_dir) if args.progress_dir else None

    if args.mode == "solo":
        payload["solo"] = evaluate_solo(client, agents, task, config, progress_dir, args.workers)
    elif args.mode == "sc":
        payload["sc_k"] = args.sc_k
        payload["sc"] = {}
        for agent in agents:
            payload["sc"][agent.name] = evaluate_self_consistency(
                client, agent, task, config, args.sc_k, args.seed, progress_dir, args.workers
            )
            print(f"[info] sc@{args.sc_k} {agent.name}: acc={payload['sc'][agent.name]['accuracy']:.4f}")
    elif args.mode == "team":
        transcript_path = out_dir / "transcripts_team.json" if args.save_transcripts else None
        payload["team"] = evaluate_team(
            client, agents, task, config, args.rounds, args.seed, transcript_path, progress_dir,
            args.workers,
        )
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
                result = evaluate_solo(client, members, task, config, progress_dir, args.workers)[
                    members[0].name
                ]
            else:
                transcript_path = (
                    out_dir / f"transcripts_{key}.json" if args.save_transcripts else None
                )
                result = evaluate_team(
                    client, members, task, config, args.rounds, args.seed, transcript_path,
                    progress_dir, args.workers,
                )
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
