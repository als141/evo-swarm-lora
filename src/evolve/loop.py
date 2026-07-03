"""協調的共進化ループ (docs/research_design.md §3-4)。

- 3 役割 × 各 K 個体のサブ集団 (Potter & De Jong 1994 の協調的共進化)
- 適応度 = 代表チーム文脈での厳密 Shapley 値 × fitness sharing ペナルティ
- 交叉 = ΔW 空間ブレンド + SVD 再分解、突然変異 = ガウスノイズ
- vLLM の動的 LoRA ロード (/v1/load_lora_adapter) を前提に、
  世代ごとに子アダプタを生成・登録・評価する
"""

from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from src.evalx.client import ChatClient, GenerationConfig
from src.evalx.debate import AgentSpec, run_debate, solo_answer
from src.evalx.shapley import shapley_values
from src.evalx.tasks import TaskSpec, score_predictions
from src.models.lora_ops import delta_blend_lora, mutate_lora


@dataclass
class Individual:
    name: str          # vLLM 上のアダプタ名（例: gen1_critic_0）
    role: str          # critic / pragmatist / explorer
    adapter_dir: str
    persona_prompt: str


class VllmLoraManager:
    """vLLM の動的 LoRA ロード API のラッパー。"""

    def __init__(self, base_url: str):
        # base_url は http://host:port/v1 を想定
        self._root = base_url.rstrip("/").removesuffix("/v1")
        self._loaded: set[str] = set()

    def load(self, name: str, path: str) -> None:
        if name in self._loaded:
            return
        response = httpx.post(
            f"{self._root}/v1/load_lora_adapter",
            json={"lora_name": name, "lora_path": str(Path(path).resolve())},
            timeout=120.0,
        )
        # 別プロセスが先に登録済みのケース（バッテリー実行）は成功として扱う
        if response.status_code not in (200, 201) and "already" not in response.text.lower():
            raise RuntimeError(f"Failed to load adapter {name}: {response.status_code} {response.text}")
        self._loaded.add(name)

    def unload(self, name: str) -> None:
        if name not in self._loaded:
            return
        httpx.post(
            f"{self._root}/v1/unload_lora_adapter",
            json={"lora_name": name},
            timeout=60.0,
        )
        self._loaded.discard(name)


class CoalitionEvaluator:
    """適応度セット上で連合（solo / debate チーム）の精度を評価し、結果をキャッシュする。"""

    def __init__(
        self,
        client: ChatClient,
        task: TaskSpec,
        config: GenerationConfig,
        rounds: int,
        tie_break_seed: int,
    ):
        self._client = client
        self._task = task
        self._config = config
        self._rounds = rounds
        self._tie_break_seed = tie_break_seed
        self.cache: Dict[frozenset, dict] = {}

    def evaluate(self, members: List[Individual]) -> dict:
        key = frozenset(m.name for m in members)
        if key in self.cache:
            return self.cache[key]
        agents = [AgentSpec(name=m.name, model=m.name, persona_prompt=m.persona_prompt) for m in members]
        predictions = {}
        if len(agents) == 1:
            for item in self._task.items:
                result = solo_answer(self._client, agents[0], item, self._task.answer_type, self._config)
                predictions[item.item_id] = result["answer"]
        else:
            for item in self._task.items:
                record = run_debate(
                    self._client, agents, item, self._task.answer_type,
                    self._rounds, self._config, self._tie_break_seed,
                )
                predictions[item.item_id] = record.majority_answer
        result = score_predictions(self._task.items, predictions, self._task.answer_type)
        self.cache[key] = result
        return result


def behavioral_distance(result_a: dict, result_b: dict) -> float:
    """solo 評価の per_item 予測の不一致率（同役割個体間の行動距離）。"""
    items = set(result_a["per_item"]) & set(result_b["per_item"])
    if not items:
        return 1.0
    disagree = sum(
        1
        for i in items
        if result_a["per_item"][i]["predicted"] != result_b["per_item"][i]["predicted"]
    )
    return disagree / len(items)


def sharing_penalty(distances: List[float], sigma: float) -> float:
    """Goldberg & Richardson (1987) の fitness sharing。近い個体が多いほど割引。"""
    niche_count = 1.0 + sum(max(0.0, 1.0 - d / sigma) for d in distances)
    return 1.0 / niche_count


def compute_candidate_fitness(
    candidate: Individual,
    representatives: Dict[str, Individual],
    evaluator: CoalitionEvaluator,
) -> dict:
    """代表チーム文脈 {candidate} ∪ {他役割の代表} での厳密 Shapley 値を計算する。"""
    others = [rep for role, rep in sorted(representatives.items()) if role != candidate.role]
    team = [candidate, *others]
    names = [m.name for m in team]
    by_name = {m.name: m for m in team}

    coalition_values = {}
    coalition_accuracy = {}
    from src.evalx.shapley import all_coalitions

    for coalition in all_coalitions(names):
        members = [by_name[n] for n in coalition]
        result = evaluator.evaluate(members)
        coalition_values[frozenset(coalition)] = result["accuracy"]
        coalition_accuracy["+".join(sorted(coalition))] = result["accuracy"]

    values = shapley_values(names, coalition_values)
    return {
        "shapley": values[candidate.name],
        "solo_accuracy": coalition_values[frozenset([candidate.name])],
        "team_accuracy": coalition_values[frozenset(names)],
        "coalition_accuracy": coalition_accuracy,
        "team_members": names,
    }


def make_child(
    parent_primary: Individual,
    parent_secondary: Individual,
    child_name: str,
    child_dir: str,
    alpha: float,
    mut_ratio: float,
    mut_std: float,
    seed: int,
    crossover: str = "delta",
) -> None:
    """交叉 + 突然変異で子アダプタを生成する。"""
    from src.models.lora_ops import alpha_blend_lora

    tmp_dir = f"{child_dir}_blend_tmp"
    if Path(tmp_dir).exists():
        shutil.rmtree(tmp_dir)
    if crossover == "delta":
        delta_blend_lora(parent_primary.adapter_dir, parent_secondary.adapter_dir, tmp_dir, alpha)
    elif crossover == "naive":
        alpha_blend_lora(parent_primary.adapter_dir, parent_secondary.adapter_dir, tmp_dir, alpha)
    else:
        raise ValueError(f"Unknown crossover '{crossover}'")
    if Path(child_dir).exists():
        shutil.rmtree(child_dir)
    mutate_lora(tmp_dir, child_dir, ratio=mut_ratio, std=mut_std, seed=seed)
    shutil.rmtree(tmp_dir)


def run_generation(
    generation: int,
    subpopulations: Dict[str, List[Individual]],
    representatives: Dict[str, Individual],
    evaluator: CoalitionEvaluator,
    lora_manager: VllmLoraManager,
    out_root: Path,
    fitness_mode: str = "shapley",
    sharing_sigma: float = 0.3,
    use_sharing: bool = True,
) -> dict:
    """1 世代の評価と選抜。次世代の代表と世代ログを返す。"""
    started_at = time.time()
    generation_log: dict = {"generation": generation, "roles": {}}

    # 全個体を vLLM に登録
    for role_members in subpopulations.values():
        for individual in role_members:
            lora_manager.load(individual.name, individual.adapter_dir)
    for rep in representatives.values():
        lora_manager.load(rep.name, rep.adapter_dir)

    new_representatives: Dict[str, Individual] = {}
    for role, members in sorted(subpopulations.items()):
        role_log = {"candidates": {}}
        fitness_records = {}
        for candidate in members:
            record = compute_candidate_fitness(candidate, representatives, evaluator)
            fitness_records[candidate.name] = record

        # fitness sharing: 同役割個体間の行動距離で割引
        for candidate in members:
            record = fitness_records[candidate.name]
            raw = record["shapley"] if fitness_mode == "shapley" else record["solo_accuracy"]
            if use_sharing and len(members) > 1:
                solo_result = evaluator.cache[frozenset([candidate.name])]
                distances = [
                    behavioral_distance(solo_result, evaluator.cache[frozenset([other.name])])
                    for other in members
                    if other.name != candidate.name
                ]
                penalty = sharing_penalty(distances, sharing_sigma)
            else:
                distances, penalty = [], 1.0
            record["raw_fitness"] = raw
            record["sharing_distances"] = distances
            record["sharing_penalty"] = penalty
            record["fitness"] = raw * penalty
            role_log["candidates"][candidate.name] = record

        best = max(members, key=lambda m: fitness_records[m.name]["fitness"])
        role_log["selected"] = best.name
        new_representatives[role] = best
        generation_log["roles"][role] = role_log

    generation_log["elapsed_seconds"] = time.time() - started_at
    generation_log["representatives"] = {r: ind.name for r, ind in new_representatives.items()}

    log_path = out_root / f"gen_{generation:02d}" / "generation_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(generation_log, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"representatives": new_representatives, "log": generation_log}


def breed_next_generation(
    generation: int,
    representatives: Dict[str, Individual],
    subpopulations: Dict[str, List[Individual]],
    fitness_by_name: Dict[str, float],
    out_root: Path,
    rng: random.Random,
    mut_ratio: float,
    mut_std: float,
    crossover: str = "delta",
    persona_prompts: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Individual]]:
    """次世代のサブ集団を生成する。各役割: [エリート代表, 交叉+突然変異の子]。"""
    next_subpopulations: Dict[str, List[Individual]] = {}
    gen_dir = out_root / f"gen_{generation + 1:02d}"
    for role, rep in sorted(representatives.items()):
        members = subpopulations[role]
        # 交叉相手: 同役割で代表以外の最良個体（いなければ代表自身の複製に突然変異）
        partners = [m for m in members if m.name != rep.name]
        partner = max(partners, key=lambda m: fitness_by_name.get(m.name, 0.0)) if partners else rep

        child_name = f"gen{generation + 1}_{role}_child"
        child_dir = gen_dir / child_name
        make_child(
            rep,
            partner,
            child_name,
            str(child_dir),
            alpha=rng.uniform(0.3, 0.7),
            mut_ratio=mut_ratio,
            mut_std=mut_std,
            seed=rng.randrange(2**31),
            crossover=crossover,
        )
        persona = (persona_prompts or {}).get(role, rep.persona_prompt)
        next_subpopulations[role] = [
            rep,  # エリート保存
            Individual(name=child_name, role=role, adapter_dir=str(child_dir), persona_prompt=persona),
        ]
    return next_subpopulations
