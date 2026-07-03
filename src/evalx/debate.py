"""ベンチマークタスク上での solo 評価と multi-agent debate 評価。

プロトコルは Du et al. (2023) "Improving Factuality and Reasoning in LMs
through Multiagent Debate" に準拠:
  round 0: 各エージェントが独立に回答
  round r: 他エージェントの直前回答を提示し、批判的に再検討して回答を更新
  集約:    最終ラウンドの回答の多数決（同数はシード付き乱数で決定）
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from src.evalx.client import ChatClient, GenerationConfig
from src.evalx.tasks import TaskItem, extract_answer

ANSWER_FORMAT_NUMBER = (
    "Think step by step. Then give your final answer on the last line "
    "in exactly this format:\nANSWER: <number>"
)
ANSWER_FORMAT_LETTER = (
    "Think step by step. Then give your final answer on the last line "
    "in exactly this format:\nANSWER: <letter>"
)


def answer_format_instruction(answer_type: str) -> str:
    return ANSWER_FORMAT_NUMBER if answer_type == "number" else ANSWER_FORMAT_LETTER


@dataclass
class AgentSpec:
    name: str
    model: str  # vLLM に登録された model 名（ベース or LoRA アダプタ名）
    persona_prompt: str = ""


@dataclass
class DebateRecord:
    item_id: str
    rounds: List[Dict[str, str]] = field(default_factory=list)  # round -> {agent: utterance}
    final_answers: Dict[str, Optional[str]] = field(default_factory=dict)
    majority_answer: Optional[str] = None


def _system_prompt(agent: AgentSpec, answer_type: str) -> str:
    parts = []
    if agent.persona_prompt:
        parts.append(agent.persona_prompt)
    parts.append(answer_format_instruction(answer_type))
    return "\n\n".join(parts)


def solo_answer(
    client: ChatClient,
    agent: AgentSpec,
    item: TaskItem,
    answer_type: str,
    config: GenerationConfig,
) -> Dict[str, Optional[str]]:
    messages = [
        {"role": "system", "content": _system_prompt(agent, answer_type)},
        {"role": "user", "content": item.question},
    ]
    utterance = client.chat(agent.model, messages, config)
    return {"utterance": utterance, "answer": extract_answer(utterance, answer_type)}


def _debate_user_prompt(item: TaskItem, others: Dict[str, str]) -> str:
    blocks = [f"Question:\n{item.question}", "\nHere are solutions from other agents:"]
    for name, utterance in others.items():
        blocks.append(f"\n--- Agent {name} ---\n{utterance}")
    blocks.append(
        "\nCarefully examine the other agents' reasoning. Point out any errors, "
        "then provide your own updated step-by-step solution. "
        "You may keep or change your previous answer."
    )
    return "\n".join(blocks)


def run_debate(
    client: ChatClient,
    agents: List[AgentSpec],
    item: TaskItem,
    answer_type: str,
    rounds: int,
    config: GenerationConfig,
    tie_break_seed: int = 0,
) -> DebateRecord:
    record = DebateRecord(item_id=item.item_id)

    def agent_config(agent_idx: int, round_idx: int) -> GenerationConfig:
        # 同一モデル・同一ペルソナの構成（温度サンプリング条件）でも初期回答の
        # 多様性が保たれるよう、エージェント×ラウンドごとに seed をずらす
        if config.seed is None:
            return config
        return replace(config, seed=config.seed * 10000 + agent_idx * 100 + round_idx)

    # round 0: 独立回答
    utterances: Dict[str, str] = {}
    for agent_idx, agent in enumerate(agents):
        result = solo_answer(client, agent, item, answer_type, agent_config(agent_idx, 0))
        utterances[agent.name] = result["utterance"]
    record.rounds.append(dict(utterances))

    # round 1..R: 他者の回答を見て更新
    for round_idx in range(1, rounds + 1):
        updated: Dict[str, str] = {}
        for agent_idx, agent in enumerate(agents):
            others = {name: text for name, text in utterances.items() if name != agent.name}
            messages = [
                {"role": "system", "content": _system_prompt(agent, answer_type)},
                {"role": "user", "content": _debate_user_prompt(item, others)},
            ]
            updated[agent.name] = client.chat(
                agent.model, messages, agent_config(agent_idx, round_idx)
            )
        utterances = updated
        record.rounds.append(dict(utterances))

    for agent in agents:
        record.final_answers[agent.name] = extract_answer(utterances[agent.name], answer_type)

    record.majority_answer = majority_vote(list(record.final_answers.values()), tie_break_seed)
    return record


def majority_vote(answers: List[Optional[str]], tie_break_seed: int = 0) -> Optional[str]:
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    top_count = max(counts.values())
    winners = sorted(a for a, c in counts.items() if c == top_count)
    if len(winners) == 1:
        return winners[0]
    return random.Random(tie_break_seed).choice(winners)
