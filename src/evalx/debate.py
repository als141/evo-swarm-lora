"""ベンチマークタスク上での solo 評価と multi-agent debate 評価。

プロトコルは Du et al. (2023) "Improving Factuality and Reasoning in LMs
through Multiagent Debate" に準拠:
  round 0: 各エージェントが独立に回答
  round r: 他エージェントの直前回答を提示し、批判的に再検討して回答を更新
  集約:    最終ラウンドの回答の多数決（同数はシード付き乱数で決定）
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

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
ANSWER_FORMAT_MATH = (
    "Think step by step. Then give your final answer on the last line "
    "in exactly this format:\nANSWER: <final simplified answer>"
)

_ANSWER_FORMATS = {
    "number": ANSWER_FORMAT_NUMBER,
    "letter": ANSWER_FORMAT_LETTER,
    "math": ANSWER_FORMAT_MATH,
}


def answer_format_instruction(answer_type: str) -> str:
    # 旧実装は number/letter の二分岐で math が letter に落ち、MATH-500 で
    # 「ANSWER: <letter>」と指示してしまう致命的バグがあった（run001 v3 で実測）
    if answer_type not in _ANSWER_FORMATS:
        raise ValueError(f"Unknown answer_type '{answer_type}'")
    return _ANSWER_FORMATS[answer_type]


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
    # v3集約用の付帯情報（aggregation="majority" の従来経路では未使用）
    confidences: Dict[str, Optional[float]] = field(default_factory=dict)
    aggregation: str = "majority"
    adjudicated: bool = False  # GenSelect裁定が発動したか


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


# 更新指示のスタイル。standard は従来どおり（本実験v3と同一文言、変更禁止）。
# conditional は追従(sycophancy)対策: 自分の推論に具体的誤りを特定できた場合のみ
# 回答変更を許可する条件付き更新（confidence-conditioned update 系文献に基づく。
# v3実測では議論が正しい多数派を壊す事例が500問中26問あった）。
DEBATE_UPDATE_INSTRUCTIONS = {
    "standard": (
        "\nCarefully examine the other agents' reasoning. Point out any errors, "
        "then provide your own updated step-by-step solution. "
        "You may keep or change your previous answer."
    ),
    "conditional": (
        "\nCarefully examine the other agents' reasoning and compare it with your own. "
        "First, briefly re-derive the key steps of your own solution. "
        "Change your answer ONLY if you can identify a concrete, specific error in your "
        "own reasoning, and state that error explicitly. If you cannot find a specific "
        "error in your own reasoning, keep your original answer even if the other agents "
        "disagree. Then provide your final step-by-step solution."
    ),
}


def _debate_user_prompt(
    item: TaskItem,
    others: Dict[str, str],
    style: str = "standard",
    anonymize: bool = False,
    shuffle_seed: Optional[int] = None,
) -> str:
    """anonymize=True では役割・人格ラベルを匿名番号に置換し提示順もシャッフルする。

    sycophancy は self-bias より優勢で、発言の匿名化だけで発言の重みが均等化する
    (arXiv:2510.07517)。既定 False で v3 プロトコルの文言を完全維持。
    """
    blocks = [f"Question:\n{item.question}", "\nHere are solutions from other agents:"]
    entries = list(others.items())
    if anonymize:
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(entries)
        for idx, (_, utterance) in enumerate(entries, start=1):
            blocks.append(f"\n--- Agent {idx} ---\n{utterance}")
    else:
        for name, utterance in entries:
            blocks.append(f"\n--- Agent {name} ---\n{utterance}")
    blocks.append(DEBATE_UPDATE_INSTRUCTIONS[style])
    return "\n".join(blocks)


GENSELECT_INSTRUCTION = (
    "You are given a question and {k} candidate solutions written by different assistants. "
    "Carefully compare the candidates step by step: check their reasoning for errors and "
    "verify their final answers where possible. Then decide which candidate is most likely "
    "correct. Answer on the last line in exactly this format:\nBEST: <candidate number>"
)


def genselect_adjudicate(
    client: ChatClient,
    judge_model: str,
    item: TaskItem,
    candidates: List[Tuple[Optional[str], str]],
    config: GenerationConfig,
    shuffle_seed: int = 0,
) -> Optional[str]:
    """GenSelect型の比較選択裁定 (arXiv:2507.17797 / 2602.09341)。

    candidates は (抽出済み回答, 発話全文) のリスト。匿名・順序ランダム化して1コンテキストに
    並べ、judge_model に最良候補を選ばせ、その候補の抽出済み回答を返す。
    4B級は採点型judgeには信頼性不足だが比較選択形式なら機能する (arXiv:2606.19544)。
    選択が解析できない場合は None（呼び出し側で多数決へフォールバック）。
    """
    order = list(range(len(candidates)))
    random.Random(shuffle_seed).shuffle(order)
    blocks = [f"Question:\n{item.question}", ""]
    for display_idx, cand_idx in enumerate(order, start=1):
        _, utterance = candidates[cand_idx]
        blocks.append(f"--- Candidate {display_idx} ---\n{utterance}\n")
    messages = [
        {"role": "system", "content": GENSELECT_INSTRUCTION.format(k=len(candidates))},
        {"role": "user", "content": "\n".join(blocks)},
    ]
    text = client.chat(judge_model, messages, config)
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("BEST"):
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                picked = int(digits[0])
                if 1 <= picked <= len(candidates):
                    return candidates[order[picked - 1]][0]
            break
    return None


def weighted_vote(
    answers: List[Tuple[Optional[str], Optional[float]]], tie_break_seed: int = 0
) -> Optional[str]:
    """logprob由来confidenceによる重み付き多数決 (DeepConf/CISC系)。

    confidence が None の票は重み1.0（プレーン票）として扱う。
    """
    weights: Dict[str, float] = defaultdict(float)
    for answer, confidence in answers:
        if answer is None:
            continue
        weights[answer] += confidence if confidence is not None else 1.0
    if not weights:
        return None
    top = max(weights.values())
    winners = sorted(a for a, w in weights.items() if abs(w - top) < 1e-12)
    if len(winners) == 1:
        return winners[0]
    return random.Random(tie_break_seed).choice(winners)


def run_debate(
    client: ChatClient,
    agents: List[AgentSpec],
    item: TaskItem,
    answer_type: str,
    rounds: int,
    config: GenerationConfig,
    tie_break_seed: int = 0,
    update_style: str = "standard",
    aggregation: str = "majority",
    anonymize: bool = False,
    judge_model: Optional[str] = None,
) -> DebateRecord:
    """aggregation: "majority"（従来） / "weighted"（logprob重み付き投票） /
    "genselect"（票が割れた問題のみ judge_model による比較選択裁定、要 judge_model）。
    anonymize: 議論プロンプトの発言者ラベルを匿名番号化+提示順シャッフル。
    既定値はすべて v3 プロトコルと同一動作。
    """
    record = DebateRecord(item_id=item.item_id, aggregation=aggregation)
    want_conf = aggregation in ("weighted", "genselect")

    def call_agent(model: str, messages: List[dict], cfg: GenerationConfig) -> Tuple[str, Optional[float]]:
        # confidence 不要時は従来の chat を使う（chat のみ実装するクライアントとの互換維持）
        if want_conf:
            result = client.chat_scored(model, messages, cfg, with_logprobs=True)
            return result.text, result.tail_confidence
        return client.chat(model, messages, cfg), None

    def agent_config(agent_idx: int, round_idx: int) -> GenerationConfig:
        # 同一モデル・同一ペルソナの構成（温度サンプリング条件）でも初期回答の
        # 多様性が保たれるよう、エージェント×ラウンドごとに seed をずらす
        if config.seed is None:
            return config
        return replace(config, seed=config.seed * 10000 + agent_idx * 100 + round_idx)

    # round 0: 独立回答
    utterances: Dict[str, str] = {}
    for agent_idx, agent in enumerate(agents):
        messages = [
            {"role": "system", "content": _system_prompt(agent, answer_type)},
            {"role": "user", "content": item.question},
        ]
        text, conf = call_agent(agent.model, messages, agent_config(agent_idx, 0))
        utterances[agent.name] = text
        record.confidences[agent.name] = conf
    record.rounds.append(dict(utterances))

    # round 1..R: 他者の回答を見て更新
    for round_idx in range(1, rounds + 1):
        updated: Dict[str, str] = {}
        for agent_idx, agent in enumerate(agents):
            others = {name: text for name, text in utterances.items() if name != agent.name}
            shuffle_seed = (
                None if config.seed is None else config.seed * 10000 + agent_idx * 100 + round_idx
            )
            messages = [
                {"role": "system", "content": _system_prompt(agent, answer_type)},
                {
                    "role": "user",
                    "content": _debate_user_prompt(
                        item, others, update_style, anonymize=anonymize, shuffle_seed=shuffle_seed
                    ),
                },
            ]
            text, conf = call_agent(agent.model, messages, agent_config(agent_idx, round_idx))
            updated[agent.name] = text
            record.confidences[agent.name] = conf
        utterances = updated
        record.rounds.append(dict(utterances))

    for agent in agents:
        record.final_answers[agent.name] = extract_answer(utterances[agent.name], answer_type)

    answers = list(record.final_answers.values())
    record.majority_answer = majority_vote(answers, tie_break_seed)

    if aggregation == "weighted":
        record.majority_answer = weighted_vote(
            [(record.final_answers[a.name], record.confidences.get(a.name)) for a in agents],
            tie_break_seed,
        )
    elif aggregation == "genselect":
        valid = [a for a in answers if a is not None]
        unanimous = len(set(valid)) <= 1 and len(valid) == len(agents)
        if not unanimous and judge_model is not None:
            picked = genselect_adjudicate(
                client,
                judge_model,
                item,
                [(record.final_answers[a.name], utterances[a.name]) for a in agents],
                config,
                shuffle_seed=tie_break_seed,
            )
            record.adjudicated = True
            if picked is not None:
                record.majority_answer = picked
            else:  # 裁定の解析失敗は重み付き投票へフォールバック
                record.majority_answer = weighted_vote(
                    [(record.final_answers[a.name], record.confidences.get(a.name)) for a in agents],
                    tie_break_seed,
                )
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
