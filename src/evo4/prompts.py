"""ペルソナと議論プロトコル（v4）。

ペルソナは 2026-07 の3役割（批判的検証者・実務的意思決定者・発散的探索者）を継承し、
英語の課題に合わせて英語で、推論の進め方が異なるように記述する（役割名は同じ）。

round1 のプロトコル:
- "v4": Du et al. (2023) の原型どおり、自分の前回解答を会話履歴（assistant ターン）として保持し、
        他者の解答を次の user ターンで提示する。更新指示は中立。
- "v4c": v4 と同じ会話構造で、「自分の推論に具体的な誤りを見つけた場合のみ変更」の条件付き指示。
- "july": 2026-07 の c7 プロトコル（自分の前回解答を提示しない・条件付き指示・匿名化）。対照用。
- "v4t" / "v4ct": v4 / v4c と同じ構造で、自分と他者の回答を末尾 TRUNCATE_CHARS 文字に切り詰める。
  パイロットで、round1 の入力（平均 7.6K トークン）が A100 40GB の KV キャッシュを埋め、同時に約20本しか
  処理できず、議論が予算内に収まらないことが判明した（2026-09-30）。結論と最後の計算は回答の末尾にあるため、
  末尾を残す。
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

PERSONAS: Dict[str, str] = {
    "critic": (
        "You are a rigorous critical verifier. You check every step for errors, hidden "
        "assumptions and edge cases, test candidate answers against the conditions of the "
        "problem, and eliminate options that fail a check."
    ),
    "pragmatist": (
        "You are a pragmatic, decisive problem solver. You pick the most direct method, keep "
        "the reasoning focused, and sanity-check the result with a quick estimate or "
        "plausibility check before committing to an answer."
    ),
    "explorer": (
        "You are a divergent, creative thinker. You consider alternative interpretations and "
        "more than one way to solve the problem, compare where the approaches agree or "
        "disagree, and settle on the best-supported answer."
    ),
}

# 2026-07 の c7 で使った日本語ペルソナ（対照・再現用）
PERSONAS_JULY: Dict[str, str] = {
    "critic": "あなたは厳密な検証を重視する批判的思考家。反証・例外・境界条件に敏感。",
    "pragmatist": "あなたは応用志向の実務家。意思決定に役立つ実装可能性とコストを重視。",
    "explorer": "あなたは創発を促す発想家。仮説生成と多角的比喩で発想を広げる。",
}

ANSWER_FORMATS = {
    "letter": "Think step by step. Then give your final answer on the last line in exactly this format:\nANSWER: <letter>",
    "math": "Think step by step. Then give your final answer on the last line in exactly this format:\nANSWER: <final simplified answer>",
    "number": "Think step by step. Then give your final answer on the last line in exactly this format:\nANSWER: <number>",
}

TRUNCATE_CHARS = 3000


def truncate_tail(text: str, limit: int = TRUNCATE_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return "[... earlier part of the solution omitted ...]\n" + text[-limit:]


R1_INSTRUCTIONS = {
    "v4": (
        "Compare their reasoning with your own solution above. Re-check the key steps and point "
        "out any errors, whether in your solution or in theirs. Then give your updated "
        "step-by-step solution. You may keep or change your answer."
    ),
    "v4c": (
        "Compare their reasoning with your own solution above. Re-check the key steps of your own "
        "solution. Change your answer only if you find a concrete error in your own reasoning, and "
        "state that error explicitly; otherwise keep your answer even if the other agents disagree. "
        "Then give your final step-by-step solution."
    ),
    "v4t": None,  # v4 と同じ文言（下で設定）
    "v4ct": None,
    "july": (
        "\nCarefully examine the other agents' reasoning and compare it with your own. "
        "First, briefly re-derive the key steps of your own solution. "
        "Change your answer ONLY if you can identify a concrete, specific error in your "
        "own reasoning, and state that error explicitly. If you cannot find a specific "
        "error in your own reasoning, keep your original answer even if the other agents "
        "disagree. Then provide your final step-by-step solution."
    ),
}


R1_INSTRUCTIONS["v4t"] = R1_INSTRUCTIONS["v4"]
R1_INSTRUCTIONS["v4ct"] = R1_INSTRUCTIONS["v4c"]


def system_prompt(persona: str, answer_type: str) -> str:
    parts = [persona] if persona else []
    parts.append(ANSWER_FORMATS[answer_type])
    return "\n\n".join(parts)


def r0_messages(persona: str, question: str, answer_type: str) -> List[dict]:
    return [
        {"role": "system", "content": system_prompt(persona, answer_type)},
        {"role": "user", "content": question},
    ]


def _others_block(others: Sequence[str], shuffle_seed: int) -> str:
    entries = list(others)
    random.Random(shuffle_seed).shuffle(entries)  # 提示順の偏り（先頭効果）を避ける
    blocks = []
    for idx, text in enumerate(entries, start=1):
        blocks.append(f"--- Agent {idx} ---\n{text.strip()}")
    return "\n\n".join(blocks)


def r1_messages(protocol: str, persona: str, question: str, answer_type: str, own_r0: str,
                others_r0: Sequence[str], shuffle_seed: int) -> List[dict]:
    if protocol in ("v4t", "v4ct"):
        own_r0 = truncate_tail(own_r0)
        others_r0 = [truncate_tail(t) for t in others_r0]
    if protocol in ("v4", "v4c", "v4t", "v4ct"):
        user = (
            "Here are solutions to the same problem from other agents:\n\n"
            + _others_block(others_r0, shuffle_seed)
            + "\n\n" + R1_INSTRUCTIONS[protocol]
        )
        return [
            {"role": "system", "content": system_prompt(persona, answer_type)},
            {"role": "user", "content": question},
            {"role": "assistant", "content": own_r0},
            {"role": "user", "content": user},
        ]
    if protocol == "july":
        blocks = [f"Question:\n{question}", "\nHere are solutions from other agents:"]
        entries = list(others_r0)
        random.Random(shuffle_seed).shuffle(entries)
        for idx, text in enumerate(entries, start=1):
            blocks.append(f"\n--- Agent {idx} ---\n{text}")
        blocks.append(R1_INSTRUCTIONS["july"])
        return [
            {"role": "system", "content": system_prompt(persona, answer_type)},
            {"role": "user", "content": "\n".join(blocks)},
        ]
    raise ValueError(f"unknown protocol '{protocol}'")
