"""ベンチマークタスクのロード・回答抽出・採点。

評価は生成ベース（generative）で統一する:
- 数値/数式回答タスク (GSM8K, MATH-500): 最終行の "ANSWER: <expr>" を抽出し正規化比較
- 選択肢タスク (MMLU-Pro, SuperGPQA, ARC): "ANSWER: <letter>" を抽出し文字比較

ベンチマーク選定の根拠 (docs/research_design.md 参照):
- GSM8K は Qwen3-4B で飽和気味 (80-92%) のためスモークテスト用途のみ
- 主要評価: MMLU-Pro (base 69.6) / MATH-500 Level4-5 / SuperGPQA (base 42.8)
- GPQA-Diamond は HF ゲート付きデータセットのため不採用（HFトークン不要方針）

lm-evaluation-harness の logprob 方式ではなく生成方式を採るのは、
solo 評価と debate（協調）評価を同一プロトコルで比較するため。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from datasets import load_dataset

CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


@dataclass
class TaskItem:
    item_id: str
    question: str
    gold: str
    choices: List[str] = field(default_factory=list)

    @property
    def is_multiple_choice(self) -> bool:
        return bool(self.choices)


@dataclass
class TaskSpec:
    name: str
    answer_type: str  # "number" | "letter" | "math"
    items: List[TaskItem]


def _format_mc_question(stem: str, choices: List[str]) -> str:
    lines = [stem.strip(), ""]
    for letter, choice in zip(CHOICE_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


def _normalize_number(text: str) -> Optional[str]:
    cleaned = text.replace(",", "").replace("$", "").replace("%", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    value = match.group()
    try:
        number = float(value)
    except ValueError:
        return None
    if number == int(number):
        return str(int(number))
    return str(number)


def normalize_math_answer(text: str) -> str:
    """MATH 形式の回答（LaTeX 混じり）を比較可能な文字列へ正規化する。"""
    answer = text.strip()
    boxed = re.search(r"\\boxed\{(.+)\}", answer)
    if boxed:
        answer = boxed.group(1)
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    answer = answer.replace("\\$", "").replace("$", "")
    answer = answer.replace("\\%", "").replace("%", "")
    answer = answer.replace("\\text{", "").replace("\\mathrm{", "")
    # \frac{a}{b} -> a/b, \dfrac も同様
    answer = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", answer)
    answer = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", answer)
    answer = answer.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    answer = answer.replace("{", "").replace("}", "").replace(" ", "")
    answer = answer.replace("\\", "")
    answer = answer.rstrip(".")
    # 純数値なら数値として正規化（12.0 == 12 等）
    numeric = _normalize_number(answer)
    if numeric is not None and re.fullmatch(r"-?[\d,.]+", answer):
        return numeric
    return answer.lower()


def load_gsm8k(n: int, seed: int, split: str = "test") -> TaskSpec:
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    items: List[TaskItem] = []
    for idx in indices[:n]:
        row = dataset[idx]
        gold_raw = row["answer"].split("####")[-1]
        gold = _normalize_number(gold_raw)
        if gold is None:
            continue
        items.append(TaskItem(item_id=f"gsm8k-{split}-{idx}", question=row["question"], gold=gold))
    return TaskSpec(name="gsm8k", answer_type="number", items=items)


def load_mmlu_pro(n: int, seed: int, split: str = "test") -> TaskSpec:
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    items: List[TaskItem] = []
    for idx in indices[:n]:
        row = dataset[idx]
        choices = list(row["options"])
        question = _format_mc_question(row["question"], choices)
        items.append(
            TaskItem(
                item_id=f"mmlupro-{split}-{row['question_id']}",
                question=question,
                gold=row["answer"].strip().upper(),
                choices=choices,
            )
        )
    return TaskSpec(name="mmlu_pro", answer_type="letter", items=items)


def load_math500(n: int, seed: int, split: str = "test", min_level: int = 4) -> TaskSpec:
    dataset = load_dataset("HuggingFaceH4/MATH-500", split=split)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    items: List[TaskItem] = []
    for idx in indices:
        if len(items) >= n:
            break
        row = dataset[idx]
        if int(row["level"]) < min_level:
            continue
        items.append(
            TaskItem(
                item_id=f"math500-{split}-{idx}",
                question=row["problem"],
                gold=normalize_math_answer(row["answer"]),
            )
        )
    return TaskSpec(name="math500", answer_type="math", items=items)


def load_supergpqa(n: int, seed: int, split: str = "train") -> TaskSpec:
    dataset = load_dataset("m-a-p/SuperGPQA", split=split)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    items: List[TaskItem] = []
    for idx in indices[:n]:
        row = dataset[idx]
        choices = list(row["options"])
        question = _format_mc_question(row["question"], choices)
        items.append(
            TaskItem(
                item_id=f"supergpqa-{row['uuid']}",
                question=question,
                gold=row["answer_letter"].strip().upper(),
                choices=choices,
            )
        )
    return TaskSpec(name="supergpqa", answer_type="letter", items=items)


def load_arc_challenge(n: int, seed: int, split: str = "test") -> TaskSpec:
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    items: List[TaskItem] = []
    for idx in indices[:n]:
        row = dataset[idx]
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        # ラベルが "1"-"4" の問題があるため A-D へ正規化する
        label_map = {label: CHOICE_LETTERS[pos] for pos, label in enumerate(labels)}
        gold = label_map.get(row["answerKey"])
        if gold is None:
            continue
        question = _format_mc_question(row["question"], texts)
        items.append(
            TaskItem(item_id=f"arc-{split}-{idx}", question=question, gold=gold, choices=texts)
        )
    return TaskSpec(name="arc_challenge", answer_type="letter", items=items)


TASK_LOADERS = {
    "gsm8k": load_gsm8k,
    "mmlu_pro": load_mmlu_pro,
    "math500": load_math500,
    "supergpqa": load_supergpqa,
    "arc_challenge": load_arc_challenge,
}


def load_task(name: str, n: int, seed: int, **kwargs) -> TaskSpec:
    if name not in TASK_LOADERS:
        raise ValueError(f"Unknown task '{name}'. Available: {sorted(TASK_LOADERS)}")
    return TASK_LOADERS[name](n=n, seed=seed, **kwargs)


ANSWER_LINE_PATTERN = re.compile(r"ANSWER\s*[:：]\s*(.+)", re.IGNORECASE)
# フォールバック時に "I"（一人称）や "A"（冠詞）を誤検出しないよう文脈を要求する
LETTER_FALLBACK_PATTERNS = [
    re.compile(r"(?:answer|option|choice)\s+(?:is\s+)?\(?([A-J])\)?\b", re.IGNORECASE),
    re.compile(r"\(([A-J])\)"),
]


def extract_answer(text: str, answer_type: str) -> Optional[str]:
    """モデル出力から最終回答を抽出する。ANSWER: 行を最優先し、無ければフォールバック。"""
    candidates = ANSWER_LINE_PATTERN.findall(text)
    tail = candidates[-1].strip() if candidates else None

    if answer_type == "letter":
        if tail:
            match = re.search(r"[A-J]", tail.upper())
            if match:
                return match.group()
        for pattern in LETTER_FALLBACK_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                return matches[-1].upper()
        return None

    if answer_type == "number":
        if tail:
            normalized = _normalize_number(tail)
            if normalized is not None:
                return normalized
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        if numbers:
            return _normalize_number(numbers[-1])
        return None

    if answer_type == "math":
        if tail:
            return normalize_math_answer(tail)
        boxed = re.findall(r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
        if boxed:
            return normalize_math_answer(boxed[-1])
        return None

    raise ValueError(f"Unknown answer_type '{answer_type}'")


def is_correct(predicted: Optional[str], gold: str, answer_type: str) -> bool:
    if predicted is None:
        return False
    if answer_type == "number":
        try:
            return abs(float(predicted) - float(gold)) < 1e-6
        except ValueError:
            return False
    if answer_type == "math":
        if predicted == gold:
            return True
        try:
            return abs(float(predicted) - float(gold)) < 1e-6
        except ValueError:
            return False
    return predicted.strip().upper() == gold.strip().upper()


def score_predictions(items: List[TaskItem], predictions: Dict[str, Optional[str]], answer_type: str) -> Dict:
    per_item = {}
    correct = 0
    for item in items:
        pred = predictions.get(item.item_id)
        ok = is_correct(pred, item.gold, answer_type)
        per_item[item.item_id] = {"predicted": pred, "gold": item.gold, "correct": ok}
        correct += int(ok)
    accuracy = correct / len(items) if items else 0.0
    return {"accuracy": accuracy, "n": len(items), "per_item": per_item}
