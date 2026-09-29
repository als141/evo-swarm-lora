"""問題（評価項目）の表現と入出力。問題集合は JSONL ファイルに固定して事前登録する。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

CHOICE_LETTERS = "ABCDEFGHIJ"


@dataclass(frozen=True)
class Item:
    item_id: str
    bench: str  # mmlu_pro / supergpqa / math
    question: str  # モデルに提示する全文（選択肢を含む）
    gold: str  # 選択肢記号、または MATH の元の LaTeX 正解
    answer_type: str  # letter / math
    meta: Dict[str, object] = field(default_factory=dict, hash=False, compare=False)


def format_mc_question(stem: str, options: List[str]) -> str:
    lines = [stem.strip(), ""]
    for letter, option in zip(CHOICE_LETTERS, options):
        lines.append(f"{letter}. {option}")
    return "\n".join(lines)


def save_items(items: List[Item], path: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def load_items(path: str) -> List[Item]:
    items = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(Item(**json.loads(line)))
    return items


def index_items(paths: List[str]) -> Dict[str, Item]:
    index: Dict[str, Item] = {}
    for path in paths:
        for item in load_items(path):
            index[item.item_id] = item
    return index
