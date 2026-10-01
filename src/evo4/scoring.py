"""v4 の回答抽出と採点（監査済み・オフライン採点用）。

2026-07 実装（src/evalx/tasks.py）で見つかった欠陥を直している:
- `ANSWER\\s*[:：]\\s*(.+)` の `\\s*` が改行をまたぎ、`Final Answer:` の次行の
  `ANSWER: H` を丸ごと捕捉して単語 ANSWER の 'A' を返していた（偽 'A'）。
  → 捕捉は同一行に限定し、空なら次の非空行を見る。
- 捕捉文字列の「最初に現れる A〜J の文字」を返していたため、
  `ANSWER: Option C` が 'I'（OPTION の I）になった。
  → 独立したトークンとしての選択肢記号だけを受理する。
- 数式は簡易な文字列正規化のみだった。
  → math-verify（HF）の記号的同値判定を使い、使えない環境では強化した正規化に落とす。
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from pathlib import Path
from typing import Optional

CHOICE_LETTERS = "ABCDEFGHIJ"

# ANSWER 行: 捕捉は同一行のみ（[^\n]*）。全角コロンも許容。
_ANSWER_LINE = re.compile(r"ANSWER\s*[:：][ \t]*([^\n]*)", re.IGNORECASE)

# 捕捉文字列の先頭にある選択肢記号: "H", "**H**", "(H)", "H.", "H)", "[H]", "\boxed{H}", "$H$"
_LEADING_LETTER = re.compile(
    r"^[\s\*\$\(\[\{]*(?:\\boxed\{|\\text\{|\\textbf\{|\\mathrm\{)?\s*\(?([A-J])\)?(?![A-Za-z])"
)
# 「Option C」「choice (C)」「answer is C」
_KEYWORD_LETTER = re.compile(
    r"(?:option|choice|answer)\s*(?:is\s*)?[:：]?\s*[\*\$\(\[]*([A-J])(?![A-Za-z])", re.IGNORECASE
)
# 独立した大文字1文字トークン（冠詞・一人称 'A'/'I' の誤検出を避けるため最後の手段）
_STANDALONE_LETTER = re.compile(r"(?<![A-Za-z])([A-J])(?![A-Za-z])")

# ANSWER 行が無い場合のフォールバック（本文全体、最後の一致を採用）
_FALLBACK_LETTER = [
    re.compile(
        r"(?:final answer|the answer|correct answer|answer)\s*(?:is|:|：)\s*[\*\$\(\[]*([A-J])(?![A-Za-z])",
        re.IGNORECASE,
    ),
    re.compile(r"\\boxed\{\s*\(?([A-J])\)?\s*\}"),
    re.compile(r"\*\*\(?([A-J])\)?\*\*"),
]


def _answer_line_payload(text: str) -> Optional[str]:
    """最後の ANSWER 行の内容を返す。内容が空なら次の非空行を返す。"""
    matches = list(_ANSWER_LINE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    payload = last.group(1).strip()
    if payload:
        # "ANSWER: ANSWER: X" の二重接頭辞を剥がす
        payload = re.sub(r"^(?:ANSWER\s*[:：]\s*)+", "", payload, flags=re.IGNORECASE).strip()
        if payload:
            return payload
    rest = text[last.end():].splitlines()
    for line in rest:
        if line.strip():
            return line.strip()
    return None


def extract_letter(text: str) -> Optional[str]:
    payload = _answer_line_payload(text)
    if payload is not None:
        match = _LEADING_LETTER.match(payload)
        if match:
            return match.group(1)
        match = _KEYWORD_LETTER.search(payload)
        if match:
            return match.group(1).upper()
        tokens = _STANDALONE_LETTER.findall(payload)
        distinct = sorted(set(tokens))
        if len(distinct) == 1:
            return distinct[0]
        if tokens:
            # 複数あれば 'I'（一人称）を除いた最後の記号を採用
            filtered = [t for t in tokens if t != "I"] or tokens
            return filtered[-1]
    for pattern in _FALLBACK_LETTER:
        found = pattern.findall(text)
        if found:
            return found[-1].upper()
    return None


def last_boxed(text: str) -> Optional[str]:
    """最後の \\boxed{...} の中身を、入れ子の括弧を数えて取り出す。"""
    idx = text.rfind("\\boxed")
    while idx != -1:
        start = text.find("{", idx)
        if start == -1:
            return None
        depth = 0
        for pos in range(start, len(text)):
            ch = text[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start + 1:pos]
        idx = text.rfind("\\boxed", 0, idx)
    return None


def extract_math(text: str) -> Optional[str]:
    """数式回答の抽出。ANSWER 行を優先し、無ければ最後の \\boxed{}。"""
    payload = _answer_line_payload(text)
    if payload is not None:
        inner = last_boxed(payload)
        answer = inner if inner is not None else payload
        answer = answer.strip().strip("$").strip()
        answer = re.sub(r"^\\\(|\\\)$", "", answer).strip()
        answer = answer.rstrip(".").strip()
        return answer or None
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed.strip() or None
    return None


def extract_answer(text: str, answer_type: str) -> Optional[str]:
    if text is None:
        return None
    if answer_type == "letter":
        return extract_letter(text)
    if answer_type == "math":
        return extract_math(text)
    if answer_type == "number":
        payload = _answer_line_payload(text)
        source = payload if payload is not None else text
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", source)
        if not numbers:
            return None
        return numbers[0 if payload is not None else -1].replace(",", "")
    raise ValueError(f"Unknown answer_type '{answer_type}'")


# ---------------------------------------------------------------- math equivalence

def _normalize_math(text: str) -> str:
    s = text.strip()
    inner = last_boxed(s)
    if inner is not None:
        s = inner
    s = re.sub(r"\\text\{\s*([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:mathrm|textbf|mathbf)\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "")
    s = s.replace("\\,", "").replace("\\;", "").replace("\\ ", "").replace("~", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\circ", "")
    s = s.replace("\\%", "").replace("%", "").replace("\\$", "").replace("$", "")
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)  # "x = 3" -> "3"
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\frac(\d)(\d)", r"\1/\2", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt(\d+)", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("{", "").replace("}", "").replace(" ", "").replace("\\", "")
    s = s.rstrip(".")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    try:
        value = float(s)
        return str(int(value)) if value == int(value) else repr(value)
    except ValueError:
        return s.lower()


# math-verify（HF）と固定版の sympy はリポジトリ直下の vendor/ に置く（ジョブではコードスナップショットに同梱、
# 手元では `uv pip install --target vendor ...`、scripts/v4/push_code.sh と同じ版）。どの入口から読み込んでも同じ版を使う。
_VENDOR = Path(__file__).resolve().parents[2] / "vendor"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(1, str(_VENDOR))

try:  # math-verify（HF）: 記号的な同値判定
    from math_verify import parse as _mv_parse
    from math_verify import verify as _mv_verify

    _HAS_MATH_VERIFY = True
    logging.getLogger("math_verify").setLevel(logging.ERROR)
except Exception:  # noqa: BLE001
    _HAS_MATH_VERIFY = False


def _mv_timeout() -> Optional[int]:
    # math-verify のタイムアウトは signal.alarm 実装のため主スレッドでのみ有効
    return 5 if threading.current_thread() is threading.main_thread() else None


def math_equivalent(predicted: str, gold: str) -> bool:
    if predicted is None:
        return False
    if _normalize_math(predicted) == _normalize_math(gold):
        return True
    if _HAS_MATH_VERIFY:
        try:
            timeout = _mv_timeout()
            gold_parsed = _mv_parse(f"${gold}$", parsing_timeout=timeout)
            pred_parsed = _mv_parse(f"${predicted}$", parsing_timeout=timeout)
            if gold_parsed and pred_parsed:
                return bool(_mv_verify(gold_parsed, pred_parsed, timeout_seconds=timeout))
        except Exception:  # noqa: BLE001 - 解析不能は不一致扱い
            return False
    return False


def is_correct(predicted: Optional[str], gold: str, answer_type: str) -> bool:
    if predicted is None:
        return False
    if answer_type == "letter":
        return predicted.strip().upper() == gold.strip().upper()
    if answer_type == "math":
        return math_equivalent(predicted, gold)
    if answer_type == "number":
        try:
            return abs(float(predicted) - float(gold)) < 1e-6
        except ValueError:
            return False
    raise ValueError(f"Unknown answer_type '{answer_type}'")


def scorer_info() -> dict:
    return {"scorer": "evo4.scoring", "math_verify": _HAS_MATH_VERIFY}


def require_math_verify() -> None:
    """オフライン採点の入口で呼ぶ。math-verify が無いと MATH の正答を誤答と数えるため、黙って続けない。"""
    if not _HAS_MATH_VERIFY:
        raise RuntimeError("math-verify が読み込めない。vendor/ を用意すること（scripts/v4/push_code.sh と同じ版）")
