"""評価結果の統計検定ユーティリティ。

同一問題セット上の 2 条件比較（対応あり）を前提とする:
- McNemar 検定（二値正誤の対応あり比較の標準）
- 対応ありブートストラップによる精度差の信頼区間
"""

from __future__ import annotations

import random
from math import comb
from typing import Dict, List, Tuple


def paired_outcomes(
    per_item_a: Dict[str, dict],
    per_item_b: Dict[str, dict],
) -> List[Tuple[bool, bool]]:
    common = sorted(set(per_item_a) & set(per_item_b))
    if not common:
        raise ValueError("No common items between the two conditions.")
    return [(per_item_a[i]["correct"], per_item_b[i]["correct"]) for i in common]


def mcnemar_exact(pairs: List[Tuple[bool, bool]]) -> Dict[str, float]:
    """正確二項 McNemar 検定。b = A のみ正解, c = B のみ正解。"""
    b = sum(1 for a_ok, b_ok in pairs if a_ok and not b_ok)
    c = sum(1 for a_ok, b_ok in pairs if not a_ok and b_ok)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    p_value = min(1.0, 2 * tail)
    return {"b": b, "c": c, "p_value": p_value}


def bootstrap_accuracy_diff(
    pairs: List[Tuple[bool, bool]],
    n_resamples: int = 10000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """対応ありブートストラップで (acc_B - acc_A) の信頼区間を推定。"""
    rng = random.Random(seed)
    n = len(pairs)
    diffs: List[float] = []
    for _ in range(n_resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        acc_a = sum(a for a, _ in sample) / n
        acc_b = sum(b for _, b in sample) / n
        diffs.append(acc_b - acc_a)
    diffs.sort()
    alpha = (1 - confidence) / 2
    lower = diffs[int(alpha * n_resamples)]
    upper = diffs[min(int((1 - alpha) * n_resamples), n_resamples - 1)]
    point = sum(b for _, b in pairs) / n - sum(a for a, _ in pairs) / n
    return {"diff": point, "ci_lower": lower, "ci_upper": upper, "confidence": confidence}
