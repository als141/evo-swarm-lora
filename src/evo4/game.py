"""協力ゲームの貢献度（3 人ゲーム用の厳密 Shapley 値と Banzhaf 値）。

特性関数 v(S) は連合 S の dev 精度（v(∅)=0）。3 人なら全 7 連合を実測でき、近似は不要。
Banzhaf 値は全部分集合への限界貢献の単純平均で、特性関数にノイズがある場合に
順位が安定しやすい（Wang & Jia, AISTATS 2023）。同じ 7 連合から追加費用なしで計算できる。
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Dict, FrozenSet, Sequence


def _subsets(players: Sequence[str]):
    for size in range(len(players) + 1):
        for combo in combinations(players, size):
            yield frozenset(combo)


def shapley(players: Sequence[str], v: Dict[FrozenSet[str], float]) -> Dict[str, float]:
    n = len(players)
    values = {}
    for p in players:
        others = [q for q in players if q != p]
        total = 0.0
        for subset in _subsets(others):
            weight = factorial(len(subset)) * factorial(n - len(subset) - 1) / factorial(n)
            total += weight * (v.get(subset | {p}, 0.0) - v.get(subset, 0.0))
        values[p] = total
    return values


def banzhaf(players: Sequence[str], v: Dict[FrozenSet[str], float]) -> Dict[str, float]:
    n = len(players)
    values = {}
    for p in players:
        others = [q for q in players if q != p]
        diffs = [v.get(s | {p}, 0.0) - v.get(s, 0.0) for s in _subsets(others)]
        values[p] = sum(diffs) / (2 ** (n - 1))
    return values
