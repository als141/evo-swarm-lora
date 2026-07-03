"""3エージェント編成における厳密 Shapley 値による協調寄与の計算。

エージェント数が 3 のため、全 7 つの非空連合の性能 v(S) を実測すれば
近似（LOO 等）ではなく厳密な Shapley 値が計算できる。
v(∅) は 0（または基準精度）とし、characteristic function は
「その連合が debate プロトコルで達成した精度」。
単独連合 v({i}) は solo 精度に一致する。
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Dict, FrozenSet, List, Tuple


def all_coalitions(agent_names: List[str]) -> List[Tuple[str, ...]]:
    coalitions: List[Tuple[str, ...]] = []
    for size in range(1, len(agent_names) + 1):
        coalitions.extend(combinations(agent_names, size))
    return coalitions


def shapley_values(
    agent_names: List[str],
    coalition_values: Dict[FrozenSet[str], float],
    empty_value: float = 0.0,
) -> Dict[str, float]:
    """coalition_values: frozenset(連合) -> v(S)。全ての非空連合が必要。"""
    n = len(agent_names)
    values: Dict[str, float] = {}

    def v(subset: FrozenSet[str]) -> float:
        if not subset:
            return empty_value
        if subset not in coalition_values:
            raise KeyError(f"Missing coalition value for {sorted(subset)}")
        return coalition_values[subset]

    for agent in agent_names:
        others = [a for a in agent_names if a != agent]
        total = 0.0
        for size in range(len(others) + 1):
            for combo in combinations(others, size):
                subset = frozenset(combo)
                weight = factorial(size) * factorial(n - size - 1) / factorial(n)
                total += weight * (v(subset | {agent}) - v(subset))
        values[agent] = total
    return values


def cooperation_gain(
    agent_name: str,
    shapley: Dict[str, float],
    solo_accuracy: Dict[str, float],
) -> float:
    """協調寄与 = Shapley 値 −solo 精度の平均寄与分。

    Shapley 値はチーム性能への総寄与を表すため、個体性能そのものと
    協調による上乗せを分離する場合は solo 精度を差し引いて解釈する。
    """
    return shapley[agent_name] - solo_accuracy[agent_name] / max(len(solo_accuracy), 1)
