from __future__ import annotations

import numpy as np


def cooperation_gain(solo_acc: float, team_acc: float) -> float:
    """Approximate marginal contribution of an agent to team accuracy."""
    return max(0.0, team_acc - solo_acc)


def novelty_score(reference: np.ndarray, agent: np.ndarray) -> float:
    """Cosine distance between agent embedding and reference embedding."""
    denom = (np.linalg.norm(reference) * np.linalg.norm(agent)) + 1e-8
    cosine = float(np.dot(reference, agent) / denom)
    return 1.0 - cosine


def fitness(
    solo_acc: float,
    team_acc: float,
    novelty: float,
    w_perf: float = 0.6,
    w_coop: float = 0.3,
    w_nov: float = 0.1,
) -> float:
    coop = cooperation_gain(solo_acc, team_acc)
    return (w_perf * solo_acc) + (w_coop * coop) + (w_nov * novelty)
