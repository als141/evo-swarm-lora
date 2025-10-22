from typing import Dict, List, Tuple


def soft_vote(candidates: List[Dict[str, str]]) -> Tuple[str, Dict[str, float]]:
    """Aggregate answers by confidence-weighted majority."""
    scores: Dict[str, float] = {}
    for candidate in candidates:
        answer = candidate.get("answer", "").strip()
        confidence = float(candidate.get("confidence", 0.0))
        if not answer:
            continue
        scores[answer] = scores.get(answer, 0.0) + confidence
    if not scores:
        return "", {}
    answer = max(scores.items(), key=lambda item: item[1])[0]
    return answer, scores
