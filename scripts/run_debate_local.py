import argparse
import json
import re
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.personalities import PERSONAS
from src.models.qwen_loader import attach_lora, load_base, generate
from src.utils.prompts import DEBATE_RULES
from src.coop.voting import soft_vote


def build_messages(persona_prompt: str, topic: str, transcript: List[str]) -> List[dict]:
    messages = [
        {"role": "system", "content": persona_prompt + "\n" + DEBATE_RULES},
    ]
    if not transcript:
        messages.append({"role": "user", "content": topic})
    else:
        history = "\n".join(transcript)
        messages.append({"role": "user", "content": f"これまでの議論:\n{history}"})
    return messages


JSON_BLOCK_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)
CONFIDENCE_PATTERN = re.compile(r"([0-1](?:\.\d+)?)")


def extract_vote(agent_idx: int, utterance: str) -> dict:
    answer = ""
    confidence = 0.6

    json_match = JSON_BLOCK_PATTERN.search(utterance)
    if json_match:
        try:
            payload = json.loads(json_match.group())
            answer = payload.get("answer") or payload.get("vote") or payload.get("conclusion") or ""
            confidence = float(payload.get("confidence", confidence))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    if not answer:
        lines = [line.strip() for line in utterance.splitlines() if line.strip()]
        keyword_line = next(
            (line for line in lines if any(keyword in line for keyword in ("Answer", "結論", "Vote", "最終"))),
            lines[0] if lines else "",
        )
        if ":" in keyword_line:
            answer = keyword_line.split(":", 1)[-1].strip()
        else:
            answer = keyword_line.strip()
        answer = answer.split()[0] if answer else ""

    if isinstance(confidence, str) and confidence:
        try:
            confidence = float(confidence)
        except ValueError:
            confidence = 0.6

    if not isinstance(confidence, (int, float)):
        confidence_match = CONFIDENCE_PATTERN.search(utterance)
        if confidence_match:
            confidence = float(confidence_match.group(1))

    confidence = max(0.0, min(1.0, confidence))
    return {"agent": f"persona_{agent_idx}", "answer": answer or "", "confidence": confidence}


def main():
    parser = argparse.ArgumentParser(description="Run a local multi-agent debate using persona LoRA adapters.")
    parser.add_argument("--topic", required=True, help="Debate topic or question.")
    parser.add_argument(
        "--adapters",
        nargs=3,
        default=["adapters/persona_a", "adapters/persona_b", "adapters/persona_c"],
        help="Adapter paths for the three persona agents.",
    )
    parser.add_argument("--rounds", type=int, default=3, help="Number of debate rounds.")
    args = parser.parse_args()

    if args.rounds <= 0:
        print("[info] Rounds set to 0; skipping debate execution.")
        return

    base_model, tokenizer = load_base()
    agents = []
    for adapter_path in args.adapters:
        path = Path(adapter_path)
        if path.exists():
            agents.append(attach_lora(base_model, adapter_path))
        else:
            print(f"[warn] Adapter not found at {adapter_path}; using base model weights.")
            agents.append(base_model)
    transcript: List[str] = []
    votes: List[dict] = []

    for round_idx in range(args.rounds):
        for agent_idx, agent in enumerate(agents):
            persona_desc = list(PERSONAS.values())[agent_idx]
            messages = build_messages(persona_desc, args.topic, transcript)
            utterance = generate(agent, tokenizer, messages, max_new_tokens=256)
            tag = f"[A{agent_idx}][R{round_idx}]"
            transcript.append(f"{tag} {utterance.strip()}")
            votes.append(extract_vote(agent_idx, utterance))

    final_answer, scores = soft_vote(votes)

    print("=== Debate Transcript ===")
    for line in transcript:
        print(line)
    print("=== Vote Scores ===")
    print(scores)
    print("Final Answer:", final_answer)


if __name__ == "__main__":
    main()
