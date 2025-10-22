import argparse
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


def extract_vote(agent_idx: int, utterance: str) -> dict:
    answer = utterance.strip().splitlines()[0][:1]
    try:
        confidence = float(next((token.split(":")[-1] for token in utterance.splitlines() if "自信" in token), "0.6"))
    except ValueError:
        confidence = 0.6
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
