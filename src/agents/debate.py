from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.agents.personalities import PERSONAS
from src.coop.voting import soft_vote
from src.models.qwen_loader import attach_lora, generate
from src.utils.prompts import DEBATE_RULES


@dataclass
class DebateAgent:
    name: str
    adapter_path: str
    persona_prompt: str


@dataclass
class DebateTurn:
    agent: str
    content: str
    round_index: int


@dataclass
class DebateSession:
    base_model: any
    tokenizer: any
    agents: List[DebateAgent]
    transcript: List[DebateTurn] = field(default_factory=list)

    def _build_messages(self, agent: DebateAgent) -> List[dict]:
        system_message = agent.persona_prompt + "\n" + DEBATE_RULES
        messages = [{"role": "system", "content": system_message}]
        if not self.transcript:
            return messages
        history = "\n".join(f"[{turn.agent}] {turn.content}" for turn in self.transcript)
        messages.append({"role": "user", "content": f"これまでの議論:\n{history}"})
        return messages

    def run_round(self, topic: str, round_index: int, max_new_tokens: int = 256) -> Dict[str, float]:
        votes = []
        for agent in self.agents:
            model = attach_lora(self.base_model, agent.adapter_path)
            if not self.transcript:
                messages = [{"role": "system", "content": agent.persona_prompt + "\n" + DEBATE_RULES}, {"role": "user", "content": topic}]
            else:
                messages = self._build_messages(agent)
            reply = generate(model, self.tokenizer, messages, max_new_tokens=max_new_tokens)
            self.transcript.append(DebateTurn(agent=agent.name, content=reply.strip(), round_index=round_index))
            votes.append({"agent": agent.name, "answer": reply.strip()[:1], "confidence": 0.6})
        final_answer, scores = soft_vote(votes)
        return {"answer": final_answer, "scores": scores}


def build_default_agents() -> List[DebateAgent]:
    names = list(PERSONAS.keys())
    return [
        DebateAgent(name=names[idx], adapter_path=f"adapters/{names[idx]}", persona_prompt=PERSONAS[names[idx]])
        for idx in range(len(names))
    ]
