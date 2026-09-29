"""ペルソナ・エージェント社会の評価エンジン（solo / 連合の議論 / Self-Consistency）。

- round0 の出力はエージェント単位で保存し、そのエージェントを含む全ての連合で共有する。
  round1 の出力は（エージェント, 相手の集合）単位で保存する。どちらも二度は生成しない。
- 乱数 seed は（問題, 生成 seed, 役割, ラウンド）から決める。同じ役割の候補同士は同じ乱数列を使う。
- 呼び出しは依存関係付きで並列実行する。round1 は相手全員の round0 が揃った時点で投入し、
  ラウンド間で GPU を遊ばせない。
"""

from __future__ import annotations

import hashlib
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.evo4.items import Item
from src.evo4.llm import GenConfig, LLMClient
from src.evo4.prompts import r0_messages, r1_messages
from src.evo4.scoring import extract_answer, is_correct
from src.evo4.store import CallStore


@dataclass(frozen=True)
class Agent:
    agent_id: str  # 一意な個体名（例: g1.critic.self）
    role: str  # critic / pragmatist / explorer / base
    model: str  # vLLM のモデル名（ベースモデル名または LoRA 名）
    persona: str = ""


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


@dataclass
class _Task:
    key: str
    model: str
    messages_fn: Callable[[], List[dict]]
    seed: int
    meta: dict
    deps: Tuple[str, ...] = ()
    children: List["_Task"] = field(default_factory=list)
    pending: int = 0


class Society:
    def __init__(self, llm: LLMClient, store: CallStore, items: Dict[str, Item], config: GenConfig,
                 protocol: str = "v4", workers: int = 128, log: Callable[[str], None] = print):
        self.llm = llm
        self.store = store
        self.items = items
        self.config = config
        self.protocol = protocol
        self.workers = workers
        self.log = log

    # ------------------------------------------------------------------ keys
    @staticmethod
    def r0_key(agent: Agent, item_id: str, gen_seed: int) -> str:
        return f"r0|{agent.agent_id}|{item_id}|g{gen_seed}"

    def r1_key(self, agent: Agent, others: Sequence[Agent], item_id: str, gen_seed: int) -> str:
        ctx = "+".join(sorted(o.agent_id for o in others))
        return f"r1|{self.protocol}|{agent.agent_id}|{ctx}|{item_id}|g{gen_seed}"

    @staticmethod
    def sc_key(agent: Agent, item_id: str, gen_seed: int, k: int) -> str:
        return f"sc|{agent.agent_id}|{item_id}|g{gen_seed}|k{k}"

    # ------------------------------------------------------------------ task builders
    def _r0_task(self, agent: Agent, item: Item, gen_seed: int) -> _Task:
        return _Task(
            key=self.r0_key(agent, item.item_id, gen_seed),
            model=agent.model,
            messages_fn=lambda: r0_messages(agent.persona, item.question, item.answer_type),
            seed=stable_seed(item.item_id, gen_seed, agent.role, "r0"),
            meta={"kind": "r0", "agent": agent.agent_id, "role": agent.role, "model": agent.model,
                  "item": item.item_id, "gen_seed": gen_seed},
        )

    def _r1_task(self, agent: Agent, others: Sequence[Agent], item: Item, gen_seed: int) -> _Task:
        own_key = self.r0_key(agent, item.item_id, gen_seed)
        other_keys = [self.r0_key(o, item.item_id, gen_seed) for o in others]
        protocol = self.protocol

        def build() -> List[dict]:
            own = self.store.get(own_key)["text"]
            other_texts = [self.store.get(k)["text"] for k in other_keys]
            return r1_messages(protocol, agent.persona, item.question, item.answer_type, own,
                               other_texts, stable_seed(item.item_id, gen_seed, agent.role, "shuffle"))

        return _Task(
            key=self.r1_key(agent, others, item.item_id, gen_seed),
            model=agent.model,
            messages_fn=build,
            seed=stable_seed(item.item_id, gen_seed, agent.role, "r1"),
            meta={"kind": "r1", "agent": agent.agent_id, "role": agent.role, "model": agent.model,
                  "item": item.item_id, "gen_seed": gen_seed, "protocol": protocol,
                  "others": sorted(o.agent_id for o in others)},
            deps=tuple([own_key, *other_keys]),
        )

    def _sc_task(self, agent: Agent, item: Item, gen_seed: int, k: int) -> _Task:
        return _Task(
            key=self.sc_key(agent, item.item_id, gen_seed, k),
            model=agent.model,
            messages_fn=lambda: r0_messages(agent.persona, item.question, item.answer_type),
            seed=stable_seed(item.item_id, gen_seed, "sc", k),
            meta={"kind": "sc", "agent": agent.agent_id, "role": agent.role, "model": agent.model,
                  "item": item.item_id, "gen_seed": gen_seed, "k": k},
        )

    # ------------------------------------------------------------------ scheduler
    def run_tasks(self, tasks: Iterable[_Task], label: str = "") -> None:
        """依存関係付きで並列実行する。保存済みのキーは生成しない。"""
        unique: Dict[str, _Task] = {}
        for task in tasks:
            if task.key not in self.store and task.key not in unique:
                unique[task.key] = task
        if not unique:
            return
        todo = list(unique.values())
        random.Random(len(todo)).shuffle(todo)  # 長い問題と短い問題を混ぜて末尾の待ちを減らす
        for task in todo:
            waiting = [d for d in task.deps if d not in self.store]
            task.pending = len(waiting)
            for dep in waiting:
                parent = unique.get(dep)
                if parent is None:
                    raise RuntimeError(f"missing dependency {dep} for {task.key}")
                parent.children.append(task)

        lock = threading.Lock()
        done_event = threading.Event()
        state = {"remaining": len(todo), "done": 0, "tokens": 0, "loops": 0, "errors": 0}
        cancelled: set = set()
        started = time.time()
        executor = ThreadPoolExecutor(max_workers=self.workers)

        def submit(task: _Task) -> None:
            executor.submit(execute, task)

        def execute(task: _Task) -> None:
            ok = False
            try:
                result = self.llm.generate(task.model, task.messages_fn(), self.config, task.seed)
                record = {"key": task.key, **task.meta, "seed": task.seed,
                          "config": self.config.as_dict(), **result}
                self.store.put(record)
                ok = True
            except Exception as error:  # noqa: BLE001 - 失敗は記録して続行（依存先は実行しない）
                self.log(f"[society] FAILED {task.key}: {error}")
            ready: List[_Task] = []
            with lock:
                state["remaining"] -= 1
                state["done"] += 1
                if ok:
                    rec = self.store.get(task.key)
                    state["tokens"] += rec.get("n_out") or 0
                    state["loops"] += rec.get("finish") == "loop_abort"
                    for child in task.children:
                        child.pending -= 1
                        if child.pending == 0:
                            ready.append(child)
                else:
                    state["errors"] += 1
                    for key in self._descendant_keys(task):
                        if key not in cancelled:
                            cancelled.add(key)
                            state["remaining"] -= 1
                if state["done"] % 500 == 0:
                    elapsed = time.time() - started
                    self.log(f"[society] {label} done={state['done']} remaining={state['remaining']} "
                             f"out_tok/s={state['tokens'] / max(elapsed, 1e-6):.0f} "
                             f"loop_aborts={state['loops']} errors={state['errors']}")
                if state["remaining"] <= 0:
                    done_event.set()
            for child in ready:
                submit(child)

        roots = [t for t in todo if t.pending == 0]
        for task in roots:
            submit(task)
        done_event.wait()
        executor.shutdown(wait=True)
        self.store.sync()
        elapsed = time.time() - started
        self.log(f"[society] {label} finished {state['done']} calls in {elapsed:.0f}s "
                 f"({state['tokens'] / max(elapsed, 1e-6):.0f} out_tok/s, loop_aborts={state['loops']}, "
                 f"errors={state['errors']})")

    @staticmethod
    def _descendant_keys(task: _Task) -> set:
        seen: set = set()
        stack = list(task.children)
        while stack:
            child = stack.pop()
            if child.key in seen:
                continue
            seen.add(child.key)
            stack.extend(child.children)
        return seen

    # ------------------------------------------------------------------ public API
    def coalition_tasks(self, coalitions: Iterable[Sequence[Agent]], item_ids: Sequence[str],
                        gen_seed: int) -> List[_Task]:
        tasks: List[_Task] = []
        for coalition in coalitions:
            for item_id in item_ids:
                item = self.items[item_id]
                for agent in coalition:
                    tasks.append(self._r0_task(agent, item, gen_seed))
                if len(coalition) >= 2:
                    for agent in coalition:
                        others = [o for o in coalition if o.agent_id != agent.agent_id]
                        tasks.append(self._r1_task(agent, others, item, gen_seed))
        return tasks

    def sc_tasks(self, agent: Agent, item_ids: Sequence[str], gen_seed: int, k: int) -> List[_Task]:
        return [self._sc_task(agent, self.items[i], gen_seed, j) for i in item_ids for j in range(k)]

    def ensure_coalitions(self, coalitions: Iterable[Sequence[Agent]], item_ids: Sequence[str],
                          gen_seed: int, label: str = "") -> None:
        self.run_tasks(self.coalition_tasks(list(coalitions), item_ids, gen_seed), label=label)

    # ------------------------------------------------------------------ answers
    def answer_of(self, key: str, answer_type: str) -> Tuple[Optional[str], Optional[float]]:
        record = self.store.get(key)
        if record is None:
            return None, None
        return extract_answer(record["text"], answer_type), record.get("mean_logprob")

    def coalition_result(self, coalition: Sequence[Agent], item_id: str, gen_seed: int) -> dict:
        """連合の最終回答。1体は round0、2体以上は round1 の回答の多数決。
        同数は logprob 確信度の高い方、それでも同じなら役割名順で決める（乱数は使わない）。"""
        item = self.items[item_id]
        votes: List[Tuple[Optional[str], Optional[float], str]] = []
        if len(coalition) == 1:
            ans, conf = self.answer_of(self.r0_key(coalition[0], item_id, gen_seed), item.answer_type)
            votes.append((ans, conf, coalition[0].role))
        else:
            for agent in coalition:
                others = [o for o in coalition if o.agent_id != agent.agent_id]
                ans, conf = self.answer_of(self.r1_key(agent, others, item_id, gen_seed), item.answer_type)
                votes.append((ans, conf, agent.role))
        final = aggregate(votes)
        return {"answer": final, "correct": is_correct(final, item.gold, item.answer_type),
                "votes": [v[0] for v in votes]}

    def coalition_accuracy(self, coalition: Sequence[Agent], item_ids: Sequence[str], gen_seed: int) -> float:
        results = [self.coalition_result(coalition, i, gen_seed)["correct"] for i in item_ids]
        return sum(results) / len(results) if results else 0.0

    def sc_answers(self, agent: Agent, item_id: str, gen_seed: int, k: int) -> List[Tuple[Optional[str], Optional[float]]]:
        item = self.items[item_id]
        return [self.answer_of(self.sc_key(agent, item_id, gen_seed, j), item.answer_type) for j in range(k)]


def aggregate(votes: Sequence[Tuple[Optional[str], Optional[float], str]]) -> Optional[str]:
    """多数決。同数は確信度（mean logprob の最大値）→ 役割名順で決定的に決める。"""
    valid = [(a, c, r) for a, c, r in votes if a is not None]
    if not valid:
        return None
    counts = Counter(a for a, _, _ in valid)
    top = max(counts.values())
    tied = [a for a, n in counts.items() if n == top]
    if len(tied) == 1:
        return tied[0]
    best_conf: Dict[str, float] = defaultdict(lambda: float("-inf"))
    first_role: Dict[str, str] = {}
    for a, c, r in valid:
        if a in tied:
            best_conf[a] = max(best_conf[a], c if c is not None else float("-inf"))
            first_role[a] = min(first_role.get(a, r), r)
    return sorted(tied, key=lambda a: (-best_conf[a], first_role[a], a))[0]
