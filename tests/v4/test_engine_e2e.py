"""evo4 エンジンの結合試験（偽 vLLM サーバ相手）。GPU 不要。"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evo4.items import Item  # noqa: E402
from src.evo4.llm import GenConfig, LLMClient, find_loop  # noqa: E402
from src.evo4.society import Agent, Society, aggregate  # noqa: E402
from src.evo4.store import CallStore  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    assert find_loop("abc " * 10) is None
    assert find_loop("x" * 5000) is not None
    assert find_loop(("The same line repeats forever. " * 200)) is not None
    normal = " ".join(f"step {i}: compute value {i * 7 % 13} and check." for i in range(400))
    assert find_loop(normal) is None, "false positive on normal text"
    assert aggregate([("A", -0.5, "critic"), ("B", -0.1, "explorer"), ("A", -0.9, "pragmatist")]) == "A"
    assert aggregate([("A", -0.5, "critic"), ("B", -0.1, "explorer")]) == "B"
    assert aggregate([(None, None, "critic"), ("C", None, "explorer")]) == "C"

    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tests.v4.fake_vllm:app", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT,
    )
    try:
        for _ in range(50):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.2)
        llm = LLMClient(f"http://127.0.0.1:{port}/v1")
        items = {
            f"q{i}": Item(item_id=f"q{i}", bench="mmlu_pro", question=f"Question {i}\nA. x\nB. y", gold="A",
                          answer_type="letter")
            for i in range(20)
        }
        items["qloop"] = Item(item_id="qloop", bench="mmlu_pro", question="loop please", gold="A",
                              answer_type="letter")
        with tempfile.TemporaryDirectory() as tmp:
            store = CallStore(os.path.join(tmp, "local"), os.path.join(tmp, "remote"), sync_every=0)
            soc = Society(llm, store, items, GenConfig(max_tokens=512), protocol="v4", workers=16)
            a = Agent("g0.critic", "critic", "m-critic", "critic persona")
            b = Agent("g0.pragmatist", "pragmatist", "m-prag", "pragmatist persona")
            c = Agent("g0.explorer", "explorer", "m-expl", "explorer persona")
            ids = sorted(items)
            coalitions = [[a], [b], [c], [a, b], [a, c], [b, c], [a, b, c]]
            soc.ensure_coalitions(coalitions, ids, gen_seed=1, label="test")
            n_calls = len(store)
            # r0: 3 agents x 21 items = 63 ; r1: pairs 3 x 2 x 21 = 126 ; triple 3 x 21 = 63
            assert n_calls == 63 + 126 + 63, n_calls
            loop_rec = store.get(soc.r0_key(a, "qloop", 1))
            assert loop_rec["finish"] == "loop_abort", loop_rec["finish"]
            acc = soc.coalition_accuracy([a, b, c], ids, 1)
            assert 0.0 <= acc <= 1.0
            # 再実行しても新規生成は起きない（キャッシュ）
            soc.ensure_coalitions(coalitions, ids, gen_seed=1, label="again")
            assert len(store) == n_calls
            # r1 プロンプトに自分の round0 解答が assistant ターンとして入っている
            from src.evo4.prompts import r1_messages
            msgs = r1_messages("v4", "p", "Q", "letter", "OWN", ["O1", "O2"], 3)
            assert msgs[2] == {"role": "assistant", "content": "OWN"}
            # 別プロセス相当: ストアを読み直すと全件復元される
            store2 = CallStore(os.path.join(tmp, "local2"), os.path.join(tmp, "remote"))
            assert len(store2) == n_calls, (len(store2), n_calls)
            # ゲーティング: 全員一致なら round1 を生成しない
            soc_g = Society(llm, store, items, GenConfig(max_tokens=512), protocol="v4t", workers=16, gate=True)
            s1 = Agent("s.critic", "critic", "same-model", "p1")
            s2 = Agent("s.pragmatist", "pragmatist", "same-model", "p2")
            s3 = Agent("s.explorer", "explorer", "same-model", "p3")
            before = len(store)
            soc_g.ensure_coalitions([[s1, s2, s3]], [i for i in ids if i != "qloop"], gen_seed=1, label="gated")
            added = len(store) - before
            assert added == 60, added  # round0 の 3体×20問だけ。全員一致なので round1 は生成されない
            assert not any(k.startswith("r1|v4t|s.") for k in (r["key"] for r in store.records()))
            res = soc_g.coalition_result([s1, s2, s3], "q0", 1)
            assert res.get("gated") is True
            print("E2E OK:", n_calls, "calls; loop abort detected; accuracy", round(acc, 3))
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
