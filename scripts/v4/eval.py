"""v4 の評価ドライバ（生成のみ。採点・統計はオフライン）。1 ジョブで複数の評価をまとめて実行する。

評価計画（JSON）の例:
{
  "items": "data/v4/items/test.jsonl",
  "evals": [
    {"kind": "sc", "agent": {"agent_id": "base", "role": "base", "model": "Qwen/Qwen3-4B-Instruct-2507",
                              "persona": "", "adapter": null}, "k": 9, "gen_seeds": [1, 2]},
    {"kind": "team", "agents": [...3 agent dict...], "gen_seeds": [1, 2], "protocol": "v4"}
  ]
}
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "vendor"))

from src.evo4.items import index_items, load_items  # noqa: E402
from src.evo4.llm import GenConfig, LLMClient  # noqa: E402
from src.evo4.server import VllmServer  # noqa: E402
from src.evo4.society import Agent, Society  # noqa: E402
from src.evo4.store import CallStore  # noqa: E402


def to_agent(d: dict) -> Agent:
    return Agent(d["agent_id"], d["role"], d["model"], d.get("persona", ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--workers", type=int, default=192)
    parser.add_argument("--max-loras", type=int, default=12)
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    item_path = plan["items"] if Path(plan["items"]).is_absolute() else str(ROOT / plan["items"])
    items = index_items([item_path])
    ids = [i.item_id for i in load_items(item_path)]
    server = VllmServer(max_loras=args.max_loras, max_lora_rank=16)
    server.start()
    llm = LLMClient(server.base_url)
    store = CallStore(tempfile.mkdtemp(prefix="evo4_store_"), args.store, sync_every=60)
    config = GenConfig()
    for idx, spec in enumerate(plan["evals"]):
        started = time.time()
        agents = [spec["agent"]] if spec["kind"] == "sc" else spec["agents"]
        for d in agents:
            if d.get("adapter"):
                server.load_lora(d["model"], d["adapter"])
        society = Society(llm, store, items, config, protocol=spec.get("protocol", "v4"), workers=args.workers)
        for gen_seed in spec["gen_seeds"]:
            if spec["kind"] == "sc":
                society.run_tasks(society.sc_tasks(to_agent(spec["agent"]), ids, gen_seed, spec["k"]),
                                  label=f"eval{idx}_sc_g{gen_seed}")
            elif spec["kind"] == "team":
                society.ensure_coalitions([[to_agent(d) for d in spec["agents"]]], ids, gen_seed,
                                          label=f"eval{idx}_team_g{gen_seed}")
            elif spec["kind"] == "solo":
                society.ensure_coalitions([[to_agent(d)] for d in spec["agents"]], ids, gen_seed,
                                          label=f"eval{idx}_solo_g{gen_seed}")
            else:
                raise ValueError(spec["kind"])
        print(f"[eval] spec {idx} ({spec['kind']}) done in {time.time() - started:.0f}s", flush=True)
    store.sync()
    server.stop()


if __name__ == "__main__":
    main()
