"""H1 の追加評価（世代0と系統 S の最終世代の test を生成 seed 3, 4 で）の評価計画 JSON を作る。

事前登録の改訂（docs/research_design_v4.md §5、2026-09-30 11:40）に対応する。
  python3 scripts/v4/make_eval_plan.py --personas results/v4/personas_selected.json \
      --state results/v4/lineages/S.state.json --seeds 3,4 --out configs/v4/eval_h1_seeds34.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas", required=True)
    parser.add_argument("--state", required=True, help="系統 S の state.json")
    parser.add_argument("--seeds", default="3,4")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sel = json.loads(Path(args.personas).read_text())
    order, personas = sel["order"], sel["personas"]
    state = json.loads(Path(args.state).read_text())
    gens = state["generations"]
    last = max(int(t) for t, g in gens.items() if g.get("steps", {}).get("test"))
    seeds = [int(s) for s in args.seeds.split(",")]
    g0 = [{"agent_id": f"p.{r}", "role": r, "model": BASE_MODEL, "persona": personas[r], "adapter": None}
          for r in order]
    final = [gens[str(last)]["reps"][r] for r in order]
    plan = {"items": "data/v4/items/test.jsonl",
            "evals": [{"kind": "team", "protocol": "v4t", "gate": True, "gen_seeds": seeds, "agents": g0},
                      {"kind": "team", "protocol": "v4t", "gate": True, "gen_seeds": seeds, "agents": final}]}
    Path(args.out).write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    print(f"last generation {last}: {[a['agent_id'] for a in final]}")


if __name__ == "__main__":
    main()
