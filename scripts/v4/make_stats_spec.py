"""事前登録の最終解析（stats.py）に渡す条件定義 JSON を、各系統の state.json から組み立てる。

主要比較（Holm, docs/research_design_v4.md §5）: S 最終 vs 世代0 / S 最終 vs N 最終 / S 最終 vs A1 最終
副次（探索的）: RQ4 の基準（ベース単体・SC@k・RFT・7月チーム）との比較、世代ごとの推移

使い方:
  python3 scripts/v4/make_stats_spec.py --personas personas_selected.json \
      --state S=.../S/state.json --state N=.../N/state.json --state A1=.../A1/state.json \
      --out configs/v4/stats_final.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SC_KS = (3, 6, 9)


def team(agents, seeds, protocol="v4t", gate=True) -> dict:
    return {"type": "team", "agents": list(agents), "gen_seeds": list(seeds), "protocol": protocol, "gate": gate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas", required=True, help="personas_selected.json（order が役割の並び）")
    parser.add_argument("--state", action="append", default=[], help="系統名=state.json のパス（複数可）")
    parser.add_argument("--test-seeds", default="1,2")
    parser.add_argument("--rft-agent", default="rft.base")
    parser.add_argument("--no-c7", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    seeds = [int(s) for s in args.test_seeds.split(",")]
    order = json.loads(Path(args.personas).read_text())["order"]
    conds = {"g0": team([f"p.{r}" for r in order], seeds)}
    finals = {}
    for spec in args.state:
        name, path = spec.split("=", 1)
        state = json.loads(Path(path).read_text())
        gens = state["generations"]
        last = max(int(t) for t, g in gens.items() if g.get("steps", {}).get("test"))
        for t, g in sorted(gens.items(), key=lambda kv: int(kv[0])):
            t = int(t)
            if t == 0 or not g.get("steps", {}).get("test"):
                continue
            agents = [g["reps"][r]["agent_id"] for r in order]
            label = f"{name}_final" if t == last else f"{name}_g{t}"
            conds[label] = team(agents, seeds if t == last else [1])
        finals[name] = f"{name}_final"

    conds["base_single"] = {"type": "single", "agent": "base", "k": 9, "gen_seeds": [1]}
    for k in SC_KS:
        conds[f"base_sc{k}"] = {"type": "sc", "agent": "base", "k": k, "gen_seeds": [1]}
    conds["rft_single"] = {"type": "single", "agent": args.rft_agent, "k": 9, "gen_seeds": [1]}
    conds["rft_sc9"] = {"type": "sc", "agent": args.rft_agent, "k": 9, "gen_seeds": [1]}
    if not args.no_c7:
        conds["c7_july"] = team(["c7.critic", "c7.pragmatist", "c7.explorer"], [1], protocol="july", gate=False)

    s = finals.get("S", "S_final")
    holm = [[s, "g0"]] + [[s, finals[x]] for x in ("N", "A1") if x in finals]
    comparisons = list(holm)
    for other in ["base_single", "base_sc3", "base_sc6", "base_sc9", "rft_single", "rft_sc9", "c7_july"]:
        if other in conds:
            comparisons.append([s, other])
    for other in ["base_single", "base_sc3", "base_sc9"]:
        comparisons.append(["g0", other])
    comparisons.append(["rft_single", "base_single"])
    comparisons.append(["rft_sc9", "base_sc9"])
    for x in ("N", "A1"):
        if x in finals:
            comparisons.append([finals[x], "g0"])
    for label in conds:
        if label.startswith("S_g"):
            comparisons.append([label, "g0"])
    out = {"conditions": conds, "comparisons": comparisons, "holm": holm}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
