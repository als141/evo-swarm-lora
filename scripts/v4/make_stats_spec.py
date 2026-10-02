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

SC_KS = tuple(range(1, 10))


def team(agents, seeds, protocol="v4t", gate=True) -> dict:
    return {"type": "team", "agents": list(agents), "gen_seeds": list(seeds), "protocol": protocol, "gate": gate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas", required=True, help="personas_selected.json（order が役割の並び）")
    parser.add_argument("--state", action="append", default=[], help="系統名=state.json のパス（複数可）")
    parser.add_argument("--test-seeds", default="1,2", help="H2・H3 と系統の最終世代に使う生成 seed")
    parser.add_argument("--h1-seeds", default="1,2,3,4",
                        help="H1（S 最終 vs 世代0）に使う生成 seed（2026-09-30 の事前登録の改訂）")
    parser.add_argument("--rft-agent", default="rft.base")
    parser.add_argument("--no-c7", action="store_true")
    parser.add_argument("--match-k", default="",
                        help="計算量を揃えた SC の k（生成回数一致,出力トークン一致。事前登録 §5 の副次）")
    parser.add_argument("--extra-k", default="",
                        help="事前登録にない計算量の揃え方の SC の k（入出力の合計トークン一致など。探索）")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    seeds = [int(s) for s in args.test_seeds.split(",")]
    h1_seeds = [int(s) for s in args.h1_seeds.split(",")]
    order = json.loads(Path(args.personas).read_text())["order"]
    conds = {"g0": team([f"p.{r}" for r in order], h1_seeds),
             "g0_s1": team([f"p.{r}" for r in order], [1]),  # 世代の推移は seed 1 どうしで比べる
             "g0_s12": team([f"p.{r}" for r in order], seeds)}  # 系統 N・A1 の最終世代と seed をそろえる
    finals = {}
    for spec in args.state:
        name, path = spec.split("=", 1)
        state = json.loads(Path(path).read_text())
        gens = state["generations"]
        last = max(int(t) for t, g in gens.items() if g.get("steps", {}).get("test"))
        prev_agents = None
        for t, g in sorted(gens.items(), key=lambda kv: int(kv[0])):
            t = int(t)
            if t == 0 or not g.get("steps", {}).get("test"):
                continue
            agents = [g["reps"][r]["agent_id"] for r in order]
            same_as_prev = agents == prev_agents  # 親が据え置かれて前世代と同じチームなら推移の行を重複させない
            prev_agents = agents
            if same_as_prev and t != last:
                continue
            label = f"{name}_final" if t == last else f"{name}_g{t}"
            if t == last and name == "S":
                conds["S_final"] = team(agents, h1_seeds)        # H1 と RQ4 の比較
                conds["S_final_s12"] = team(agents, seeds)       # H2・H3（N・A1 と seed をそろえる）
            else:
                conds[label] = team(agents, seeds if t == last else [1])
        finals[name] = f"{name}_final"

    # 議論なし（構成員の round0 の多数決）: 議論の上積みを分ける（事前登録 §5 の副次）
    conds["g0_r0vote"] = {"type": "team_r0vote", "agents": conds["g0"]["agents"], "gen_seeds": h1_seeds}
    if "S_final" in conds:
        conds["S_final_r0vote"] = {"type": "team_r0vote", "agents": conds["S_final"]["agents"], "gen_seeds": h1_seeds}
    conds["base_single"] = {"type": "single", "agent": "base", "k": 9, "gen_seeds": [1]}
    for k in SC_KS:
        conds[f"base_sc{k}"] = {"type": "sc", "agent": "base", "k": k, "gen_seeds": [1]}
    conds["rft_single"] = {"type": "single", "agent": args.rft_agent, "k": 9, "gen_seeds": [1]}
    conds["rft_sc9"] = {"type": "sc", "agent": args.rft_agent, "k": 9, "gen_seeds": [1]}
    if not args.no_c7:
        conds["c7_july"] = team(["c7.critic", "c7.pragmatist", "c7.explorer"], [1], protocol="july", gate=False)

    s = finals.get("S", "S_final")
    # 区分: 主要＝H1〜H3（事前登録）、副次＝事前登録の副次（RQ4 の位置づけ・世代推移・議論の上積み）、探索＝事前登録にない比較
    holm = [[s, "g0", "主要"]] + [["S_final_s12", finals[x], "主要"] for x in ("N", "A1") if x in finals]
    comparisons = list(holm)
    match = [f"base_sc{int(k)}" for k in args.match_k.split(",") if k.strip()]
    # SC@k の定義への頑健性: 計算量を揃えた k を「9本から k 本の全組合せの平均」でも比べる（探索）
    match_exp = []
    for k in args.match_k.split(","):
        if k.strip():
            conds[f"base_sc{int(k)}_exp"] = {"type": "sc_expect", "agent": "base", "k": int(k), "n": 9, "gen_seeds": [1]}
            match_exp.append(f"base_sc{int(k)}_exp")
    extra = []
    for k in args.extra_k.split(","):
        if k.strip():
            conds[f"base_sc{int(k)}_exp"] = {"type": "sc_expect", "agent": "base", "k": int(k), "n": 9, "gen_seeds": [1]}
            extra += [f"base_sc{int(k)}", f"base_sc{int(k)}_exp"]
    for other in ["base_single", "base_sc3", "base_sc6", "base_sc9", *match, "rft_single", "rft_sc9", "c7_july"]:
        if other in conds:
            comparisons.append([s, other, "副次"])
    for other in [*match_exp, *extra]:
        comparisons.append([s, other, "探索"])
    if "S_final_r0vote" in conds:
        comparisons.append([s, "S_final_r0vote", "副次"])
    for label in conds:
        if label.startswith("S_g"):
            comparisons.append([label, "g0_s1", "副次"])  # 途中世代は seed 1 だけなので世代0も seed 1 で比べる
    for other in ["base_single", "base_sc3", "base_sc6", "base_sc9", *match, *match_exp, *extra, "rft_sc9", "c7_july",
                  "g0_r0vote"]:
        if other in conds:
            comparisons.append(["g0", other, "探索"])
    for x in ("N", "A1"):
        if x in finals:
            comparisons.append([finals[x], "g0_s12", "探索"])  # seed 1,2 どうしで比べる
    comparisons.append(["rft_single", "base_single", "探索"])
    comparisons.append(["rft_sc9", "base_sc9", "探索"])
    if "c7_july" in conds:
        comparisons.append(["c7_july", "base_sc9", "探索"])
    out = {"conditions": conds, "comparisons": comparisons, "holm": holm}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
