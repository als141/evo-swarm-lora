"""【解析4b】dev セットの大きさ・問題帯域と「正しい方を選べる確率（PCS）」の実データ再標本化。

真の差が既知の系統対（新環境・同一 seed・同一問題, 問題IDクラスタ解析で有意）:
  c7 vs c5  : +2.7〜3.2pt（処方の効果, p=0.0003）
  c1 vs c5  : +2.9〜3.0pt（ペルソナSFTの能力毀損）
  c7 vs c1  : ≈0pt（同等性成立）→ 「差がない対でどれだけ誤って差を見るか」の参照
各反復で dev セット n 問を無作為抽出（問題が複数 seed に出現する場合は 1 seed を無作為に採用＝
単発評価を模擬）し、Δ̂ の符号が真の向きと一致する確率 PCS を推定する。
帯域: all / mid(p_base∈[0.1,0.9]) / mix(mid 70% + p=1 帯 30%: 能力毀損の検出力を残す)
実行: python3 scripts/analysis/ra4b_selection_reliability.py
"""
import random
from collections import defaultdict

import numpy as np

from ra_common import NEW_SEEDS, OUT, dump_json, load_calls_index, new_path, per_item_maps

BENCH = ("mmlu_pro", "math500", "supergpqa")
N_REP = 4000


def correct_map(cond, bench, seed):
    p = new_path(cond, bench, seed)
    if p is None:
        return None
    m = per_item_maps(p)
    series = m.get("team") or m.get("sc") or m.get("base")
    return {k: bool(v["correct"]) for k, v in series.items()}


def main():
    calls = [c for c in load_calls_index() if c.get("item_id") and c["kind"] == "sc"]
    sc = defaultdict(lambda: [0, 0])
    for c in calls:
        sc[c["item_id"]][0] += int(c["ok"])
        sc[c["item_id"]][1] += 1
    p_base = {i: s / k for i, (s, k) in sc.items() if k >= 9}

    rng = random.Random(20260929)
    out = {}
    for a, b in (("c7", "c5"), ("c1", "c5"), ("c7", "c1")):
        seeds = sorted(set(NEW_SEEDS[a]) & set(NEW_SEEDS[b]))
        obs = defaultdict(list)  # item → list of (a_ok - b_ok) per seed
        for bench in BENCH:
            for s in seeds:
                ma, mb = correct_map(a, bench, s), correct_map(b, bench, s)
                for i in ma.keys() & mb.keys():
                    if i in p_base:
                        obs[i].append(int(ma[i]) - int(mb[i]))
        items = list(obs)
        full_delta = np.mean([np.mean(v) for v in obs.values()])
        sign = 1 if full_delta > 0 else -1
        pools = {
            "all": items,
            "mid": [i for i in items if 0.1 <= p_base[i] <= 0.9],
            "easy": [i for i in items if p_base[i] >= 1.0],
        }
        res = {"full_delta_pt": 100 * full_delta, "n_items_all": len(items),
               "n_items_mid": len(pools["mid"]), "pcs": {}}
        for design in ("all", "mid", "mix"):
            for n in (50, 100, 200, 300, 500, 1000):
                if design != "mix" and n > len(pools[design]):
                    continue
                if design == "mix" and (int(n * 0.7) > len(pools["mid"]) or n - int(n * 0.7) > len(pools["easy"])):
                    continue
                correct = ties = 0
                deltas = []
                for _ in range(N_REP):
                    if design == "mix":
                        chosen = rng.sample(pools["mid"], int(n * 0.7)) + rng.sample(pools["easy"], n - int(n * 0.7))
                    else:
                        chosen = rng.sample(pools[design], n)
                    d = np.mean([rng.choice(obs[i]) for i in chosen])
                    deltas.append(d)
                    if d * sign > 0:
                        correct += 1
                    elif d == 0:
                        ties += 1
                res["pcs"][f"{design}_n{n}"] = {
                    "pcs": (correct + 0.5 * ties) / N_REP,
                    "mean_delta_pt": 100 * float(np.mean(deltas)),
                    "sd_delta_pt": 100 * float(np.std(deltas)),
                }
        out[f"{a}_vs_{b}"] = res
    dump_json(out, OUT / "ra4b_selection_reliability.json")
    for k, v in out.items():
        print(f"== {k}: 全体Δ={v['full_delta_pt']:+.2f}pt  問題数 all={v['n_items_all']} mid={v['n_items_mid']}")
        for d, x in v["pcs"].items():
            print(f"    {d:10s} PCS={x['pcs']:.3f}  E[Δ̂]={x['mean_delta_pt']:+.2f}pt  SD(Δ̂)={x['sd_delta_pt']:.2f}pt")


if __name__ == "__main__":
    main()
