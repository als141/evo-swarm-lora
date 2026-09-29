"""【解析2b】結果 JSON（per_item）による再現性: 同一条件の別 seed / 同一 seed の別環境。

(1) 同一条件・別 seed・同一問題（主に MATH-500 L4-5 は 262 問しかなく seed 間で大きく重複）
    での最終回答の不一致率と正誤不一致率（旧環境 final_eval3 と新環境 run002）。
(2) 同一条件・同一 seed・旧環境 vs 新環境（イメージ再ビルドのみの差）: c1 / c2 / c5。
    seed が同じでも生成が変わる程度（＝ seed による再現性の実態）と、系統差（精度差）を分ける。
実行: python3 scripts/analysis/ra2b_per_item_repeatability.py
"""
import itertools
from collections import defaultdict

import numpy as np

from ra_common import OUT, dump_json, new_path, old_path, per_item_maps

BENCH = ("mmlu_pro", "math500", "supergpqa")


def series_maps(path):
    """{系列名: {item: (pred, ok)}}"""
    return {name: {k: (v["predicted"], bool(v["correct"])) for k, v in pi.items()}
            for name, pi in per_item_maps(path).items()}


def compare(ma, mb):
    common = ma.keys() & mb.keys()
    if not common:
        return None
    dis = np.mean([ma[i][0] != mb[i][0] for i in common])
    cd = np.mean([ma[i][1] != mb[i][1] for i in common])
    acc_a = np.mean([ma[i][1] for i in common])
    acc_b = np.mean([mb[i][1] for i in common])
    return {"n": len(common), "disagree": float(dis), "correct_discord": float(cd),
            "acc_a": float(acc_a), "acc_b": float(acc_b)}


def pooled(rows):
    rows = [r for r in rows if r]
    n = sum(r["n"] for r in rows)
    if not n:
        return None
    return {"n": n,
            "disagree": sum(r["disagree"] * r["n"] for r in rows) / n,
            "correct_discord": sum(r["correct_discord"] * r["n"] for r in rows) / n}


def main():
    res = {"cross_seed_old": {}, "cross_seed_new": {}, "old_vs_new_same_seed": {}}
    # (1) 旧環境: 条件ごと・系列ごとに seed ペア
    for cond in ("c1", "c2", "c3", "c3p", "c4", "c5", "c6"):
        for bench in BENCH:
            maps = {s: series_maps(old_path(cond, bench, s)) for s in (1, 2, 3)}
            rows = defaultdict(list)
            for s1, s2 in itertools.combinations((1, 2, 3), 2):
                for name in maps[s1]:
                    rows[name].append(compare(maps[s1][name], maps[s2][name]))
            for name, rs in rows.items():
                p = pooled(rs)
                if p and p["n"] >= 20:
                    res["cross_seed_old"][f"{cond}:{name}|{bench}"] = p
    # 新環境
    for cond, seeds in (("c7", range(1, 7)), ("c2", range(1, 7)), ("c1", range(1, 4)), ("c5", range(1, 4))):
        for bench in BENCH:
            maps = {s: series_maps(new_path(cond, bench, s)) for s in seeds if new_path(cond, bench, s)}
            rows = defaultdict(list)
            for s1, s2 in itertools.combinations(sorted(maps), 2):
                for name in maps[s1]:
                    rows[name].append(compare(maps[s1][name], maps[s2][name]))
            for name, rs in rows.items():
                p = pooled(rs)
                if p and p["n"] >= 20:
                    res["cross_seed_new"][f"{cond}:{name}|{bench}"] = p
    # (2) 同一 seed・旧 vs 新環境
    for cond in ("c1", "c2", "c5"):
        for bench in BENCH:
            rs = []
            for s in (1, 2, 3):
                pn = new_path(cond, bench, s)
                if not pn:
                    continue
                mo = next(iter(series_maps(old_path(cond, bench, s)).values()))
                mn = next(iter(series_maps(pn).values()))
                rs.append(compare(mo, mn))
            rs = [r for r in rs if r]
            n = sum(r["n"] for r in rs)
            res["old_vs_new_same_seed"][f"{cond}|{bench}"] = {
                "n": n,
                "disagree": sum(r["disagree"] * r["n"] for r in rs) / n,
                "correct_discord": sum(r["correct_discord"] * r["n"] for r in rs) / n,
                "acc_old": sum(r["acc_a"] * r["n"] for r in rs) / n,
                "acc_new": sum(r["acc_b"] * r["n"] for r in rs) / n,
            }
    dump_json(res, OUT / "ra2b_per_item_repeatability.json")

    for sec in ("cross_seed_old", "cross_seed_new"):
        print(f"== {sec}（同一条件・別 seed・同一問題の最終回答）")
        for k, v in res[sec].items():
            print(f"  {k:40s} n={v['n']:5d} 不一致={v['disagree']:.3f} 正誤不一致={v['correct_discord']:.3f}")
    print("== 同一 seed・旧環境 vs 新環境（イメージ再ビルドのみ）")
    for k, v in res["old_vs_new_same_seed"].items():
        print(f"  {k:18s} n={v['n']:5d} 不一致={v['disagree']:.3f} 正誤不一致={v['correct_discord']:.3f} "
              f"精度 旧{v['acc_old']:.3f} → 新{v['acc_new']:.3f} ({(v['acc_new']-v['acc_old'])*100:+.1f}pt)")


if __name__ == "__main__":
    main()
