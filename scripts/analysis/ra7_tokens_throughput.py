"""【解析7】コスト見積の較正: 出力/入力トークン分布・打ち切りの影響・A100 実スループット。

データ: 新環境 llm_calls 全 113,750 呼び出し（ra0_extract_calls.py の索引）。
  - トークン数は Qwen3-4B 実トークナイザ（入力はチャットテンプレート適用後）
  - 打ち切り: out_tok ≥ max_tokens−2 を max_tokens 到達とみなす（finish_reason は未記録）
  - 4096/2048 打ち切りの精度影響: solo と SC は厳密（生成分布は max_tokens に依存しない）。
    チームは最終ラウンドのみ打ち切る近似（round0 を打ち切ると round1 の入力が変わるため）
  - スループット: 各ファイル＝1バッテリーエントリ＝1台の A100 上の 1 vLLM サーバ。
    呼び出しの [ts, ts+elapsed] 区間から壁時計時間・平均同時実行数・出力 tok/s を復元。
    30秒ビンで「同時実行数 → スループット」の関係も推定（クライアント並列は全ジョブ 32）。
実行: python3 scripts/analysis/ra7_tokens_throughput.py
"""
import random
from collections import Counter, defaultdict

import numpy as np

from ra_common import OUT, dump_json, load_calls_index

BENCH = ("mmlu_pro", "math500", "supergpqa")


def q(x, p):
    return float(np.percentile(x, p)) if len(x) else None


def dist(rows, key):
    x = np.array([r[key] for r in rows], float)
    return {"n": len(x), "mean": float(x.mean()), "median": q(x, 50), "p90": q(x, 90),
            "p99": q(x, 99), "max": float(x.max())}


def majority(ans, seed):
    valid = [a for a in ans if a is not None]
    if not valid:
        return None
    c = Counter(valid)
    top = max(c.values())
    w = sorted(a for a, k in c.items() if k == top)
    return w[0] if len(w) == 1 else random.Random(seed).choice(w)


def main():
    calls = [c for c in load_calls_index() if c.get("item_id")]
    res = {"token_dist": {}, "truncation_rate": {}, "trunc_accuracy": {}, "throughput": {}}

    # ---- 1) トークン分布（条件別）
    def cond_of(c):
        if c["kind"] == "team":
            return f"{c['family']}_team_r{c['round']}"
        return f"{c['family']}_{c['kind']}"
    groups = defaultdict(list)
    for c in calls:
        if c["kind"] in ("solo", "sc", "team") and c["dir"] != "g2_aggregation":
            groups[(cond_of(c), c["bench"])].append(c)
    for (cond, bench), rows in sorted(groups.items()):
        k = f"{cond}|{bench}"
        res["token_dist"][k] = {"out": dist(rows, "out_tok"), "in": dist(rows, "in_tok")}
        mt = rows[0]["max_tokens"]
        res["truncation_rate"][k] = {
            "hit_max_tokens": float(np.mean([r["out_tok"] >= mt - 2 for r in rows])),
            **{f"over_{n}": float(np.mean([r["out_tok"] > n for r in rows])) for n in (1024, 2048, 4096)}}

    # 1問あたり（条件別の呼び出し数を掛けた）平均トークン
    per_item = {}
    for bench in BENCH:
        for cond, calls_per_item in (("base_solo", 1), ("base_sc", 9)):
            rows = groups.get((cond, bench), [])
            if rows:
                per_item[f"{cond}|{bench}"] = {
                    "calls": calls_per_item,
                    "out_per_item": calls_per_item * float(np.mean([r["out_tok"] for r in rows])),
                    "in_per_item": calls_per_item * float(np.mean([r["in_tok"] for r in rows]))}
        for fam in ("c7", "c5"):
            r0 = groups.get((f"{fam}_team_r0", bench), [])
            r1 = groups.get((f"{fam}_team_r1", bench), [])
            if r0 and r1:
                per_item[f"{fam}_team|{bench}"] = {
                    "calls": 6,
                    "out_per_item": 3 * np.mean([r["out_tok"] for r in r0]) + 3 * np.mean([r["out_tok"] for r in r1]),
                    "in_per_item": 3 * np.mean([r["in_tok"] for r in r0]) + 3 * np.mean([r["in_tok"] for r in r1])}
    res["per_item_tokens"] = per_item

    # ---- 2) 打ち切りの精度影響（solo・SC は厳密、チームは最終ラウンドのみの近似）
    for bench in BENCH:
        solo = groups.get(("base_solo", bench), [])
        if solo:
            res["trunc_accuracy"][f"base_solo|{bench}"] = {
                t: float(np.mean([r[f"ok{t}"] for r in solo])) for t in ("", "4096", "2048", "1024")}
        sc = defaultdict(list)
        for r in groups.get(("base_sc", bench), []):
            sc[(r["run_seed"], r["item_id"])].append(r)
        accs = {}
        for t in ("", "4096", "2048"):
            ok = []
            for (s, _i), rows in sc.items():
                amap = {}
                for r in rows:
                    a = r[f"ans{t}"]
                    if a is not None:
                        amap[a] = amap.get(a, False) or r[f"ok{t}"]
                m = majority([r[f"ans{t}"] for r in rows], s)
                ok.append(bool(amap.get(m, False)) if m is not None else False)
            accs[t or "8192"] = float(np.mean(ok))
        res["trunc_accuracy"][f"base_sc9|{bench}"] = accs
        for fam in ("c7", "c5"):
            r1 = defaultdict(list)
            for r in groups.get((f"{fam}_team_r1", bench), []):
                r1[(r["run_seed"], r["item_id"])].append(r)
            accs = {}
            for t in ("", "4096", "2048"):
                ok = []
                for (s, _i), rows in r1.items():
                    amap = {}
                    for r in rows:
                        a = r[f"ans{t}"]
                        if a is not None:
                            amap[a] = amap.get(a, False) or r[f"ok{t}"]
                    m = majority([r[f"ans{t}"] for r in rows], s)
                    ok.append(bool(amap.get(m, False)) if m is not None else False)
                accs[t or "8192"] = float(np.mean(ok))
            res["trunc_accuracy"][f"{fam}_team_r1maj|{bench}"] = accs

    # ---- 3) スループット（ファイル＝エントリ単位）
    by_file = defaultdict(list)
    for c in calls:
        if c["kind"] in ("solo", "sc", "team", "judge"):
            by_file[(c["dir"], c["file"])].append(c)
    per_file = []
    bins_all = []
    for (d, f), rows in by_file.items():
        t0 = min(r["ts"] for r in rows)
        t1 = max(r["ts"] + r["elapsed"] for r in rows)
        wall = t1 - t0
        out = sum(r["out_tok"] for r in rows)
        inp = sum(r["in_tok"] for r in rows)
        busy = sum(r["elapsed"] for r in rows)
        kinds = Counter(cond_of(r) if r["kind"] != "judge" else "judge" for r in rows)
        # 30秒ビン: 各呼び出しの出力トークンを区間に一様配分
        nb = int(wall // 30) + 1
        tok_b = np.zeros(nb)
        conc_b = np.zeros(nb)
        for r in rows:
            s, e = r["ts"] - t0, r["ts"] - t0 + max(r["elapsed"], 1e-3)
            b0, b1 = int(s // 30), int(e // 30)
            for b in range(b0, min(b1, nb - 1) + 1):
                lo, hi = max(s, b * 30), min(e, (b + 1) * 30)
                if hi > lo:
                    frac = (hi - lo) / (e - s)
                    tok_b[b] += r["out_tok"] * frac
                    conc_b[b] += (hi - lo) / 30
        # 末尾の低並列時間（同時実行 <8 のビンの割合）
        tail_frac = float(np.mean(conc_b < 8))
        per_file.append({"dir": d, "file": f, "calls": len(rows), "wall_min": wall / 60,
                         "out_tok": out, "in_tok": inp, "out_tok_per_s": out / wall,
                         "in_tok_per_s": inp / wall, "mean_concurrency": busy / wall,
                         "per_request_tok_s": float(np.median([r["out_tok"] / max(r["elapsed"], 1e-3) for r in rows])),
                         "low_conc_time_frac": tail_frac, "kinds": dict(kinds)})
        for tb, cb in zip(tok_b, conc_b):
            bins_all.append((cb, tb / 30))
    res["throughput"]["per_file"] = sorted(per_file, key=lambda x: (x["dir"], x["file"]))
    ba = np.array(bins_all)
    conc_curve = {}
    for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 24), (24, 28), (28, 30), (30, 31), (31, 33)):
        m = (ba[:, 0] >= lo) & (ba[:, 0] < hi)
        if m.sum() > 5:
            conc_curve[f"{lo}-{hi}"] = {"bins": int(m.sum()), "median_tok_s": float(np.median(ba[m, 1])),
                                        "mean_tok_s": float(np.mean(ba[m, 1]))}
    res["throughput"]["tok_s_by_concurrency"] = conc_curve
    tot_out = sum(p["out_tok"] for p in per_file)
    tot_wall = sum(p["wall_min"] for p in per_file) * 60
    res["throughput"]["overall"] = {"files": len(per_file), "total_out_tok": tot_out,
                                    "total_wall_h": tot_wall / 3600,
                                    "overall_out_tok_s": tot_out / tot_wall,
                                    "mean_concurrency": float(np.mean([p["mean_concurrency"] for p in per_file]))}
    dump_json(res, OUT / "ra7_tokens_throughput.json")

    print("== 出力トークン分布（mean / median / p90 / p99 / max）と max_tokens 到達率・>4096率")
    for k, v in res["token_dist"].items():
        o, i = v["out"], v["in"]
        tr = res["truncation_rate"][k]
        print(f"  {k:28s} n={o['n']:6d} out {o['mean']:6.0f}/{o['median']:5.0f}/{o['p90']:5.0f}/{o['p99']:5.0f}/{o['max']:5.0f}"
              f"  到達{tr['hit_max_tokens']:.3%} >4096 {tr['over_4096']:.2%} >2048 {tr['over_2048']:.2%}"
              f"  | in mean {i['mean']:6.0f} p99 {i['p99']:6.0f}")
    print("== 1問あたりトークン")
    for k, v in per_item.items():
        print(f"  {k:26s} calls={v['calls']} out={v['out_per_item']:7.0f} in={v['in_per_item']:7.0f}")
    print("== 打ち切りの精度影響（8192 → 4096 → 2048）")
    for k, v in res["trunc_accuracy"].items():
        print(f"  {k:28s} " + "  ".join(f"{t or '8192'}:{a:.4f}" for t, a in v.items()))
    print("== スループット（ファイル=エントリ単位, A100 Spot, クライアント並列32）")
    for p in res["throughput"]["per_file"]:
        print(f"  {p['dir']:14s} {p['file'][:24]:24s} calls={p['calls']:5d} wall={p['wall_min']:6.1f}min "
              f"out={p['out_tok_per_s']:6.0f}tok/s in={p['in_tok_per_s']:6.0f}tok/s conc={p['mean_concurrency']:5.1f} "
              f"req={p['per_request_tok_s']:5.1f}tok/s 低並列時間={p['low_conc_time_frac']:.0%} {p['kinds']}")
    print("== 同時実行数ビン → 出力 tok/s（30秒ビン）")
    for k, v in conc_curve.items():
        print(f"  conc {k:6s} bins={v['bins']:5d} median {v['median_tok_s']:7.0f} mean {v['mean_tok_s']:7.0f} tok/s")
    print(f"== 全体: {res['throughput']['overall']}")


if __name__ == "__main__":
    main()
