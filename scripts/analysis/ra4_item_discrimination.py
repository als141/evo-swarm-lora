"""【解析4】問題の識別力・必要問題数（検出力）・dev/test 問題IDプール。

1) 問題ごとの難易度プロファイル（新環境）: ベースモデルの正答確率 p_base を SC サンプル
   （9サンプル×出現 seed 数）から推定し、帯域別の分布を出す。
2) 条件間の差（c7 vs c2 / c7 vs c1 / c7 vs c5）がどの難易度帯から生じているか。
3) 検出力: 2 系統の精度差の標準誤差 SE = sqrt([Var_i(δ_i) + (σ²_A+σ²_B)/r] / n)。
   σ² = E_i[p_i(1−p_i)]（同一問題の再サンプリング分散。解析2で seed による
   common random numbers は効かないと判明したため共分散項は 0 とする）。
   単発比較の実測不一致率 d からの SE ≈ sqrt(d/n) も併記。
4) 問題IDプール:
   - used: これまでに一度でも評価・選抜に使った全問題（汚染管理用）
   - dev_pool: 新環境で難易度が既知（SC ≥9 サンプル）の既使用問題。帯域ラベル付き
   - test_pool: 一度も使っていない新規問題（固定 seed で層化抽出）。MATH は MATH-500 の
     L4-5（262問）が全て既使用のため、MATH test 全体（5,000問）の L4-5 から
     MATH-500 収録分を除いた問題を使う
実行: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python3 scripts/analysis/ra4_item_discrimination.py
"""
import json
import math
import random
from collections import defaultdict

import numpy as np
from scipy.stats import norm

from ra_common import (NEW_SEEDS, OUT, ROOT, all_result_files, bench_of_item,
                       dump_json, import_tasks_without_datasets, item_meta, load_calls_index,
                       new_path, per_item_maps)

BENCH = ("mmlu_pro", "math500", "supergpqa")
BANDS = [(0.0, 0.0, "p=0"), (0.0, 0.2, "(0,0.2]"), (0.2, 0.4, "(0.2,0.4]"), (0.4, 0.6, "(0.4,0.6]"),
         (0.6, 0.8, "(0.6,0.8]"), (0.8, 1.0, "(0.8,1)"), (1.0, 1.0, "p=1")]
Z = norm.ppf(0.975) + norm.ppf(0.80)


def band_of(p):
    if p <= 0.0:
        return "p=0"
    if p >= 1.0:
        return "p=1"
    for lo, hi, name in BANDS[1:-1]:
        if lo < p <= hi:
            return name
    return "(0.8,1)"


def unbiased_pq(successes, k):
    """p(1−p) の不偏推定（k 回の独立試行）。"""
    if k < 2:
        return None
    p = successes / k
    return p * (1 - p) * k / (k - 1)


def correct_map(cond, bench, seed):
    p = new_path(cond, bench, seed)
    if p is None:
        return None
    m = per_item_maps(p)
    series = m.get("team") or m.get("sc") or m.get("base")
    return {k: bool(v["correct"]) for k, v in series.items()}


def boxed_answer(solution):
    idx = solution.rfind("\\boxed")
    if idx < 0:
        return None
    i = solution.find("{", idx)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(solution):
        if solution[j] == "{":
            depth += 1
        elif solution[j] == "}":
            depth -= 1
            if depth == 0:
                return solution[i + 1:j]
        j += 1
    return None


def main():
    tasks = import_tasks_without_datasets()
    meta = item_meta()
    calls = [c for c in load_calls_index() if c.get("item_id")]

    # ---- 1) p_base（SC サンプル）と c7 r0 の正答率
    sc_s = defaultdict(lambda: [0, 0])
    sc_run = defaultdict(lambda: [0, 0])      # (item, seed) → [succ, k]（seed 内 9 サンプル）
    for c in calls:
        if c["kind"] == "sc":
            sc_s[c["item_id"]][0] += int(c["ok"])
            sc_s[c["item_id"]][1] += 1
            sc_run[(c["item_id"], c["run_seed"])][0] += int(c["ok"])
            sc_run[(c["item_id"], c["run_seed"])][1] += 1
    c7r0 = defaultdict(lambda: [0, 0])
    for c in calls:
        if c["kind"] == "team" and c["family"] == "c7" and c["round"] == 0 and c["dir"] != "g2_aggregation":
            c7r0[c["item_id"]][0] += int(c["ok"])
            c7r0[c["item_id"]][1] += 1
    p_base = {i: s / k for i, (s, k) in sc_s.items() if k >= 9}

    dist = {}
    for bench in BENCH:
        ps = [p for i, p in p_base.items() if bench_of_item(i) == bench]
        counts = defaultdict(int)
        for p in ps:
            counts[band_of(p)] += 1
        dist[bench] = {"n_items": len(ps), "mean_p": float(np.mean(ps)),
                       "bands": {b[2]: counts[b[2]] / len(ps) for b in BANDS},
                       "mid_0.1_0.9_share": float(np.mean([(0.1 <= p <= 0.9) for p in ps]))}

    # ---- 2) 条件間差の難易度帯別寄与（同一 seed・同一問題の対）
    contrib = {}
    for a, b in (("c7", "c2"), ("c7", "c1"), ("c7", "c5"), ("c1", "c2")):
        per_band = defaultdict(lambda: [0.0, 0])
        total_n = 0
        seeds = sorted(set(NEW_SEEDS[a]) & set(NEW_SEEDS[b]))
        disc = defaultdict(list)
        for bench in BENCH:
            for s in seeds:
                ma, mb = correct_map(a, bench, s), correct_map(b, bench, s)
                if ma is None or mb is None:
                    continue
                for i in ma.keys() & mb.keys():
                    if i not in p_base:
                        continue
                    d = int(ma[i]) - int(mb[i])
                    band = band_of(p_base[i])
                    per_band[band][0] += d
                    per_band[band][1] += 1
                    total_n += 1
                    disc[bench].append(ma[i] != mb[i])
        total_d = sum(v[0] for v in per_band.values())
        contrib[f"{a}_vs_{b}"] = {
            "seeds": seeds, "n_obs": total_n, "delta_pt": 100 * total_d / total_n,
            "by_band": {bn: {"share_obs": v[1] / total_n, "delta_in_band_pt": 100 * v[0] / v[1] if v[1] else None,
                             "share_of_total_delta": (v[0] / total_d) if total_d else None}
                        for bn, v in sorted(per_band.items())},
            "discordance_rate": {bn: float(np.mean(v)) for bn, v in disc.items()},
        }

    # ---- 3) 同一問題の再サンプリング分散 σ² = E[p(1−p)]
    sigma2 = {}
    # 単一サンプル（ベース）: seed 内 9 サンプルから不偏推定
    for bench in BENCH:
        v = [unbiased_pq(s, k) for (i, _sd), (s, k) in sc_run.items() if bench_of_item(i) == bench and k == 9]
        sigma2[f"base_single_{bench}"] = float(np.mean(v))
    # c7 r0（エージェント単発）: 同一問題の全 r0 から（エージェント差は解析2で無視できる規模）
    for bench in BENCH:
        v = [unbiased_pq(s, k) for i, (s, k) in c7r0.items() if bench_of_item(i) == bench and k >= 3]
        sigma2[f"c7_agent_single_{bench}"] = float(np.mean(v))
    # チーム最終・SC@9 多数決: 複数 seed に出現した問題の seed 間分散（主に MATH）
    for cond in ("c7", "c2", "c1", "c5"):
        for bench in BENCH:
            occ = defaultdict(list)
            for s in NEW_SEEDS[cond]:
                m = correct_map(cond, bench, s)
                if m:
                    for i, ok in m.items():
                        occ[i].append(int(ok))
            v = [unbiased_pq(sum(x), len(x)) for x in occ.values() if len(x) >= 2]
            if len(v) >= 30:
                sigma2[f"{cond}_final_{bench}"] = float(np.mean(v))
                sigma2[f"{cond}_final_{bench}_n_items"] = len(v)

    # 必要問題数: 真の差 δ を 80% 検出力で検出（両側 5%）。同質系統（Var(δ_i)≈0）の近似
    def n_needed(delta, s2a, s2b, r=1, var_delta=0.0):
        return math.ceil(Z**2 * (var_delta + (s2a + s2b) / r) / delta**2)

    power = {}
    for bench in BENCH:
        s2_team = sigma2.get(f"c7_final_{bench}", sigma2[f"c7_agent_single_{bench}"])
        s2_single = sigma2[f"c7_agent_single_{bench}"]
        power[bench] = {
            f"team_vs_team_delta{d}pt_r{r}": n_needed(d / 100, s2_team, s2_team, r)
            for d in (1, 2, 3, 5) for r in (1, 3)
        } | {
            f"single_vs_single_delta{d}pt_r{r}": n_needed(d / 100, s2_single, s2_single, r)
            for d in (1, 2, 3, 5) for r in (1, 3)
        }
    # 実測の不一致率から: SE = sqrt(d/n)
    se_table = {}
    for key, v in contrib.items():
        for bench, d in v["discordance_rate"].items():
            se_table[f"{key}_{bench}"] = {
                "discordance": d, **{f"SE_pt_n{n}": 100 * math.sqrt(d / n) for n in (100, 300, 600, 1000)}}

    # ---- 3b) 帯域を絞った dev セットの効率（c7 vs c1, c7 vs c2 の実データで z²/観測 を比較）
    eff = {}
    for a, b in (("c7", "c1"), ("c7", "c2"), ("c7", "c5")):
        seeds = sorted(set(NEW_SEEDS[a]) & set(NEW_SEEDS[b]))
        for lo, hi, name in ((0.0, 1.0, "all"), (0.1, 0.9, "mid_0.1_0.9"), (0.2, 0.8, "mid_0.2_0.8")):
            d = []
            for bench in BENCH:
                for s in seeds:
                    ma, mb = correct_map(a, bench, s), correct_map(b, bench, s)
                    for i in ma.keys() & mb.keys():
                        if i in p_base and lo <= p_base[i] <= hi:
                            d.append(int(ma[i]) - int(mb[i]))
            d = np.array(d, float)
            z2_per_obs = (d.mean() ** 2) / d.var() if d.var() > 0 else float("nan")
            eff[f"{a}_vs_{b}_{name}"] = {"n_obs": len(d), "delta_pt": 100 * d.mean(),
                                         "z2_per_obs": z2_per_obs,
                                         "obs_for_z2.8": (Z**2) / z2_per_obs if z2_per_obs else None}

    # ---- 3c) 計算量あたりの情報量: 帯域別の「不一致（=検定に効く観測）」の出現率
    # 対応ありの符号検定では一致した問題は統計量に寄与しない。したがって常に正解/常に不正解の
    # 問題は「計算を払っても情報を生まない」。c2 は p_base 推定に自分のサンプルを使うため除外。
    info = {}
    for a, b in (("c7", "c5"), ("c7", "c1"), ("c5", "c1")):
        seeds = sorted(set(NEW_SEEDS[a]) & set(NEW_SEEDS[b]))
        n_band, dis_band, net_band = defaultdict(int), defaultdict(int), defaultdict(int)
        for bench in BENCH:
            for s in seeds:
                ma, mb = correct_map(a, bench, s), correct_map(b, bench, s)
                for i in ma.keys() & mb.keys():
                    if i not in p_base:
                        continue
                    bn = band_of(p_base[i])
                    n_band[bn] += 1
                    dis_band[bn] += int(ma[i] != mb[i])
                    net_band[bn] += int(ma[i]) - int(mb[i])
        tn, td = sum(n_band.values()), sum(dis_band.values())
        info[f"{a}_vs_{b}"] = {
            bn: {"item_share": n_band[bn] / tn, "discord_rate": dis_band[bn] / n_band[bn],
                 "discord_share": dis_band[bn] / td, "info_per_compute": (dis_band[bn] / td) / (n_band[bn] / tn),
                 "net_delta_items": net_band[bn]}
            for bn in [b_[2] for b_ in BANDS] if n_band[bn]}

    # ---- 4) 問題IDプール
    used = set()
    for f in all_result_files():
        try:
            for series in per_item_maps(f).values():
                used.update(series.keys())
        except Exception:
            pass
    log = json.loads((ROOT / "results/evolution_run_log.json").read_text())
    used.update(log["fitness_items"])
    for c in calls:
        used.add(c["item_id"])
    for extra in ("results/gcs/run001/transcripts_demo/transcripts_team.json",
                  "results/gcs/run001/transcripts_math500_diag/transcripts_team.json"):
        for r in json.loads((ROOT / extra).read_text()):
            used.add(r["item_id"])

    dev_pool = {}
    for bench in BENCH:
        items = sorted(i for i in p_base if bench_of_item(i) == bench)
        dev_pool[bench] = [{"item_id": i, "p_base": round(p_base[i], 4), "n_sc": sc_s[i][1],
                            "band": band_of(p_base[i]),
                            "p_c7_r0": round(c7r0[i][0] / c7r0[i][1], 4) if c7r0[i][1] else None}
                           for i in items]

    # test_pool: 新規問題（固定 seed 20260929 で抽出、各ベンチ最大 1000 問）
    rng = random.Random(20260929)
    test_pool = {}
    for bench in ("mmlu_pro", "supergpqa"):
        fresh = sorted(i for i, m in meta.items() if m["bench"] == bench and i not in used)
        rng.shuffle(fresh)
        test_pool[bench] = {"n_fresh_total": len(fresh), "sample_1000": fresh[:1000]}
    # MATH: hendrycks_math test の L4-5 から MATH-500 収録問題を除外
    from datasets import load_dataset
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    m500_set = {r["problem"].strip() for r in math500}
    fresh_math = []
    for cfg in ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"):
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="test")
        for idx, r in enumerate(ds):
            if r["level"] not in ("Level 4", "Level 5") or r["problem"].strip() in m500_set:
                continue
            ans = boxed_answer(r["solution"])
            if ans is None:
                continue
            fresh_math.append({"item_id": f"mathfull-test-{cfg}-{idx}", "subject": cfg,
                               "level": int(r["level"].split()[-1]),
                               "gold": tasks.normalize_math_answer(ans), "raw_answer": ans})
    rng.shuffle(fresh_math)
    test_pool["math_full_l45"] = {"n_fresh_total": len(fresh_math), "sample_1000": fresh_math[:1000],
                                  "note": "MATH test 全5,000問の L4-5 から MATH-500 収録の262問を除外"}

    result = {"difficulty_distribution": dist, "delta_by_band": contrib, "sigma2": sigma2,
              "n_needed": power, "se_from_discordance": se_table, "band_efficiency": eff,
              "info_per_compute_by_band": info,
              "pools": {"n_used_items": len(used),
                        "used_by_bench": {b: sum(1 for i in used if i.split('-')[0] == p)
                                          for b, p in (("mmlu_pro", "mmlupro"), ("math500", "math500"),
                                                       ("supergpqa", "supergpqa"))}}}
    dump_json(result, OUT / "ra4_item_discrimination.json")
    dump_json({"used_item_ids": sorted(used)}, OUT / "pools/used_items.json")
    dump_json(dev_pool, OUT / "pools/dev_pool_known_difficulty.json")
    dump_json(test_pool, OUT / "pools/test_pool_fresh.json")

    print("== 難易度分布（p_base = ベース SC サンプル正答率, 新環境）")
    for b, v in dist.items():
        print(f"  {b:10s} n={v['n_items']:5d} mean p={v['mean_p']:.3f}  mid[0.1,0.9]={v['mid_0.1_0.9_share']:.0%}  "
              + "  ".join(f"{k}:{x:.0%}" for k, x in v["bands"].items()))
    print("== 条件間差の難易度帯別寄与")
    for k, v in contrib.items():
        print(f"  {k}: Δ={v['delta_pt']:+.2f}pt (n_obs={v['n_obs']})  discordance="
              + " ".join(f"{b}:{d:.3f}" for b, d in v["discordance_rate"].items()))
        for bn, x in v["by_band"].items():
            sh = x["share_of_total_delta"]
            print(f"      {bn:10s} 観測比 {x['share_obs']:.0%}  帯内Δ {x['delta_in_band_pt']:+.1f}pt  "
                  f"全体差への寄与 {sh:+.0%}" if sh is not None else f"      {bn} —")
    print("== σ² = E[p(1−p)]（同一問題の再サンプリング分散）")
    for k, v in sigma2.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("== 必要問題数（80%検出力, 両側5%, 同質系統の近似）")
    for b, v in power.items():
        print(f"  {b}: " + "  ".join(f"{k}={x}" for k, x in v.items()))
    print("== 帯域フィルタの効率（z²/観測, 大きいほど少ない問題で差を検出できる）")
    for k, v in eff.items():
        print(f"  {k:28s} n={v['n_obs']:5d} Δ={v['delta_pt']:+.2f}pt z²/obs={v['z2_per_obs']:.5f} "
              f"→ z=2.8 に必要な観測数 {v['obs_for_z2.8']:.0f}")
    print("== 帯域別の不一致率と情報/計算比（不一致シェア ÷ 問題シェア）")
    for k, v in info.items():
        print(f"  {k}: " + "  ".join(f"{bn}:{x['item_share']:.0%}/不一致{x['discord_rate']:.2f}/比{x['info_per_compute']:.2f}"
                                    for bn, x in v.items()))
    print(f"== プール: 既使用 {len(used)} 問  内訳 {result['pools']['used_by_bench']}")
    for b, v in test_pool.items():
        print(f"  新規 {b}: {v['n_fresh_total']} 問")


if __name__ == "__main__":
    main()
