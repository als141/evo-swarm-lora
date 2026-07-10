"""問題IDクラスタリングを考慮した最終統計の再解析（査読対応）。

従来の final_stats_clean.py は複数シードの結果を「シード×問題」を独立観測として
プールした McNemar 正確検定を用いていたが、同一問題への複数シード回答は相関する
反復測定であり、独立性の仮定を満たさない（p 値が反保守的になる）。

本スクリプトは問題 ID をクラスタ単位として:
  (1) 問題ごとにシード平均を取った per-item Δ_i を構成
  (2) クラスタ符号反転置換検定（item-level sign-flip permutation, 20,000回）で p 値
  (3) クラスタブートストラップ（item resampling, 10,000回）で 95% / 90% CI
  (4) 「互角」主張には TOST 型の同等性判定（等価マージン ±2pt、90%CI ⊂ ±2pt）
を計算する。従来のプール McNemar p 値も参考値として併記する。

対象:
  - 主検定: c7 vs c2 (SC@9)、6シード（新環境）
  - 副検定: c7vsc5 / c7vsc1 / c5vsc1 / c5vsc2 / c1vsc2、3シード（新環境）
  - Holm 補正（主+副 = 6 比較、置換 p 値に適用）
  - 同等性: c7vsc1（プール）、c7vsc2（MATH-500）
  - 実験1の領域依存: c3(素の議論) vs c1、ベンチ別（旧環境 final_eval3、3シード）

実行: uv run python scripts/analysis/final_stats_clustered.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[2]
R2 = ROOT / "results/gcs/run002"
E3 = ROOT / "results/gcs/run001/final_eval3"
sys.path.insert(0, str(ROOT))
from src.evalx.stats import mcnemar_exact  # noqa: E402

BENCHES = ["mmlu_pro", "math500", "supergpqa"]
RNG_PERM = np.random.default_rng(20260710)
RNG_BOOT = np.random.default_rng(20260711)
N_PERM = 20_000
N_BOOT = 10_000
EQUIV_MARGIN = 2.0  # pt。本研究で意味があるとみなす最小差（処方効果+3.2ptより小さい）


# ---------------- データ読み込み ----------------
def path_for(cond: str, bench: str, seed: int) -> Path:
    if cond == "c7":
        if seed == 1:
            return R2 / f"g3_team_check/g3_{bench}.json"
        if seed in (2, 3):
            return R2 / f"final_c7/c7_run002_team_{bench}_s{seed}.json"
        return R2 / f"robust_c7/c7_run002_team_{bench}_s{seed}.json"
    if cond == "c2":
        if bench == "mmlu_pro" and seed == 1:
            return R2 / "recheck_c2s1/c2_sc9_mmlu_pro_s1_recheck.json"
        if seed <= 3:
            return R2 / f"remeasure_v1/r_c2_sc9_{bench}_s{seed}.json"
        return R2 / f"robust_c2/c2_sc9_{bench}_s{seed}.json"
    if cond == "c1":
        return R2 / f"remeasure_v1/r_c1_base_solo_{bench}_s{seed}.json"
    if cond == "c5":
        return R2 / f"remeasure_v1/r_c5_evolved_team_{bench}_s{seed}.json"
    # 実験1（旧環境）
    return E3 / f"{cond}_{bench}_s{seed}.json"


def correct_map(cond: str, bench: str, seed: int) -> dict:
    d = json.loads(path_for(cond, bench, seed).read_text())
    if d.get("team"):
        pi = d["team"]["per_item"]
    elif d.get("sc"):
        pi = next(iter(d["sc"].values()))["per_item"]
    else:
        pi = next(iter(d["solo"].values()))["per_item"]
    return {k: bool(v["correct"]) for k, v in pi.items()}


def item_deltas(cond_a, cond_b, seeds, benches):
    """問題ごとの Δ_i = mean_s(a_is − b_is) と、プール McNemar 用ペアを返す。"""
    per_item = defaultdict(list)
    pairs = []
    for bench in benches:
        for seed in seeds:
            ca = correct_map(cond_a, bench, seed)
            cb = correct_map(cond_b, bench, seed)
            for k in ca.keys() & cb.keys():
                per_item[f"{bench}:{k}"].append(int(ca[k]) - int(cb[k]))
                pairs.append((ca[k], cb[k]))
    deltas = np.array([np.mean(v) for v in per_item.values()])
    return deltas, pairs


# ---------------- 検定 ----------------
def cluster_sign_flip_p(deltas: np.ndarray) -> float:
    """クラスタ（問題）単位の符号反転置換検定（両側）。"""
    obs = abs(deltas.mean())
    n = len(deltas)
    signs = RNG_PERM.choice([-1.0, 1.0], size=(N_PERM, n))
    null = np.abs((signs * deltas).mean(axis=1))
    return float((np.sum(null >= obs - 1e-15) + 1) / (N_PERM + 1))


def cluster_bootstrap_ci(deltas: np.ndarray, level: float) -> tuple:
    n = len(deltas)
    idx = RNG_BOOT.integers(0, n, size=(N_BOOT, n))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def analyze(cond_a, cond_b, seeds, benches, label):
    deltas, pairs = item_deltas(cond_a, cond_b, seeds, benches)
    d_pt = deltas.mean() * 100
    p_perm = cluster_sign_flip_p(deltas)
    ci95 = tuple(x * 100 for x in cluster_bootstrap_ci(deltas, 0.95))
    ci90 = tuple(x * 100 for x in cluster_bootstrap_ci(deltas, 0.90))
    p_naive = mcnemar_exact(pairs)["p_value"]
    return {
        "label": label, "delta_pt": d_pt, "n_items": len(deltas),
        "n_obs": len(pairs), "p_perm": p_perm, "ci95": ci95, "ci90": ci90,
        "p_naive_mcnemar": p_naive,
    }


def show(r, indent=""):
    lo, hi = r["ci95"]
    print(f"{indent}{r['label']:<38} Δ={r['delta_pt']:+.2f}pt "
          f"[95%CI {lo:+.2f}, {hi:+.2f}] p_perm={r['p_perm']:.5f} "
          f"(問題{r['n_items']}件/観測{r['n_obs']}件, 旧p={r['p_naive_mcnemar']:.6f})")


def main():
    print("=" * 100)
    print("A. 主検定: c7(新チーム) vs c2(SC@9) — 6シード、問題IDクラスタ解析")
    print("=" * 100)
    main_r = analyze("c7", "c2", range(1, 7), BENCHES, "c7 vs SC@9（3ベンチ・プール）")
    show(main_r)
    per_bench_main = {}
    for b in BENCHES:
        r = analyze("c7", "c2", range(1, 7), [b], f"  {b}")
        per_bench_main[b] = r
        show(r, "  ")

    print()
    print("=" * 100)
    print("B. 副検定 — 3シード、問題IDクラスタ解析")
    print("=" * 100)
    subs = []
    for ca, cb, lab in [("c7", "c5", "c7 vs 旧チーム（処方の効果）"),
                        ("c7", "c1", "c7 vs ベース"),
                        ("c5", "c1", "旧チーム vs ベース"),
                        ("c5", "c2", "旧チーム vs SC@9"),
                        ("c1", "c2", "ベース vs SC@9")]:
        r = analyze(ca, cb, [1, 2, 3], BENCHES, lab)
        subs.append(r)
        show(r)
        for b in BENCHES:
            show(analyze(ca, cb, [1, 2, 3], [b], f"  {b}"), "  ")

    print()
    print("--- Holm 補正（主検定+副検定5 = 6比較、置換p値に適用）---")
    tests = [("★ " + main_r["label"], main_r["p_perm"])] + \
            [(r["label"], r["p_perm"]) for r in subs]
    m = len(tests)
    for rank, (lab, p) in enumerate(sorted(tests, key=lambda x: x[1])):
        adj = min(1.0, p * (m - rank))
        print(f"  {lab:<40} p={p:.5f} p_adj={adj:.5f} {'有意' if adj < 0.05 else 'n.s.'}")

    print()
    print("=" * 100)
    print(f"C. 同等性の検定（TOST型: 90%CI が ±{EQUIV_MARGIN}pt に収まれば α=0.05 で同等）")
    print("=" * 100)
    for r, note in [
        (analyze("c7", "c1", [1, 2, 3], BENCHES, "c7 vs ベース（プール）"), "『ベースと互角』の主張"),
        (analyze("c7", "c2", range(1, 7), ["math500"], "c7 vs SC@9（MATH-500）"), "『数学で肩を並べた』の主張"),
    ]:
        lo, hi = r["ci90"]
        equiv = (-EQUIV_MARGIN < lo) and (hi < EQUIV_MARGIN)
        print(f"{r['label']:<34} Δ={r['delta_pt']:+.2f}pt 90%CI[{lo:+.2f},{hi:+.2f}] "
              f"→ 同等性{'成立' if equiv else '不成立（未確定）'}  ({note})")

    print()
    print("=" * 100)
    print("D. 実験1 領域依存: 素の議論(c3) vs ベース(c1) — 旧環境・3シード、クラスタ解析")
    print("=" * 100)
    for b in BENCHES:
        show(analyze("c3_base_team", "c1_base_solo", [1, 2, 3], [b], f"素の議論−ベース {b}"))


if __name__ == "__main__":
    main()
