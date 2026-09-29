"""【解析1・5】進化ログ（run001, 6世代）の測定ノイズ床と Shapley 値の情報量。

進化ループは世代ごとに連合評価キャッシュを作り直し、全世代で同一の適応度セット
（MMLU-Pro 100問, seed 777）・同一生成 seed（777 系列）で測り直している。したがって
「同じメンバー構成の連合」が複数世代に現れれば、それは同一条件の再測定であり、
その値のばらつきが測定ノイズの床（vLLM の非決定性を含む）を与える。

出力:
  - 連合サイズ別の再測定 SD と、100問二項 SE との比較
  - Shapley 値（および候補間差）の誤差伝播 SD
  - 役割内選抜 18 件の「差 / ノイズ SD」比、Shapley と solo / team 選抜の一致
  - 勝者の呪い: 選抜時の solo 値と次世代での再測定値の差
実行: python3 scripts/analysis/ra1_evolution_noise.py
"""
import itertools
import json
import math
from collections import defaultdict

import numpy as np

from ra_common import OUT, ROOT, dump_json

LOG = ROOT / "results/evolution_run_log.json"
N_FIT = 100


def main():
    log = json.loads(LOG.read_text())
    gens = log["generations"]

    # ---- 1) 同一連合の再測定を集める（世代内はキャッシュ共有＝1回の測定）
    meas = defaultdict(dict)  # coalition(frozenset) -> {gen: acc}
    for g in gens:
        t = g["generation"]
        for role, rl in g["roles"].items():
            for cand, rec in rl["candidates"].items():
                for key, acc in rec["coalition_accuracy"].items():
                    members = frozenset(key.split("+"))
                    prev = meas[members].get(t)
                    assert prev is None or abs(prev - acc) < 1e-9, (members, t)
                    meas[members][t] = acc

    by_size = defaultdict(list)       # size -> list of (values)
    rows = []
    for members, series in meas.items():
        if len(series) >= 2:
            vals = [series[t] for t in sorted(series)]
            by_size[len(members)].append(vals)
            rows.append({"members": sorted(members), "gens": sorted(series), "values": vals})

    noise = {}
    for size, groups in sorted(by_size.items()):
        # プール内分散（自由度 = Σ(n_g − 1)）
        ss = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in groups)
        dof = sum(len(v) - 1 for v in groups)
        sd = math.sqrt(ss / dof) if dof else float("nan")
        pbar = float(np.mean([x for v in groups for x in v]))
        binom = math.sqrt(pbar * (1 - pbar) / N_FIT)
        diffs = [abs(a - b) for v in groups for a, b in itertools.combinations(v, 2)]
        noise[size] = {
            "n_coalitions": len(groups), "n_measurements": sum(len(v) for v in groups),
            "dof": dof, "pooled_sd": sd, "mean_acc": pbar, "binomial_se_100": binom,
            "sd_over_binom": sd / binom, "mean_abs_pair_diff": float(np.mean(diffs)),
            "max_abs_pair_diff": float(np.max(diffs)),
        }
    all_groups = [v for gs in by_size.values() for v in gs]
    ss = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in all_groups)
    dof = sum(len(v) - 1 for v in all_groups)
    sigma_all = math.sqrt(ss / dof)

    # ---- 2) Shapley の誤差伝播（連合値の測定誤差を独立と仮定）
    s1 = noise.get(1, {}).get("pooled_sd", sigma_all)
    s2 = noise.get(2, {}).get("pooled_sd", sigma_all)
    s3 = noise.get(3, {}).get("pooled_sd", sigma_all)
    # φ_c = v(c)/3 + [v(c,ρ1)-v(ρ1)]/6 + [v(c,ρ2)-v(ρ2)]/6 + [v(T)-v(ρ1,ρ2)]/3
    var_phi = s1**2 / 9 + 2 * (s2**2 + s1**2) / 36 + (s3**2 + s2**2) / 9
    # 候補差 φ_a − φ_b: 代表のみの連合 v(ρ1), v(ρ2), v(ρ1,ρ2) は相殺
    var_dphi = 2 * s1**2 / 9 + 2 * 2 * s2**2 / 36 + 2 * s3**2 / 9
    shap_err = {"sd_phi": math.sqrt(var_phi), "sd_phi_diff": math.sqrt(var_dphi),
                "sigma_used": {"solo": s1, "pair": s2, "team": s3}}

    # ---- 3) 役割内選抜 18 件: 差の大きさ / ノイズ, 選抜基準の一致
    decisions = []
    for g in gens:
        for role, rl in g["roles"].items():
            c = rl["candidates"]
            names = list(c)
            a, b = names
            dphi = c[a]["shapley"] - c[b]["shapley"]
            dsolo = c[a]["solo_accuracy"] - c[b]["solo_accuracy"]
            dteam = c[a]["team_accuracy"] - c[b]["team_accuracy"]
            sel = rl["selected"]
            by_solo = a if c[a]["solo_accuracy"] > c[b]["solo_accuracy"] else (
                b if c[b]["solo_accuracy"] > c[a]["solo_accuracy"] else "tie")
            by_team = a if c[a]["team_accuracy"] > c[b]["team_accuracy"] else (
                b if c[b]["team_accuracy"] > c[a]["team_accuracy"] else "tie")
            decisions.append({
                "gen": g["generation"], "role": role, "selected": sel,
                "abs_dphi": abs(dphi), "z_dphi": abs(dphi) / shap_err["sd_phi_diff"],
                "abs_dsolo": abs(dsolo), "abs_dteam": abs(dteam),
                "agree_with_solo": by_solo == sel, "solo_tie": by_solo == "tie",
                "agree_with_team": by_team == sel, "team_tie": by_team == "tie",
            })
    z = np.array([d["z_dphi"] for d in decisions])
    # 観測された候補差の分散 vs 純ノイズ期待値 → 真の差の分散成分（負なら 0）
    dphis = np.array([d["abs_dphi"] for d in decisions])
    obs_var = float(np.mean(dphis**2))
    true_var = max(0.0, obs_var - var_dphi)
    # 正しい方を選ぶ確率（真の差 δ, ノイズ SD s）: Φ(δ/s)
    from scipy.stats import norm
    sel_stats = {
        "n_decisions": len(decisions),
        "median_abs_dphi_pt": float(np.median(dphis) * 100),
        "median_z": float(np.median(z)), "frac_z_lt1": float(np.mean(z < 1)),
        "frac_z_lt2": float(np.mean(z < 2)),
        "obs_mean_sq_dphi": obs_var, "noise_var_dphi": var_dphi,
        "implied_true_sd_dphi": math.sqrt(true_var),
        "agree_with_solo_rate": float(np.mean([d["agree_with_solo"] for d in decisions])),
        "solo_ties": int(sum(d["solo_tie"] for d in decisions)),
        "agree_with_team_rate": float(np.mean([d["agree_with_team"] for d in decisions])),
        "team_ties": int(sum(d["team_tie"] for d in decisions)),
        "p_correct_choice_if_true_diff": {
            f"{delta_pt}pt": float(norm.cdf(delta_pt / 100 / shap_err["sd_phi_diff"]))
            for delta_pt in (1, 2, 3, 5)
        },
    }

    # ---- 4) 勝者の呪い: 選ばれた個体の solo 値 → 次世代の同個体 solo 再測定
    solo_by = defaultdict(dict)
    for members, series in meas.items():
        if len(members) == 1:
            (name,) = tuple(members)
            solo_by[name] = series
    curse = []
    for g in gens[:-1]:
        t = g["generation"]
        for role, rl in g["roles"].items():
            sel = rl["selected"]
            other = [n for n in rl["candidates"] if n != sel][0]
            if t + 1 in solo_by[sel]:
                curse.append({"gen": t, "role": role, "selected": sel,
                              "solo_at_sel": solo_by[sel][t], "solo_next": solo_by[sel][t + 1],
                              "loser_solo": solo_by[other][t]})
    d_next = [c["solo_next"] - c["solo_at_sel"] for c in curse]

    # 勝者の呪い（チーム水準）: 各世代の「最良候補チームの精度（選抜時の最大値）」と、
    # 選ばれた代表3体のチームを次世代冒頭で測り直した値（代表のみの3連合）の比較
    rep_traj = []
    for i, g in enumerate(gens):
        # 当世代の評価文脈 = 前世代で選ばれた代表（gen0 は初期代表）
        ctx = (gens[i - 1]["representatives"] if i > 0 else
               {r: [n for n in rl["candidates"] if n.endswith("_base")][0]
                for r, rl in g["roles"].items()})
        key = frozenset(ctx.values())
        ctx_acc = meas[key].get(g["generation"])
        best_cand_team = max(rec["team_accuracy"] for rl in g["roles"].values()
                             for rec in rl["candidates"].values())
        rep_traj.append({"gen": g["generation"], "context_team": sorted(key),
                         "context_team_acc": ctx_acc, "max_candidate_team_acc": best_cand_team})
    # Shapley と solo の相関（候補レベル, 36 個体-世代）
    cand = []
    for g in gens:
        for role, rl in g["roles"].items():
            for name, rec in rl["candidates"].items():
                cand.append((rec["shapley"], rec["solo_accuracy"], rec["team_accuracy"],
                             rec["shapley"] - rec["solo_accuracy"] / 3))
    arr = np.array(cand)
    corr = {
        "shapley_vs_solo": float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]),
        "shapley_vs_team": float(np.corrcoef(arr[:, 0], arr[:, 2])[0, 1]),
        "coop_part_vs_solo": float(np.corrcoef(arr[:, 3], arr[:, 1])[0, 1]),
        "coop_part_vs_team": float(np.corrcoef(arr[:, 3], arr[:, 2])[0, 1]),
        "coop_part_share_of_phi_var": float(np.var(arr[:, 3]) / np.var(arr[:, 0])),
        "n": len(cand),
    }
    result = {
        "noise_by_coalition_size": noise, "sigma_all_sizes": sigma_all,
        "repeated_coalitions": rows, "shapley_error": shap_err,
        "selection": sel_stats, "decisions": decisions,
        "winners_curse": {"cases": curse, "mean_solo_change_next_gen_pt": float(np.mean(d_next) * 100),
                          "n": len(curse), "rep_team_trajectory": rep_traj},
        "shapley_correlations": corr,
    }
    dump_json(result, OUT / "ra1_evolution_noise.json")

    print("== 再測定による測定SD（100問, 同一 seed 777） ==")
    for size, v in noise.items():
        print(f" size={size}: 連合{v['n_coalitions']}件/測定{v['n_measurements']}回  pooled SD="
              f"{v['pooled_sd']*100:.2f}pt  二項SE={v['binomial_se_100']*100:.2f}pt  "
              f"比={v['sd_over_binom']:.2f}  平均|差|={v['mean_abs_pair_diff']*100:.1f}pt  "
              f"最大|差|={v['max_abs_pair_diff']*100:.0f}pt")
    print(f" 全サイズ pooled SD = {sigma_all*100:.2f}pt")
    print(f"== Shapley 誤差: SD(φ)={shap_err['sd_phi']*100:.2f}pt  SD(φa−φb)={shap_err['sd_phi_diff']*100:.2f}pt")
    s = sel_stats
    print(f"== 選抜18件: 中央|Δφ|={s['median_abs_dphi_pt']:.2f}pt  中央z={s['median_z']:.2f}  "
          f"z<1:{s['frac_z_lt1']:.0%}  z<2:{s['frac_z_lt2']:.0%}")
    print(f"   観測E[Δφ²]={s['obs_mean_sq_dphi']:.5f} vs ノイズ期待={s['noise_var_dphi']:.5f} → "
          f"真の差SD≈{s['implied_true_sd_dphi']*100:.2f}pt")
    print(f"   solo基準と一致={s['agree_with_solo_rate']:.0%}（同点{s['solo_ties']}） team基準と一致="
          f"{s['agree_with_team_rate']:.0%}（同点{s['team_ties']}）")
    print(f"   真の差 δ で正しい方を選ぶ確率: {s['p_correct_choice_if_true_diff']}")
    print(f"== 勝者の呪い: 選抜個体の solo 次世代変化 平均 {np.mean(d_next)*100:+.2f}pt (n={len(curse)})")
    print("== 代表チーム（評価文脈）の精度推移 vs 当世代の最大候補チーム精度")
    for r in rep_traj:
        print(f"   gen{r['gen']}: 文脈チーム={r['context_team_acc']}  最大候補チーム={r['max_candidate_team_acc']}")
    print(f"== 相関: {corr}")


if __name__ == "__main__":
    main()
