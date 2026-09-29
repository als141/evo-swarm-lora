"""【解析3・6】チームの多様性・誤り相関・oracle・集約損失と、議論による正誤遷移（追従の定量）。

データ（新環境 llm_calls を問題単位に再構成）:
  c7 = run002 再学習チーム（conditional 更新・匿名化・weighted 投票）6 seeds × 1000問
  c5 = 実験1 進化後チーム（standard 更新・ラベル付き・多数決）3 seeds × 1000問
  base SC@9 サンプル（比較用: 「3サンプル」の oracle / 多数決を同条件で算出）
最終チーム回答の正誤は結果 JSON（per_item）を正とする。

出力（ベンチ別）:
  - solo平均(r0) / oracle@3(r0) / r0多数決 / r1多数決 / 最終 / 集約損失 = oracle − 最終
  - ベース3サンプルの oracle@3 / 多数決@3 と、SC@9 の oracle@9 / 多数決@9（多様性の比較）
  - pairwise: 回答一致率・誤り相関(phi)・同一誤答率(coincident error)・double fault
  - 問題単位: r0 の相異なる回答数 → チーム利得（最終 − r0 solo 平均）
  - 遷移: エージェント単位 r0→r1 の C→C / C→W / W→C / W→W、多数派/少数派別の変更率、
          「正しい多数派の崩壊」「正しい少数派の逆転」件数
実行: python3 scripts/analysis/ra3_team_dynamics.py
"""
import itertools
import random
from collections import Counter, defaultdict

import numpy as np

from ra_common import NEW_SEEDS, OUT, dump_json, load_calls_index, new_path, per_item_maps

BENCH = ("mmlu_pro", "math500", "supergpqa")


def majority(answers, seed):
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    cnt = Counter(valid)
    top = max(cnt.values())
    win = sorted(a for a, c in cnt.items() if c == top)
    return win[0] if len(win) == 1 else random.Random(seed).choice(win)


def phi(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def build_instances(calls):
    """(family, bench, seed, item) → {r0: {agent_idx: row}, r1: {...}}"""
    inst = defaultdict(lambda: {0: {}, 1: {}})
    for c in calls:
        if c["kind"] == "team" and c["dir"] != "g2_aggregation" and c.get("item_id"):
            inst[(c["family"], c["bench"], c["run_seed"], c["item_id"])][c["round"]][c["agent_idx"]] = c
    return inst


def final_correct_maps():
    out = {}
    for fam in ("c7", "c5"):
        for bench in BENCH:
            for s in NEW_SEEDS[fam]:
                p = new_path(fam, bench, s)
                if p:
                    out[(fam, bench, s)] = {k: v["correct"] for k, v in per_item_maps(p)["team"].items()}
    return out


def main():
    calls = [c for c in load_calls_index() if c.get("item_id")]
    inst = build_instances(calls)
    finals = final_correct_maps()
    result = {"teams": {}, "sc_reference": {}, "transitions": {}}

    for fam in ("c7", "c5"):
        for bench in BENCH:
            rows = []
            for (f, b, s, item), rr in inst.items():
                if f != fam or b != bench or len(rr[0]) != 3 or len(rr[1]) != 3:
                    continue
                fin = finals.get((fam, bench, s), {}).get(item)
                if fin is None:
                    continue
                a0 = [rr[0][i]["ans"] for i in range(3)]
                o0 = [bool(rr[0][i]["ok"]) for i in range(3)]
                a1 = [rr[1][i]["ans"] for i in range(3)]
                o1 = [bool(rr[1][i]["ok"]) for i in range(3)]
                gold = rr[0][0]["gold"]
                m0 = majority(a0, s)
                m1 = majority(a1, s)
                rows.append({"seed": s, "item": item, "a0": a0, "o0": o0, "a1": a1, "o1": o1,
                             "fin": bool(fin), "gold": gold, "m0": m0, "m1": m1})
            if not rows:
                continue
            # 多数決回答の正誤は「回答→正誤」の写像で判定（math の数値同値判定を再利用するため）
            for r in rows:
                amap = {}
                for a, o in zip(r["a0"] + r["a1"], r["o0"] + r["o1"]):
                    if a is not None:
                        amap[a] = amap.get(a, False) or o
                r["m0_ok"] = bool(amap.get(r["m0"], False)) if r["m0"] is not None else False
                r["m1_ok"] = bool(amap.get(r["m1"], False)) if r["m1"] is not None else False
            n = len(rows)
            solo = np.mean([np.mean(r["o0"]) for r in rows])
            oracle = np.mean([any(r["o0"]) for r in rows])
            m0 = np.mean([r["m0_ok"] for r in rows])
            m1 = np.mean([r["m1_ok"] for r in rows])
            fin = np.mean([r["fin"] for r in rows])
            # pairwise
            pw = {}
            for i, j in itertools.combinations(range(3), 2):
                ei = [not r["o0"][i] for r in rows]
                ej = [not r["o0"][j] for r in rows]
                agree = np.mean([r["a0"][i] == r["a0"][j] for r in rows])
                both_wrong = np.mean([ei_ and ej_ for ei_, ej_ in zip(ei, ej)])
                same_wrong = np.mean([(not r["o0"][i]) and (not r["o0"][j]) and r["a0"][i] == r["a0"][j]
                                      and r["a0"][i] is not None for r in rows])
                pw[f"{i}{j}"] = {"agree": float(agree), "err_phi": phi(ei, ej),
                                 "double_fault": float(both_wrong), "same_wrong": float(same_wrong)}
            # 問題単位: r0 の相異なる回答数別のチーム利得
            by_div = defaultdict(list)
            for r in rows:
                k = len({a for a in r["a0"]})
                by_div[k].append(float(r["fin"]) - float(np.mean(r["o0"])))
            div = {k: {"n": len(v), "share": len(v) / n, "mean_gain_pt": float(np.mean(v) * 100)}
                   for k, v in sorted(by_div.items())}
            # 遷移（エージェント単位）
            tr = Counter()
            change_by_status = defaultdict(lambda: [0, 0])  # status -> [changed, total]
            for r in rows:
                for i in range(3):
                    tr[("C" if r["o0"][i] else "W") + "→" + ("C" if r["o1"][i] else "W")] += 1
                    others = [r["a0"][j] for j in range(3) if j != i]
                    maj_other = others[0] if others[0] == others[1] and others[0] is not None else None
                    if maj_other is None:
                        status = "others_split"
                    elif r["a0"][i] == maj_other:
                        status = "in_majority"
                    else:
                        status = ("minority_vs_correct_majority" if any(
                            r["o0"][j] for j in range(3) if j != i) else "minority_vs_wrong_majority")
                        status += "_self_correct" if r["o0"][i] else "_self_wrong"
                    changed = r["a1"][i] != r["a0"][i]
                    change_by_status[status][0] += int(changed)
                    change_by_status[status][1] += 1
            maj_broken = sum(1 for r in rows if r["m0_ok"] and not r["fin"])
            minority_rescued = sum(1 for r in rows if (not r["m0_ok"]) and r["fin"])
            total_agent = sum(tr.values())
            result["teams"][f"{fam}_{bench}"] = {
                "n_instances": n, "solo_r0": float(solo), "oracle3_r0": float(oracle),
                "maj_r0": float(m0), "maj_r1": float(m1), "final": float(fin),
                "aggregation_loss_pt": float((oracle - fin) * 100),
                "debate_gain_over_r0maj_pt": float((fin - m0) * 100),
                "pairwise": pw, "gain_by_r0_distinct_answers": div,
            }
            result["transitions"][f"{fam}_{bench}"] = {
                "agent_transitions": {k: v / total_agent for k, v in sorted(tr.items())},
                "change_rate_by_status": {k: {"rate": v[0] / v[1], "n": v[1]}
                                          for k, v in sorted(change_by_status.items())},
                "correct_r0_majority_broken": maj_broken,
                "wrong_r0_majority_rescued": minority_rescued,
                "net_items": minority_rescued - maj_broken, "n_instances": n,
            }

    # ベース SC サンプルでの「3サンプル」比較（同じ run-seed・同じ問題）
    sc = defaultdict(dict)
    for c in calls:
        if c["kind"] == "sc":
            sc[(c["bench"], c["run_seed"], c["item_id"])][c["sample_idx"]] = c
    for bench in BENCH:
        rows = [v for (b, _s, _i), v in sc.items() if b == bench and len(v) == 9]
        o = [[bool(v[k]["ok"]) for k in range(9)] for v in rows]
        a = [[v[k]["ans"] for k in range(9)] for v in rows]
        seeds = [s for (b, s, _i), v in sc.items() if b == bench and len(v) == 9]
        def maj_ok(ans, oks, seed):
            amap = {}
            for x, y in zip(ans, oks):
                if x is not None:
                    amap[x] = amap.get(x, False) or y
            m = majority(ans, seed)
            return bool(amap.get(m, False)) if m is not None else False
        pw_agree = np.mean([np.mean([x[i] == x[j] for i, j in itertools.combinations(range(3), 2)]) for x in a])
        e = np.array([[not y for y in row] for row in o], float)
        err_phi = np.nanmean([phi(e[:, i], e[:, j]) for i, j in itertools.combinations(range(9), 2)])
        result["sc_reference"][bench] = {
            "n": len(rows),
            "solo_mean": float(np.mean(o)),
            "oracle3": float(np.mean([any(r[:3]) for r in o])),
            "maj3": float(np.mean([maj_ok(x[:3], y[:3], s) for x, y, s in zip(a, o, seeds)])),
            "oracle6": float(np.mean([any(r[:6]) for r in o])),
            "oracle9": float(np.mean([any(r) for r in o])),
            "maj9": float(np.mean([maj_ok(x, y, s) for x, y, s in zip(a, o, seeds)])),
            "pairwise_agree_3": float(pw_agree), "err_phi_mean": float(err_phi),
        }
    dump_json(result, OUT / "ra3_team_dynamics.json")

    print("== チーム: solo(r0) / oracle@3 / r0多数決 / r1多数決 / 最終 / 集約損失 / 議論利得(最終−r0多数決)")
    for k, v in result["teams"].items():
        print(f"  {k:16s} n={v['n_instances']:5d} solo={v['solo_r0']:.3f} oracle3={v['oracle3_r0']:.3f} "
              f"maj0={v['maj_r0']:.3f} maj1={v['maj_r1']:.3f} final={v['final']:.3f} "
              f"loss={v['aggregation_loss_pt']:+.1f}pt debate={v['debate_gain_over_r0maj_pt']:+.1f}pt")
        pw = v["pairwise"]
        print("      pairwise agree/err_phi/double_fault/same_wrong: " + "  ".join(
            f"{p}:{d['agree']:.2f}/{d['err_phi']:.2f}/{d['double_fault']:.2f}/{d['same_wrong']:.2f}"
            for p, d in pw.items()))
        print("      distinct answers → gain: " + "  ".join(
            f"{k2}:{d['share']:.0%},{d['mean_gain_pt']:+.1f}pt" for k2, d in v["gain_by_r0_distinct_answers"].items()))
    print("== ベース SC サンプル（同条件）: solo / oracle@3 / maj@3 / oracle@9 / maj@9 / 3サンプル一致率 / 誤りphi")
    for b, v in result["sc_reference"].items():
        print(f"  {b:10s} n={v['n']:5d} solo={v['solo_mean']:.3f} oracle3={v['oracle3']:.3f} maj3={v['maj3']:.3f} "
              f"oracle6={v['oracle6']:.3f} oracle9={v['oracle9']:.3f} maj9={v['maj9']:.3f} "
              f"agree3={v['pairwise_agree_3']:.2f} err_phi={v['err_phi_mean']:.2f}")
    print("== 遷移（エージェント単位 r0→r1）と状態別の回答変更率")
    for k, v in result["transitions"].items():
        t = v["agent_transitions"]
        print(f"  {k:16s} C→C {t.get('C→C',0):.3f} C→W {t.get('C→W',0):.3f} W→C {t.get('W→C',0):.3f} "
              f"W→W {t.get('W→W',0):.3f} | 正しい多数派崩壊 {v['correct_r0_majority_broken']} / "
              f"誤り多数派の逆転 {v['wrong_r0_majority_rescued']} (純 {v['net_items']:+d}, n={v['n_instances']})")
        for st, d in v["change_rate_by_status"].items():
            print(f"      {st:48s} 変更率 {d['rate']:.2f} (n={d['n']})")


if __name__ == "__main__":
    main()
