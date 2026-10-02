"""v4 の副次（探索的）解析。docs/research_design_v4.md §5 の「副次」を保存済みの生出力から計算する。

1. SC@k 曲線（k=1..9）: 9 標本から k 個を選ぶ全組合せの平均（期待値）。ベースと RFT。
2. 計算量: 条件ごとの生成回数と出力トークン（1問あたり）。チームはゲーティングで省いた議論を数えない。
3. 議論の動態: round0 の全員一致率・oracle@3・多数派正解率・最終正解率、正誤遷移、少数派正解の維持率。
4. 多様性: round0 の2体間の不一致率（同じモデルの再標本＝SC の2標本間と比べる）。
5. 系統の推移と、Shapley と単独精度の順位相関（state.json の fitness）。

使い方: python3 scripts/v4/secondary.py --spec configs/v4/stats_final.json --store <dir> \
            --state S=.../state.json --state N=... --out results/v4/secondary.json
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import tempfile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "vendor"))

from scripts.v4.stats import BENCHES, Scorer  # noqa: E402
from src.evo4.items import index_items, load_items  # noqa: E402
from src.evo4.scoring import is_correct, require_math_verify  # noqa: E402
from src.evo4.society import Agent, Society, aggregate  # noqa: E402
from src.evo4.store import CallStore  # noqa: E402


def role_of(agent_id: str) -> str:
    parts = agent_id.split(".")
    return parts[-2] if len(parts) >= 3 else parts[-1]


def macro_of(values: dict, bench_of: dict) -> float:
    by = collections.defaultdict(list)
    for i, v in values.items():
        by[bench_of[i]].append(v)
    return float(np.mean([np.mean(by[b]) for b in BENCHES if by[b]])) if by else float("nan")


def sc_curve(agent: str, ids, scorer: Scorer, bench_of: dict, seed: int = 1, kmax: int = 9) -> dict:
    per_k = {k: {} for k in range(1, kmax + 1)}
    tokens, in_tokens = [], []
    for item_id in ids:
        item = scorer.items[item_id]
        keys = [Society.sc_key(Agent(agent, "base", "", ""), item_id, seed, k) for k in range(kmax)]
        if any(scorer.store.get(k) is None for k in keys):
            continue
        tokens.append(np.mean([scorer.store.get(k).get("n_out", 0) for k in keys]))
        in_tokens.append(np.mean([scorer.store.get(k).get("n_in", 0) or 0 for k in keys]))
        scored = [scorer.correct_of(k) for k in keys]
        for k in range(1, kmax + 1):
            accs = []
            for combo in itertools.combinations(range(kmax), k):
                final = aggregate([(scored[j][0], scored[j][1], "x") for j in combo])
                accs.append(float(is_correct(final, item.gold, item.answer_type)))
            per_k[k][item_id] = float(np.mean(accs))
    return {"n_items": len(per_k[1]), "mean_out_tokens_per_call": float(np.mean(tokens)) if tokens else None,
            "mean_in_tokens_per_call": float(np.mean(in_tokens)) if in_tokens else None,
            "macro_by_k": {k: macro_of(v, bench_of) for k, v in per_k.items() if v}}


def sc_pair_disagreement(agent: str, ids, scorer: Scorer, seed: int = 1) -> float | None:
    """同じモデル・同じプロンプトの再標本（SC の標本 0,1,2）の2体間不一致率。多様性の基準線。"""
    rates = []
    for item_id in ids:
        keys = [Society.sc_key(Agent(agent, "base", "", ""), item_id, seed, k) for k in range(3)]
        if any(scorer.store.get(k) is None for k in keys):
            continue
        ans = [scorer.correct_of(k)[0] for k in keys]
        rates.append(np.mean([ans[a] != ans[b] for a, b in itertools.combinations(range(3), 2)]))
    return float(np.mean(rates)) if rates else None


def team_dynamics(cond: dict, ids, scorer: Scorer, bench_of: dict) -> dict:
    soc = Society(None, scorer.store, scorer.items, None, protocol=cond.get("protocol", "v4t"),
                  gate=bool(cond.get("gate", False)))
    agents = [Agent(a, role_of(a), "", "") for a in cond["agents"]]
    c = collections.Counter()
    calls, out_tokens, in_tokens, n_items = [], [], [], 0
    disagree = []
    solo = {a.agent_id: {} for a in agents}  # 構成員の round0 の単独正答（問題ごと、生成 seed 平均）
    r0_vote = {}
    for seed in cond["gen_seeds"]:
        for item_id in ids:
            r0 = [scorer.correct_of(soc.r0_key(a, item_id, seed)) for a in agents]
            recs0 = [scorer.store.get(soc.r0_key(a, item_id, seed)) for a in agents]
            if any(r is None for r in recs0):
                continue
            unanimous = soc.gate and soc._r0_unanimous(agents, item_id, seed)
            r1_keys = [soc.r1_key(a, [o for o in agents if o is not a], item_id, seed) for a in agents]
            recs1 = None if unanimous else [scorer.store.get(k) for k in r1_keys]
            if recs1 is not None and any(r is None for r in recs1):
                c["missing_r1"] += 1  # 生成途中など。stats.py と同じく集計から除く
                continue
            n_items += 1
            ans0 = [r[0] for r in r0]
            ok0 = [bool(r[2]) for r in r0]
            for a, ok in zip(agents, ok0):
                solo[a.agent_id].setdefault(item_id, []).append(float(ok))
            item = scorer.items[item_id]
            vote0 = aggregate([(r[0], r[1], a.role) for r, a in zip(r0, agents)])
            r0_vote.setdefault(item_id, []).append(float(is_correct(vote0, item.gold, item.answer_type)))
            disagree.append(np.mean([ans0[a] != ans0[b] for a, b in itertools.combinations(range(3), 2)]))
            n_ok0 = sum(ok0)
            c["oracle3"] += n_ok0 > 0
            c["majority0_correct"] += n_ok0 >= 2
            c[f"r0_correct_{n_ok0}"] += 1
            tok = sum(r.get("n_out", 0) for r in recs0)
            tok_in = sum(r.get("n_in", 0) or 0 for r in recs0)
            n_call = 3
            res = soc.coalition_result(agents, item_id, seed)
            c["final_correct"] += bool(res["correct"])
            if unanimous:
                c["gated"] += 1
            else:
                tok += sum(r.get("n_out", 0) for r in recs1)
                tok_in += sum(r.get("n_in", 0) or 0 for r in recs1)
                n_call += 3
                ok1 = [bool(scorer.correct_of(k)[2]) for k in r1_keys]
                for before, after in zip(ok0, ok1):
                    c[f"trans_{int(before)}{int(after)}"] += 1
                if n_ok0 == 1:  # 少数派（1体）だけが正解
                    idx = ok0.index(True)
                    c["minority_correct_items"] += 1
                    c["minority_kept"] += ok1[idx]
                    c["minority_team_correct"] += bool(res["correct"])
                if n_ok0 == 2:  # 多数派（2体）が正解
                    c["majority_correct_items"] += 1
                    c["majority_team_correct"] += bool(res["correct"])
            calls.append(n_call)
            out_tokens.append(tok)
            in_tokens.append(tok_in)
    if not n_items:
        return {"n": 0}
    kept = c["trans_11"] + c["trans_10"]
    gained = c["trans_01"] + c["trans_00"]
    solo_macro = {a: macro_of({i: float(np.mean(v)) for i, v in d.items()}, bench_of) for a, d in solo.items()}
    return {
        "n": n_items,
        "member_solo_macro": solo_macro,
        "member_solo_mean": float(np.mean(list(solo_macro.values()))),
        "r0_majority_macro": macro_of({i: float(np.mean(v)) for i, v in r0_vote.items()}, bench_of),
        "calls_per_item": float(np.mean(calls)), "out_tokens_per_item": float(np.mean(out_tokens)),
        "in_tokens_per_item": float(np.mean(in_tokens)),
        "gated_rate": c["gated"] / n_items, "oracle3": c["oracle3"] / n_items,
        "majority0_correct": c["majority0_correct"] / n_items, "final_correct": c["final_correct"] / n_items,
        "r0_pair_disagreement": float(np.mean(disagree)),
        "r0_n_correct_dist": {k: c[f"r0_correct_{k}"] / n_items for k in range(4)},
        "correct_to_wrong_rate": c["trans_10"] / kept if kept else None,
        "wrong_to_correct_rate": c["trans_01"] / gained if gained else None,
        "minority_correct_items": c["minority_correct_items"],
        "minority_kept_rate": c["minority_kept"] / c["minority_correct_items"] if c["minority_correct_items"] else None,
        "minority_team_correct_rate": (c["minority_team_correct"] / c["minority_correct_items"]
                                       if c["minority_correct_items"] else None),
        "majority_team_correct_rate": (c["majority_team_correct"] / c["majority_correct_items"]
                                       if c["majority_correct_items"] else None),
        "missing_r1": c["missing_r1"],
    }


def rank_corr(x, y) -> float | None:
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def lineage_summary(state: dict) -> dict:
    out = {"generations": {}}
    xs, ys, disagreements, total = [], [], 0, 0
    for t, g in sorted(state["generations"].items(), key=lambda kv: int(kv[0])):
        row = {"reps": {r: d["agent_id"] for r, d in g.get("reps", {}).items()},
               "dev_team": (g.get("dev_team") or {}).get("macro"),
               "test_team": (g.get("test_team") or {}).get("macro"),
               "test_team_by_seed": {s: v.get("macro") for s, v in (g.get("test_team_by_seed") or {}).items()},
               "train_team": (g.get("train_team") or {}).get("macro"),
               "data_counts": g.get("data_counts")}
        fit = g.get("fitness") or {}
        by_role = collections.defaultdict(list)
        for agent_id, f in fit.items():
            by_role[f["role"]].append((agent_id, f))
        per_role = {}
        for role, rows in by_role.items():
            if "shapley" in rows[0][1]:
                sh = [f["shapley"] for _, f in rows]
                so = [f["solo"] for _, f in rows]
                xs += sh
                ys += so
                best_sh = max(rows, key=lambda r: (r[1]["shapley"], r[0]))[0]
                best_so = max(rows, key=lambda r: (r[1]["solo"], r[0]))[0]
                total += 1
                disagreements += best_sh != best_so
                per_role[role] = {"spearman_shapley_solo": rank_corr(sh, so), "argmax_shapley": best_sh,
                                  "argmax_solo": best_so,
                                  "banzhaf_agrees": max(rows, key=lambda r: (r[1]["banzhaf"], r[0]))[0] == best_sh}
        if per_role:
            row["selection"] = per_role
        out["generations"][t] = row
    if total:
        out["shapley_vs_solo"] = {"spearman_all": rank_corr(xs, ys), "argmax_differs": disagreements,
                                  "decisions": total}
    return out


def main() -> None:
    require_math_verify()  # 手元で採点するとき math-verify が無いと MATH を過小評価する
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--items", default=str(ROOT / "data/v4/items/test.jsonl"))
    parser.add_argument("--state", action="append", default=[])
    parser.add_argument("--sc-agents", default="base,rft.base")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    items = index_items([args.items])
    ids = [i.item_id for i in load_items(args.items)]
    bench_of = {i: items[i].bench for i in ids}
    store = CallStore(tempfile.mkdtemp(prefix="evo4_secondary_local_"), args.store, resolve="earliest")
    print(f"[secondary] records={len(store)} duplicated_keys={store.duplicates} (earliest t_start wins)", file=sys.stderr)
    scorer = Scorer(store, items)
    result = {"sc_curves": {}, "sc_pair_disagreement": {}, "teams": {}, "lineages": {}}
    for agent in [a for a in args.sc_agents.split(",") if a]:
        result["sc_curves"][agent] = sc_curve(agent, ids, scorer, bench_of)
        result["sc_pair_disagreement"][agent] = sc_pair_disagreement(agent, ids, scorer)
    for name, cond in spec["conditions"].items():
        if cond["type"] == "team":
            result["teams"][name] = team_dynamics(cond, ids, scorer, bench_of)
    for s in args.state:
        name, path = s.split("=", 1)
        result["lineages"][name] = lineage_summary(json.loads(Path(path).read_text()))
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
