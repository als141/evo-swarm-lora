"""v4 の事前登録した統計解析（docs/research_design_v4.md §5）。保存済みの生出力をオフラインで採点する。

条件の定義（JSON）:
  {"conditions": {
     "S_final": {"type": "team", "agents": ["S.g3.critic.b", ...], "gen_seeds": [1, 2], "protocol": "v4"},
     "base_single": {"type": "single", "agent": "base", "k": 9, "gen_seeds": [1, 2]},
     "sc9": {"type": "sc", "agent": "base", "k": 9, "gen_seeds": [1, 2]}},
   "comparisons": [["S_final", "g0"], ...],
   "holm": [["S_final", "g0"], ["S_final", "N_final"], ["S_final", "A1_final"]]}

主要指標: ベンチ等重み（macro）精度。問題ごとに生成 seed 平均を取ってから比較する。
検定: 問題 ID をクラスタとする符号反転置換（ベンチ内で反転、20,000 回）。区間: ベンチ内で問題を再抽出するブートストラップ（10,000 回）。
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evo4.items import index_items, load_items  # noqa: E402
from src.evo4.scoring import extract_answer, is_correct  # noqa: E402
from src.evo4.society import Agent, Society, aggregate  # noqa: E402
from src.evo4.store import CallStore  # noqa: E402

BENCHES = ("mmlu_pro", "supergpqa", "math")


class Scorer:
    def __init__(self, store: CallStore, items: dict):
        self.store = store
        self.items = items
        self._cache = {}

    def correct_of(self, key: str) -> tuple:
        if key in self._cache:
            return self._cache[key]
        rec = self.store.get(key)
        if rec is None:
            self._cache[key] = (None, None, None)
            return self._cache[key]
        item = self.items[rec["item"]]
        ans = extract_answer(rec["text"], item.answer_type)
        self._cache[key] = (ans, rec.get("mean_logprob"), is_correct(ans, item.gold, item.answer_type))
        return self._cache[key]


def per_item(cond: dict, ids: list, scorer: Scorer, soc_by_protocol: dict) -> dict:
    """条件の問題別正答（生成 seed 平均）。欠損の問題は含めない。"""
    out = {}
    for item_id in ids:
        item = scorer.items[item_id]
        vals = []
        for seed in cond["gen_seeds"]:
            if cond["type"] == "team":
                soc = soc_by_protocol[cond.get("protocol", "v4")]
                agents = [Agent(a, a.split(".")[-2] if a.count(".") >= 2 else a.split(".")[-1], "", "")
                          for a in cond["agents"]]
                votes, ok = [], True
                for agent in agents:
                    others = [o for o in agents if o.agent_id != agent.agent_id]
                    ans, conf, _ = scorer.correct_of(soc.r1_key(agent, others, item_id, seed))
                    if ans is None and scorer.store.get(soc.r1_key(agent, others, item_id, seed)) is None:
                        ok = False
                        break
                    votes.append((ans, conf, agent.role))
                if not ok:
                    continue
                final = aggregate(votes)
                vals.append(float(is_correct(final, item.gold, item.answer_type)))
            elif cond["type"] in ("sc", "single"):
                keys = [Society.sc_key(Agent(cond["agent"], "base", "", ""), item_id, seed, k)
                        for k in range(cond["k"])]
                scored = [scorer.correct_of(k) for k in keys]
                if any(s[0] is None and scorer.store.get(k) is None for s, k in zip(scored, keys)):
                    continue
                if cond["type"] == "single":
                    vals.append(float(np.mean([bool(s[2]) for s in scored])))
                else:
                    final = aggregate([(s[0], s[1], "x") for s in scored])
                    vals.append(float(is_correct(final, item.gold, item.answer_type)))
            elif cond["type"] == "solo":
                ans, conf, ok_ = scorer.correct_of(Society.r0_key(Agent(cond["agent"], "", "", ""), item_id, seed))
                if ans is None and scorer.store.get(Society.r0_key(Agent(cond["agent"], "", "", ""), item_id, seed)) is None:
                    continue
                vals.append(float(bool(ok_)))
        if vals:
            out[item_id] = float(np.mean(vals))
    return out


def macro(values: dict, bench_of: dict) -> dict:
    by = collections.defaultdict(list)
    for i, v in values.items():
        by[bench_of[i]].append(v)
    res = {b: float(np.mean(by[b])) for b in BENCHES if by[b]}
    res["macro"] = float(np.mean([res[b] for b in BENCHES if b in res]))
    res["micro"] = float(np.mean(list(values.values())))
    res["n"] = len(values)
    return res


def paired_test(a: dict, b: dict, bench_of: dict, n_perm: int = 20000, n_boot: int = 10000, seed: int = 0) -> dict:
    common = sorted(set(a) & set(b))
    diffs = {bn: np.array([a[i] - b[i] for i in common if bench_of[i] == bn]) for bn in BENCHES}
    diffs = {bn: d for bn, d in diffs.items() if len(d)}
    observed = float(np.mean([d.mean() for d in diffs.values()]))
    rng = np.random.default_rng(seed)
    perm = np.zeros(n_perm)
    for bn, d in diffs.items():
        signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
        perm += (signs * d).mean(axis=1)
    perm /= len(diffs)
    p = float((np.sum(np.abs(perm) >= abs(observed) - 1e-12) + 1) / (n_perm + 1))
    boot = np.zeros(n_boot)
    for bn, d in diffs.items():
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boot += d[idx].mean(axis=1)
    boot /= len(diffs)
    return {"n": len(common), "diff_macro": observed, "p": p,
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "ci90": [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))],
            "per_bench": {bn: float(d.mean()) for bn, d in diffs.items()}}


def holm(pvals: dict) -> dict:
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[key]))
        adjusted[key] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--items", default=str(ROOT / "data/v4/items/test.jsonl"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    items = index_items([args.items])
    ids = [i.item_id for i in load_items(args.items)]
    bench_of = {i: items[i].bench for i in ids}
    store = CallStore("/tmp/evo4_stats_local", args.store)
    scorer = Scorer(store, items)
    socs = {p: Society(None, store, items, None, protocol=p) for p in ("v4", "v4c", "july", "v4t", "v4ct")}
    values = {name: per_item(cond, ids, scorer, socs) for name, cond in spec["conditions"].items()}
    result = {"accuracy": {name: macro(v, bench_of) for name, v in values.items()}, "comparisons": {}}
    for a, b in spec.get("comparisons", []):
        result["comparisons"][f"{a} vs {b}"] = paired_test(values[a], values[b], bench_of)
    if spec.get("holm"):
        pv = {f"{a} vs {b}": result["comparisons"][f"{a} vs {b}"]["p"] for a, b in spec["holm"]}
        result["holm_adjusted"] = holm(pv)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
