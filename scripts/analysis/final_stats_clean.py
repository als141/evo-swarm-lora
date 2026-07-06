"""新環境（同一評価イメージ）に統一した最終統計。

- c7（新チーム）と c2（SC@9）は6シード（s1-6）、c1（ベース）と c5（旧チーム）は3シード（s1-3）。
- 全ファイルが 2026-07-05 の同一 eval イメージで測定されたもののみを使用する。
- 主検定: c7 vs c2 の6シード6,000問プール McNemar 正確検定。
- 副検定: 3シード3,000問プールでの c7vsc5 / c7vsc1 / c5vsc1 / c5vsc2 / c1vsc2、Holm補正。
"""
import json
import sys
from pathlib import Path

BASE = Path("/tmp/claude-1000/-home-als0028-study-research-evo-swarm-lora/132de56c-1ddc-424a-86ce-ae486acc7b2e/scratchpad")
sys.path.insert(0, "/home/als0028/study/research/evo-swarm-lora")
from src.evalx.stats import mcnemar_exact  # noqa: E402

BENCHES = ["mmlu_pro", "math500", "supergpqa"]


def path_for(cond, bench, seed):
    if cond == "c7":
        if seed == 1:
            return BASE / f"g3/g3_{bench}.json"
        if seed in (2, 3):
            return BASE / f"final_c7/c7_run002_team_{bench}_s{seed}.json"
        return BASE / f"robust/c7_run002_team_{bench}_s{seed}.json"
    if cond == "c2":
        if bench == "mmlu_pro" and seed == 1:
            return BASE / "robust/c2_sc9_mmlu_pro_s1_recheck.json"
        if seed <= 3:
            return BASE / f"remeasure/r_c2_sc9_{bench}_s{seed}.json"
        return BASE / f"robust/c2_sc9_{bench}_s{seed}.json"
    if cond == "c1":
        return BASE / f"remeasure/r_c1_base_solo_{bench}_s{seed}.json"
    if cond == "c5":
        return BASE / f"remeasure/r_c5_evolved_team_{bench}_s{seed}.json"
    raise ValueError(cond)


def correct_map(cond, bench, seed):
    with path_for(cond, bench, seed).open() as f:
        d = json.load(f)
    if d.get("team"):
        pi = d["team"]["per_item"]
    elif d.get("sc"):
        pi = next(iter(d["sc"].values()))["per_item"]
    else:
        pi = next(iter(d["solo"].values()))["per_item"]
    return {k: v["correct"] for k, v in pi.items()}


SEEDS = {"c7": [1, 2, 3, 4, 5, 6], "c2": [1, 2, 3, 4, 5, 6], "c1": [1, 2, 3], "c5": [1, 2, 3]}
NAMES = {"c1": "ベース単体", "c2": "SC@9", "c5": "旧チーム(v1構成)", "c7": "新チーム(処方後)"}

print("=" * 76)
print("1. 新環境統一の最終精度表（c1/c5=3シード、c2/c7=6シード平均）")
print("=" * 76)
for cond in ["c1", "c2", "c5", "c7"]:
    cells = []
    for bench in BENCHES:
        accs = []
        for seed in SEEDS[cond]:
            cm = correct_map(cond, bench, seed)
            accs.append(sum(cm.values()) / len(cm))
        cells.append(f"{bench}={sum(accs)/len(accs):.3f}")
    print(f"{cond} {NAMES[cond]:<18} " + " ".join(cells) + f"  ({len(SEEDS[cond])}シード)")


def pooled(cond_a, cond_b, seeds, benches=BENCHES):
    a_only = b_only = n = 0
    pairs = []
    for bench in benches:
        for seed in seeds:
            ca = correct_map(cond_a, bench, seed)
            cb = correct_map(cond_b, bench, seed)
            keys = ca.keys() & cb.keys()
            n += len(keys)
            for k in keys:
                if ca[k] and not cb[k]:
                    a_only += 1
                elif cb[k] and not ca[k]:
                    b_only += 1
                pairs.append((ca[k], cb[k]))
    p = mcnemar_exact(pairs)["p_value"]
    return (a_only - b_only) / n * 100, a_only, b_only, n, p


print()
print("=" * 76)
print("2. 主検定: c7 vs SC@9（6シード6,000問プール、ベンチ別内訳付き）")
print("=" * 76)
d, a, b, n, p = pooled("c7", "c2", [1, 2, 3, 4, 5, 6])
print(f"プール: Δ={d:+.2f}pt (c7勝ち{a}/SC@9勝ち{b}, n={n}) p={p:.6f}")
for bench in BENCHES:
    d, a, b, n, p = pooled("c7", "c2", [1, 2, 3, 4, 5, 6], [bench])
    print(f"  {bench:<10}: Δ={d:+.2f}pt ({a}/{b}, n={n}) p={p:.5f}")

print()
print("=" * 76)
print("3. 副検定（3シード3,000問プール、Holm補正）")
print("=" * 76)
SUB = [("c7", "c5", "新チーム vs 旧チーム（処方の効果）"),
       ("c7", "c1", "新チーム vs ベース（チーム化+処方）"),
       ("c5", "c1", "旧チーム vs ベース（v1チーム化）"),
       ("c5", "c2", "旧チーム vs SC@9"),
       ("c1", "c2", "ベース vs SC@9")]
results = []
for ca, cb, label in SUB:
    d, a, b, n, p = pooled(ca, cb, [1, 2, 3])
    results.append((label, d, a, b, n, p))
    print(f"{label:<34} Δ={d:+.2f}pt ({a}/{b}) p={p:.5f}")
    for bench in BENCHES:
        db, ab, bb, nb, pb = pooled(ca, cb, [1, 2, 3], [bench])
        print(f"    {bench:<10}: Δ={db:+.2f}pt p={pb:.4f}")

print("\n--- Holm補正（主検定+副検定5 = 6比較） ---")
d, a, b, n, p_main = pooled("c7", "c2", [1, 2, 3, 4, 5, 6])
all_tests = [("★c7 vs SC@9(主検定)", p_main)] + [(r[0], r[5]) for r in results]
m = len(all_tests)
for rank, (label, p) in enumerate(sorted(all_tests, key=lambda x: x[1])):
    adj = min(1.0, p * (m - rank))
    print(f"  {label:<36} p={p:.5f} p_adj={adj:.5f} {'✅' if adj < 0.05 else '❌'}")
