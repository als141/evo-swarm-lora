"""v3最終評価の全確定統計（63エントリ）+ パイロットA/B判定。

- 7条件×3ベンチ×3シードの精度表
- 主要対比較: McNemar exact + Holm補正（ベンチ別プール=シード結合、全体プール3000問）
- パイロットA/B: conditional vs standard（MMLU-Pro 300問 seed555）
"""
import json
import sys
from pathlib import Path

SP = Path(__file__).parent
sys.path.insert(0, "/home/als0028/study/research/evo-swarm-lora")
from src.evalx.stats import mcnemar_exact  # noqa: E402

BENCHES = ["mmlu_pro", "math500", "supergpqa"]
CONDS = ["c1_base_solo", "c2_sc9", "c3_base_team", "c3p_prompt_persona_team",
         "c4_gen0_team", "c5_evolved_team", "c7_run002_team"]
SEEDS = [1, 2, 3]


def correct_map(d):
    """条件JSONから {item_id: bool} を返す。c6(solo複数体)は最良1体でなく critic を採用しない
    ——集計は各体別に扱うため、ここでは solo は代表としてすべて返す。"""
    if d.get("team") and "per_item" in d["team"]:
        return {k: v["correct"] for k, v in d["team"]["per_item"].items()}
    if d.get("sc"):
        sec = next(iter(d["sc"].values()))
        return {k: v["correct"] for k, v in sec["per_item"].items()}
    if d.get("solo"):
        # 単体条件(c1)は1体のみ。c6は3体 →呼び出し側で agent 指定
        if len(d["solo"]) == 1:
            sec = next(iter(d["solo"].values()))
            return {k: v["correct"] for k, v in sec["per_item"].items()}
        return {a: {k: v["correct"] for k, v in sec["per_item"].items()} for a, sec in d["solo"].items()}
    raise ValueError("unknown structure")


C7_G3 = Path("/tmp/claude-1000/-home-als0028-study-research-evo-swarm-lora/132de56c-1ddc-424a-86ce-ae486acc7b2e/scratchpad/g3")
C7_FIN = Path("/tmp/claude-1000/-home-als0028-study-research-evo-swarm-lora/132de56c-1ddc-424a-86ce-ae486acc7b2e/scratchpad/final_c7")


def load(cond, bench, seed):
    if cond == "c7_run002_team":
        if seed == 1:
            path = C7_G3 / f"g3_{bench}.json"
        else:
            path = C7_FIN / f"c7_run002_team_{bench}_s{seed}.json"
    else:
        path = SP / f"{cond}_{bench}_s{seed}.json"
    with path.open() as f:
        return json.load(f)


# ---------- 1. 精度表 ----------
print("=" * 78)
print("1. 最終確定 精度表（500/200/300問×3シード）")
print("=" * 78)
acc = {}
for cond in CONDS:
    row = {}
    for bench in BENCHES:
        per_seed = []
        for seed in SEEDS:
            d = load(cond, bench, seed)
            cm = correct_map(d)
            per_seed.append(sum(cm.values()) / len(cm))
        row[bench] = sum(per_seed) / len(per_seed)
        acc[(cond, bench)] = per_seed
    print(f"{cond:<28} " + " ".join(f"{bench}={row[bench]:.3f}" for bench in BENCHES))

# ---------- 2. ペア検定（シード結合プール） ----------
print()
print("=" * 78)
print("2. 主要対比較（McNemar exact、シード結合プール。Holmは主要6比較に適用）")
print("=" * 78)


def pooled_pair(cond_a, cond_b, benches):
    """シード×ベンチを結合した対応ペアで McNemar。(a勝ち, b勝ち, n, p) を返す"""
    a_only = b_only = n = 0
    pairs = []
    for bench in benches:
        for seed in SEEDS:
            ca = correct_map(load(cond_a, bench, seed))
            cb = correct_map(load(cond_b, bench, seed))
            if isinstance(next(iter(ca.values())), dict):
                raise ValueError("solo multi-agent in pair test")
            keys = ca.keys() & cb.keys()
            n += len(keys)
            for k in keys:
                if ca[k] and not cb[k]:
                    a_only += 1
                elif cb[k] and not ca[k]:
                    b_only += 1
                pairs.append((ca[k], cb[k]))
    p = mcnemar_exact(pairs)["p_value"]
    return a_only, b_only, n, p


PAIRS = [
    ("c7_run002_team", "c2_sc9", "★c7 vs SC@9（主検定）"),
    ("c7_run002_team", "c5_evolved_team", "c7 vs 旧チームc5"),
    ("c7_run002_team", "c1_base_solo", "c7 vs ベース"),
    ("c7_run002_team", "c3_base_team", "c7 vs 素の議論"),
    ("c7_run002_team", "c3p_prompt_persona_team", "c7 vs プロンプトペルソナ"),
    ("c5_evolved_team", "c2_sc9", "（参考）c5 vs SC@9"),
]

results = []
for bench in BENCHES + ["ALL"]:
    benches = BENCHES if bench == "ALL" else [bench]
    print(f"\n--- {bench} ---")
    for cond_a, cond_b, label in PAIRS:
        a_only, b_only, n, p = pooled_pair(cond_a, cond_b, benches)
        diff = (a_only - b_only) / n * 100
        print(f"  {label:<34} Δ={diff:+.1f}pt (a+{a_only}/b+{b_only}, n={n}) p={p:.5f}")
        if bench == "ALL":
            results.append((label, p))

# Holm補正（ALLプールの6比較）
print("\n--- Holm補正（ALLプール6比較） ---")
sorted_res = sorted(results, key=lambda x: x[1])
m = len(sorted_res)
for rank, (label, p) in enumerate(sorted_res):
    adj = min(1.0, p * (m - rank))
    sig = "✅" if adj < 0.05 else "❌"
    print(f"  {label:<34} p={p:.5f} p_adj={adj:.5f} {sig}")

