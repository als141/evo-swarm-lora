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
         "c4_gen0_team", "c5_evolved_team", "c6_evolved_solo"]
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


def load(cond, bench, seed):
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
            if cond == "c6_evolved_solo":
                # 3体の最良を報告（従来の報告と整合）
                best = max(sum(m.values()) / len(m) for m in cm.values())
                per_seed.append(best)
            else:
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
    ("c5_evolved_team", "c2_sc9", "c5 vs SC@9"),
    ("c5_evolved_team", "c1_base_solo", "c5 vs ベース"),
    ("c5_evolved_team", "c4_gen0_team", "c5 vs gen0（進化寄与）"),
    ("c4_gen0_team", "c1_base_solo", "gen0 vs ベース"),
    ("c3_base_team", "c1_base_solo", "素の議論 vs ベース"),
    ("c3p_prompt_persona_team", "c1_base_solo", "プロンプトペルソナ vs ベース"),
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

# ---------- 3. パイロットA/B ----------
print()
print("=" * 78)
print("3. パイロットA/B: conditional vs standard（MMLU-Pro 300問 seed555）")
print("=" * 78)
PD = SP.parent / "pilot_debate_style"
with (PD / "pilot_std_mmlu_pro_s555.json").open() as f:
    std = json.load(f)
with (PD / "pilot_cond_mmlu_pro_s555.json").open() as f:
    cond = json.load(f)
sm = correct_map(std)
cm = correct_map(cond)
keys = sm.keys() & cm.keys()
s_acc = sum(sm[k] for k in keys) / len(keys)
c_acc = sum(cm[k] for k in keys) / len(keys)
c_only = sum(1 for k in keys if cm[k] and not sm[k])
s_only = sum(1 for k in keys if sm[k] and not cm[k])
p = mcnemar_exact([(cm[k], sm[k]) for k in keys])["p_value"]
print(f"standard: {s_acc:.3f} / conditional: {c_acc:.3f} (n={len(keys)})")
print(f"conditionalのみ正解={c_only} / standardのみ正解={s_only} / McNemar p={p:.4f}")
print(f"判定: {'✅ conditional採用（有意改善）' if p < 0.05 and c_acc > s_acc else '⚠️ 有意差なし→効果量と方向で判断' if c_acc > s_acc else '❌ conditional不採用'}")
