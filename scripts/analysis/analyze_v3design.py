"""設計v3のための法医学的分析。

1. SuperGPQA s1 の分野層別（6条件×discipline/difficulty）
2. ペアflip分析（c3p vs c5 / c1 vs c5 / c3 vs c3p）
3. c6 3体の誤り相関と oracle 上限（3ベンチ共通）
   - 多様性の実測: pairwise agreement / 全員正解 / 全員誤答 / 割れ
   - oracle = 少なくとも1体正解の割合（集約を完璧にした場合の上限）
"""
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

SP = Path(__file__).parent


def load_json(name):
    with (SP / name).open() as f:
        return json.load(f)


def get_per_item(d):
    """条件JSONから {item_id: correct(bool)} と {item_id: predicted} を返す"""
    if d.get("team") and "per_item" in d["team"]:
        pi = d["team"]["per_item"]
        return (
            {k: v["correct"] for k, v in pi.items()},
            {k: v["predicted"] for k, v in pi.items()},
        )
    if d.get("sc"):
        # sc は {model_name: {accuracy, n, per_item}}
        sec = next(iter(d["sc"].values()))
        pi = sec["per_item"]
        return (
            {k: v["correct"] for k, v in pi.items()},
            {k: v["predicted"] for k, v in pi.items()},
        )
    if d.get("solo"):
        # solo は agent 名 -> {..per_item..}（複数体ありうる）
        out = {}
        for agent, sec in d["solo"].items():
            pi = sec["per_item"]
            out[agent] = (
                {k: v["correct"] for k, v in pi.items()},
                {k: v["predicted"] for k, v in pi.items()},
            )
        return out
    raise ValueError("unknown structure")


# ============ 1. SuperGPQA 分野マップ ============
print("=" * 70)
print("1. SuperGPQA s1 分野層別")
print("=" * 70)
from datasets import load_dataset

ds = load_dataset("m-a-p/SuperGPQA", split="train")
meta = {}
for row in ds:
    meta[f"supergpqa-{row['uuid']}"] = {
        "discipline": row.get("discipline"),
        "field": row.get("field"),
        "difficulty": row.get("difficulty"),
    }
print(f"dataset rows: {len(meta)}")

# solo構造の戻りは {agent: (correct_map, pred_map)}
c1 = get_per_item(load_json("c1_base_solo_supergpqa_s1.json"))["base"]
c2 = get_per_item(load_json("c2_sc9_supergpqa_s1.json"))
c3 = get_per_item(load_json("c3_base_team_supergpqa_s1.json"))
c3p = get_per_item(load_json("c3p_prompt_persona_team_supergpqa_s1.json"))
c5 = get_per_item(load_json("c5_evolved_team_supergpqa_s1.json"))
c6all = get_per_item(load_json("c6_evolved_solo_supergpqa_s1.json"))

sg_conds = {"c1_base": c1, "c2_sc9": c2, "c3_debate": c3, "c3p_prompt": c3p, "c5_evolved": c5}

ids = sorted(c1[0].keys())
print(f"items: {len(ids)}")

# 分野層別（discipline 粗粒度 + difficulty）
for axis in ("discipline", "difficulty"):
    groups = defaultdict(list)
    for iid in ids:
        groups[meta.get(iid, {}).get(axis, "?")].append(iid)
    print(f"\n--- {axis} 別精度 (n>=15のみ) ---")
    header = f"{'group':<28} {'n':>4} " + " ".join(f"{c:>10}" for c in sg_conds)
    print(header)
    for g, gids in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(gids) < 15:
            continue
        row = f"{str(g)[:28]:<28} {len(gids):>4} "
        for cname, (cmap, _) in sg_conds.items():
            acc = sum(cmap[i] for i in gids) / len(gids)
            row += f" {acc:>9.3f}"
        print(row)

# ============ 2. ペアflip分析 ============
print("\n" + "=" * 70)
print("2. SuperGPQA s1 ペアflip（行=勝ち, 列=負け, セル=行だけ正解した問題数）")
print("=" * 70)
pairs = [("c3p_prompt", "c5_evolved"), ("c1_base", "c5_evolved"), ("c3_debate", "c3p_prompt"), ("c2_sc9", "c5_evolved")]
for a, b in pairs:
    amap, bmap = sg_conds[a][0], sg_conds[b][0]
    a_only = sum(1 for i in ids if amap[i] and not bmap[i])
    b_only = sum(1 for i in ids if bmap[i] and not amap[i])
    both = sum(1 for i in ids if amap[i] and bmap[i])
    neither = sum(1 for i in ids if not amap[i] and not bmap[i])
    print(f"{a} vs {b}: {a}のみ正解={a_only} / {b}のみ={b_only} / 両方={both} / 両方誤={neither}")

# c3pが正解しc5が誤答の問題の分野内訳
amap, bmap = sg_conds["c3p_prompt"][0], sg_conds["c5_evolved"][0]
flip_ids = [i for i in ids if amap[i] and not bmap[i]]
fields = defaultdict(int)
for i in flip_ids:
    fields[meta.get(i, {}).get("discipline", "?")] += 1
print(f"\nc3p正解・c5誤答 {len(flip_ids)}問の discipline 内訳: {dict(sorted(fields.items(), key=lambda x: -x[1]))}")

# ============ 3. c6 3体の誤り相関と oracle（3ベンチ） ============
print("\n" + "=" * 70)
print("3. 進化後solo 3体の多様性（誤り相関・oracle上限）")
print("=" * 70)

def three_agent_analysis(c6_solo, label, team_acc=None, base_acc=None):
    agents = list(c6_solo.keys())
    cmaps = {a: c6_solo[a][0] for a in agents}
    pmaps = {a: c6_solo[a][1] for a in agents}
    ids3 = sorted(set.intersection(*(set(m.keys()) for m in cmaps.values())))
    n = len(ids3)
    solo_accs = {a: sum(cmaps[a][i] for i in ids3) / n for a in agents}
    # oracle: 少なくとも1体正解
    oracle = sum(1 for i in ids3 if any(cmaps[a][i] for a in agents)) / n
    all_c = sum(1 for i in ids3 if all(cmaps[a][i] for a in agents)) / n
    none_c = sum(1 for i in ids3 if not any(cmaps[a][i] for a in agents)) / n
    split = 1 - all_c - none_c
    # majority-of-solo（議論なしで3体の初期回答を多数決したら）
    import random as _r
    maj = 0
    for i in ids3:
        from collections import Counter
        votes = [pmaps[a][i] for a in agents if pmaps[a][i] is not None]
        if not votes:
            continue
        cnt = Counter(votes)
        top = max(cnt.values())
        winners = sorted(k for k, v in cnt.items() if v == top)
        pick = winners[0] if len(winners) == 1 else _r.Random(0).choice(winners)
        # gold を取得するには correct と predicted から逆算できないので、correct==Trueのpredicted==gold
        # gold map を c1 から取る代わりに: correct な agent の predicted が gold
        gold = None
        for a in agents:
            if cmaps[a][i] and pmaps[a][i] is not None:
                gold = pmaps[a][i]
                break
        if gold is not None and pick == gold:
            maj += 1
    # pairwise agreement（予測一致率）
    agree = {}
    for a, b in combinations(agents, 2):
        agree[f"{a[:4]}-{b[:4]}"] = sum(
            1 for i in ids3 if pmaps[a][i] == pmaps[b][i]
        ) / n
    print(f"\n[{label}] n={n}")
    print(f"  solo精度: " + " ".join(f"{a}={v:.3f}" for a, v in solo_accs.items()))
    print(f"  oracle(≥1体正解)={oracle:.3f} / 全員正解={all_c:.3f} / 全員誤答={none_c:.3f} / 割れ={split:.3f}")
    print(f"  多数決(round0相当)={maj / n:.3f}")
    if team_acc:
        print(f"  実測チーム(議論後)={team_acc:.3f} → 議論の寄与={team_acc - maj / n:+.3f} / oracleとの差={oracle - team_acc:.3f}")
    if base_acc:
        print(f"  ベース単体={base_acc:.3f} / oracle−ベース={oracle - base_acc:+.3f}")
    print(f"  pairwise予測一致率: " + " ".join(f"{k}={v:.3f}" for k, v in agree.items()))
    return oracle

# SuperGPQA
three_agent_analysis(c6all, "SuperGPQA s1", team_acc=0.387, base_acc=0.407)

# MMLU-Pro s1
c6_mmlu = get_per_item(load_json("c6_evolved_solo_mmlu_pro_s1.json"))
three_agent_analysis(c6_mmlu, "MMLU-Pro s1", team_acc=0.714, base_acc=0.660)

# MATH-500 s1
c6_math = get_per_item(load_json("c6_evolved_solo_math500_s1.json"))
three_agent_analysis(c6_math, "MATH-500 s1", team_acc=None, base_acc=None)
