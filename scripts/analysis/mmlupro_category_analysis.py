"""MMLU-Pro のカテゴリ別分析 — 「検証可能性」仮説のベンチマーク内検証（査読対応）。

「議論は検証可能な領域で効き、知識依存領域で逆効果」という解釈は、これまで
ベンチマーク間（MMLU-Pro vs MATH-500 vs SuperGPQA）の対比に基づいていた。
しかし MMLU-Pro 自体が数学・物理から法学・心理まで14分野を含むため、
同一ベンチマーク内のカテゴリ別に議論効果が反転するかを検証する。

- 実験1（旧環境）: 素の議論(c3) − ベース(c1)、3シード
- 実験2（新環境）: 新チーム(c7) − SC@9(c2)、6シード
- 事前グループ（探索的・事後定義であることを明記）:
    定量系（検算可能）: math, physics, chemistry, engineering, computer science
    知識系（検算困難）: law, history, philosophy, psychology, business, health,
                        economics, biology, other
- 各カテゴリ Δ と、2グループの Δ 差を問題IDクラスタ置換検定で評価。

実行: uv run python scripts/analysis/mmlupro_category_analysis.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.final_stats_clustered import correct_map  # noqa: E402

QUANT = {"math", "physics", "chemistry", "engineering", "computer science"}
RNG = np.random.default_rng(20260712)
N_PERM = 20_000


def load_categories() -> dict:
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    return {f"mmlupro-test-{r['question_id']}": r["category"] for r in ds}


def deltas_by_item(cond_a, cond_b, seeds):
    per_item = defaultdict(list)
    for seed in seeds:
        ca = correct_map(cond_a, "mmlu_pro", seed)
        cb = correct_map(cond_b, "mmlu_pro", seed)
        for k in ca.keys() & cb.keys():
            per_item[k].append(int(ca[k]) - int(cb[k]))
    return {k: float(np.mean(v)) for k, v in per_item.items()}


def sign_flip_p(deltas: np.ndarray) -> float:
    obs = abs(deltas.mean())
    signs = RNG.choice([-1.0, 1.0], size=(N_PERM, len(deltas)))
    null = np.abs((signs * deltas).mean(axis=1))
    return float((np.sum(null >= obs - 1e-15) + 1) / (N_PERM + 1))


def group_diff_p(dq: np.ndarray, dk: np.ndarray) -> float:
    """グループ間のΔ差: ラベル並べ替え置換検定。"""
    obs = abs(dq.mean() - dk.mean())
    allv = np.concatenate([dq, dk])
    nq = len(dq)
    cnt = 0
    for _ in range(N_PERM):
        RNG.shuffle(allv)
        if abs(allv[:nq].mean() - allv[nq:].mean()) >= obs - 1e-15:
            cnt += 1
    return (cnt + 1) / (N_PERM + 1)


def report(title, cond_a, cond_b, seeds, cats):
    print("=" * 86)
    print(title)
    print("=" * 86)
    d = deltas_by_item(cond_a, cond_b, seeds)
    by_cat = defaultdict(list)
    for k, v in d.items():
        by_cat[cats.get(k, "?")].append(v)
    rows = sorted(by_cat.items(), key=lambda kv: np.mean(kv[1]))
    for cat, vals in rows:
        arr = np.array(vals)
        tag = "定量" if cat in QUANT else "知識"
        print(f"  {cat:<18} [{tag}] Δ={arr.mean()*100:+6.2f}pt (n={len(arr):>3}) "
              f"p={sign_flip_p(arr):.4f}")
    dq = np.array([v for k, v in d.items() if cats.get(k) in QUANT])
    dk = np.array([v for k, v in d.items() if cats.get(k) not in QUANT])
    p_grp = group_diff_p(dq.copy(), dk.copy())
    print(f"  --- 定量系 Δ={dq.mean()*100:+.2f}pt (n={len(dq)}, p={sign_flip_p(dq):.4f}) / "
          f"知識系 Δ={dk.mean()*100:+.2f}pt (n={len(dk)}, p={sign_flip_p(dk):.4f})")
    print(f"  --- グループ差 {abs(dq.mean()-dk.mean())*100:.2f}pt の置換検定 p={p_grp:.4f}")
    print()


def main():
    cats = load_categories()
    report("1) 実験1: 素の議論(c3) − ベース(c1)  [旧環境・3シード]",
           "c3_base_team", "c1_base_solo", [1, 2, 3], cats)
    report("2) 実験2: 新チーム(c7) − SC@9(c2)  [新環境・6シード]",
           "c7", "c2", range(1, 7), cats)
    report("3) 実験2: 新チーム(c7) − ベース(c1)  [新環境・3シード]",
           "c7", "c1", [1, 2, 3], cats)


if __name__ == "__main__":
    main()
