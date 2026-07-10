"""進化前後の LoRA ΔW の cosine 類似度分析（査読対応）。

「ノルム比 1.00 ＋ 性能回復」だけでは「方向が回転して損傷成分を打ち消した」とは
言えない、という査読指摘に対し、gen0 と進化後最終チームの各アダプタについて
モジュールごとの ΔW = B@A を構成し、Frobenius 内積での cosine 類似度と
ノルム比を実測する。

- cos ≈ 1  : ほぼ同一方向（実質的な変化なし）
- cos が中程度: 方向が部分的に回転（ブレンド・変異による混合と整合）
- ノルム比 ≈ 1 と併せて「大きさ不変・方向変化」の定量的根拠とする。

役割対応: critic=persona_a / pragmatist=persona_b / explorer=persona_c
最終チーム: gen_04/gen4_critic_child, gen_05/gen5_pragmatist_child,
            gen_05/gen5_explorer_child （run001 進化ログの final_team）

実行: uv run python scripts/analysis/delta_w_similarity.py
"""
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).parents[2]
AD = ROOT / "artifacts_local/adapters"

PAIRS = [
    ("critic", AD / "run001_gen0/persona_a", AD / "run001_evolution/gen_04/gen4_critic_child"),
    ("pragmatist", AD / "run001_gen0/persona_b", AD / "run001_evolution/gen_05/gen5_pragmatist_child"),
    ("explorer", AD / "run001_gen0/persona_c", AD / "run001_evolution/gen_05/gen5_explorer_child"),
]


def load_lora(path: Path) -> dict:
    """{module_prefix: (A, B)} を返す。bf16 は torch 経由で float32 化。"""
    tensors = {}
    with safe_open(path / "adapter_model.safetensors", framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key).float().numpy()
    mods = {}
    for key, t in tensors.items():
        if key.endswith("lora_A.weight"):
            mods.setdefault(key[: -len(".lora_A.weight")], {})["A"] = t
        elif key.endswith("lora_B.weight"):
            mods.setdefault(key[: -len(".lora_B.weight")], {})["B"] = t
    return {m: (v["A"], v["B"]) for m, v in mods.items() if "A" in v and "B" in v}


EXTRA_PAIRS = [
    # 進化演算子1回あたりの実効摂動の定量化
    ("変異1回 (persona_b vs gen0_mutant)",
     AD / "run001_gen0/persona_b", AD / "run001_evolution/gen_00/gen0_pragmatist_mutant"),
    ("交叉1回 (persona_b vs gen1_child)",
     AD / "run001_gen0/persona_b", AD / "run001_evolution/gen_01/gen1_pragmatist_child"),
    ("gen1→gen5 (4世代分)",
     AD / "run001_evolution/gen_01/gen1_pragmatist_child",
     AD / "run001_evolution/gen_05/gen5_pragmatist_child"),
]


def main():
    summary = {}
    for role, p0, p1 in PAIRS:
        m0, m1 = load_lora(p0), load_lora(p1)
        keys = sorted(m0.keys() & m1.keys())
        assert keys, f"{role}: 共有モジュールなし"
        coss, n0s, n1s = [], [], []
        dot_total = norm0_sq = norm1_sq = 0.0
        for k in keys:
            A0, B0 = m0[k]
            A1, B1 = m1[k]
            W0 = B0 @ A0
            W1 = B1 @ A1
            dot = float((W0 * W1).sum())
            n0 = float(np.linalg.norm(W0))
            n1 = float(np.linalg.norm(W1))
            coss.append(dot / (n0 * n1))
            n0s.append(n0)
            n1s.append(n1)
            dot_total += dot
            norm0_sq += n0 * n0
            norm1_sq += n1 * n1
        coss = np.array(coss)
        cos_global = dot_total / (np.sqrt(norm0_sq) * np.sqrt(norm1_sq))
        norm_ratio = np.sqrt(norm1_sq) / np.sqrt(norm0_sq)
        summary[role] = {
            "modules": len(keys),
            "cos_global": float(cos_global),
            "cos_module_mean": float(coss.mean()),
            "cos_module_min": float(coss.min()),
            "cos_module_max": float(coss.max()),
            "norm_ratio_total": float(norm_ratio),
        }
        print(f"[{role}] modules={len(keys)}")
        print(f"  全体cos（連結ΔW）      = {cos_global:.4f}")
        print(f"  モジュール別cos 平均/最小/最大 = "
              f"{coss.mean():.4f} / {coss.min():.4f} / {coss.max():.4f}")
        print(f"  ノルム比 (進化後/gen0) = {norm_ratio:.4f}")
    print("\n--- 進化演算子の実効摂動 ---")
    for label, p0, p1 in EXTRA_PAIRS:
        m0, m1 = load_lora(p0), load_lora(p1)
        keys = sorted(m0.keys() & m1.keys())
        dot = n0sq = n1sq = 0.0
        for k in keys:
            W0 = m0[k][1] @ m0[k][0]
            W1 = m1[k][1] @ m1[k][0]
            dot += float((W0 * W1).sum())
            n0sq += float((W0 * W0).sum())
            n1sq += float((W1 * W1).sum())
        cos = dot / np.sqrt(n0sq * n1sq)
        ratio = np.sqrt(n1sq / n0sq)
        print(f"  {label:<38} cos={cos:.6f} ノルム比={ratio:.6f}")
        summary[label] = {"cos": float(cos), "norm_ratio": float(ratio)}
    out = ROOT / "results/analysis_delta_w_similarity.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
