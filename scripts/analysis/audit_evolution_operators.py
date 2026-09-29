"""進化演算子の実効摂動と ΔW 空間交叉の SVD 切り捨て損失の監査（float64 で再計算）。

既存の delta_w_similarity.py は float32 の素朴な総和で cos を計算しており、
cos>1（1.0004 等）が出る＝桁落ちの精度が ~1e-4 で、報告値 0.9997 と 1.0000 の差は
数値誤差と同程度である。本スクリプトは float64 で
  (1) 相対変化 ||ΔW1-ΔW0||_F / ||ΔW0||_F と cos（全モジュール連結）
  (2) 異なる親（別ペルソナ）を α=0.5 でブレンドしたときの rank-r 切り捨てで失うエネルギー比
      ||ΔW' - trunc_r(ΔW')||_F^2 / ||ΔW'||_F^2
  (3) アダプタの保存 dtype
を出す。GPU 不要（CPU・数分）。

実行: uv run python scripts/analysis/audit_evolution_operators.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).parents[2]
AD = ROOT / "artifacts_local/adapters"


def load(path: Path):
    mods, dtypes = {}, set()
    with safe_open(path / "adapter_model.safetensors", framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            dtypes.add(str(t.dtype))
            if key.endswith("lora_A.weight"):
                mods.setdefault(key[: -len(".lora_A.weight")], {})["A"] = t.double()
            elif key.endswith("lora_B.weight"):
                mods.setdefault(key[: -len(".lora_B.weight")], {})["B"] = t.double()
    return {m: (v["A"], v["B"]) for m, v in mods.items()}, dtypes


def compare(p0: Path, p1: Path) -> dict:
    m0, _ = load(p0)
    m1, _ = load(p1)
    dot = n0 = n1 = nd = 0.0
    for k in sorted(m0.keys() & m1.keys()):
        w0 = m0[k][1] @ m0[k][0]
        w1 = m1[k][1] @ m1[k][0]
        dot += float((w0 * w1).sum())
        n0 += float((w0 * w0).sum())
        n1 += float((w1 * w1).sum())
        nd += float(((w1 - w0) ** 2).sum())
    return {"cos": dot / np.sqrt(n0 * n1), "rel_change": np.sqrt(nd / n0), "norm_ratio": np.sqrt(n1 / n0)}


def truncation_loss(p_a: Path, p_b: Path, alpha: float = 0.5, max_modules: int | None = None) -> dict:
    """別個体同士を ΔW 空間でブレンドし、元 rank r に切り詰めたときに失うエネルギー比。"""
    ma, _ = load(p_a)
    mb, _ = load(p_b)
    lost = total = 0.0
    keys = sorted(ma.keys() & mb.keys())
    if max_modules:
        keys = keys[:: max(1, len(keys) // max_modules)]
    for k in keys:
        a1, b1 = ma[k]
        a2, b2 = mb[k]
        r = a1.shape[0]
        w = (1 - alpha) * (b1 @ a1) + alpha * (b2 @ a2)
        # 厳密 SVD（rank ≤ 2r なので小さい因子行列から計算: w = [b1 b2] diag [a1; a2]）
        left = torch.cat([(1 - alpha) * b1, alpha * b2], dim=1)  # out x 2r
        right = torch.cat([a1, a2], dim=0)  # 2r x in
        q_l, r_l = torch.linalg.qr(left)
        q_r, r_r = torch.linalg.qr(right.T)
        s = torch.linalg.svdvals(r_l @ r_r.T)
        energy = float((s**2).sum())
        lost += float((s[r:] ** 2).sum())
        total += energy
        assert abs(energy - float((w * w).sum())) / max(energy, 1e-30) < 1e-6
    return {"modules": len(keys), "energy_lost_frac": lost / total}


def main() -> None:
    out = {}
    _, dt = load(AD / "run001_gen0/persona_a")
    out["stored_dtype_gen0"] = sorted(dt)
    _, dt = load(AD / "run001_evolution/gen_01/gen1_pragmatist_child")
    out["stored_dtype_child"] = sorted(dt)
    pairs = {
        "mutation (gen0 persona_b -> gen0_pragmatist_mutant)": (
            AD / "run001_gen0/persona_b", AD / "run001_evolution/gen_00/gen0_pragmatist_mutant"),
        "crossover+mutation (gen0 persona_b -> gen1_pragmatist_child)": (
            AD / "run001_gen0/persona_b", AD / "run001_evolution/gen_01/gen1_pragmatist_child"),
        "6 generations (gen0 persona_b -> final gen5_pragmatist_child)": (
            AD / "run001_gen0/persona_b", AD / "run001_evolution/gen_05/gen5_pragmatist_child"),
        "different persona (gen0 persona_a vs persona_b)": (
            AD / "run001_gen0/persona_a", AD / "run001_gen0/persona_b"),
        "different training recipe (run002 persona_a vs persona_b)": (
            AD / "run002_replay/persona_a", AD / "run002_replay/persona_b"),
    }
    for label, (p0, p1) in pairs.items():
        out[label] = compare(p0, p1)
        print(f"{label:62s} cos={out[label]['cos']:.6f} rel_change={out[label]['rel_change']:.4f} "
              f"norm_ratio={out[label]['norm_ratio']:.4f}")
    for label, (pa, pb) in {
        "blend persona_a+persona_b (gen0, r=32)": (AD / "run001_gen0/persona_a", AD / "run001_gen0/persona_b"),
        "blend persona_b+its mutant (near-clone, r=32)": (
            AD / "run001_gen0/persona_b", AD / "run001_evolution/gen_00/gen0_pragmatist_mutant"),
        "blend run002 persona_a+persona_b (r=16)": (AD / "run002_replay/persona_a", AD / "run002_replay/persona_b"),
    }.items():
        out[label] = truncation_loss(pa, pb, max_modules=None)
        print(f"{label:62s} modules={out[label]['modules']} energy_lost_by_rank_r_truncation="
              f"{out[label]['energy_lost_frac']*100:.2f}%")
    (ROOT / "results/analysis_evolution_operators.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
