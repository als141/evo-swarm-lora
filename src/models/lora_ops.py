"""LoRA アダプタの交叉・突然変異オペレーション。

交叉は 2 方式:
- delta_blend_lora (主方式): 実効更新 ΔW = B·A を層ごとに計算してから補間し、
  ランダム化 SVD で rank r に再分解する。A/B 行列を別々に補間すると
  ((1-α)B1+αB2)((1-α)A1+αA2) に交差項 B1A2, B2A1 が混入する問題
  (KnOTS, arXiv:2410.19735 等) を回避する。
- alpha_blend_lora (naive・アブレーション用): A/B 行列を別々に線形補間する従来方式。

突然変異はガウスノイズの付加（seed 指定で再現可能）。
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import torch
from safetensors.torch import load_file, save_file


def _load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_config(path: str, config: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)


def _copy_config(src_dir: str, out_dir: str) -> None:
    config = _load_config(os.path.join(src_dir, "adapter_config.json"))
    _write_config(os.path.join(out_dir, "adapter_config.json"), config)


def _pair_lora_keys(weights: Dict[str, torch.Tensor]) -> Dict[str, Tuple[str, str]]:
    """モジュールプレフィックス -> (lora_A キー, lora_B キー) の対応を作る。"""
    pairs: Dict[str, Tuple[str, str]] = {}
    for key in weights:
        if ".lora_A." in key:
            prefix = key.split(".lora_A.")[0]
            b_key = key.replace(".lora_A.", ".lora_B.")
            if b_key in weights:
                pairs[prefix] = (key, b_key)
    return pairs


def alpha_blend_lora(parent_a_dir: str, parent_b_dir: str, out_dir: str, alpha: float = 0.5) -> None:
    """naive 方式: A/B 行列を別々に線形補間する（アブレーション用に保持）。"""
    os.makedirs(out_dir, exist_ok=True)
    parent_a = load_file(os.path.join(parent_a_dir, "adapter_model.safetensors"))
    parent_b = load_file(os.path.join(parent_b_dir, "adapter_model.safetensors"))
    blended = {}
    for key, value in parent_a.items():
        if key in parent_b:
            blended[key] = ((1 - alpha) * value) + (alpha * parent_b[key])
    save_file(blended, os.path.join(out_dir, "adapter_model.safetensors"))
    _copy_config(parent_a_dir, out_dir)


def delta_blend_lora(
    parent_a_dir: str,
    parent_b_dir: str,
    out_dir: str,
    alpha: float = 0.5,
    svd_niter: int = 4,
    svd_oversample: int = 8,
) -> None:
    """主方式: ΔW 空間で補間し、ランダム化 SVD で元の rank に再分解する。

    親同士の rank・対象モジュールは一致している必要がある。
    """
    os.makedirs(out_dir, exist_ok=True)
    parent_a = load_file(os.path.join(parent_a_dir, "adapter_model.safetensors"))
    parent_b = load_file(os.path.join(parent_b_dir, "adapter_model.safetensors"))

    pairs_a = _pair_lora_keys(parent_a)
    pairs_b = _pair_lora_keys(parent_b)
    if set(pairs_a) != set(pairs_b):
        raise ValueError("Parent adapters target different modules; cannot blend.")

    blended: Dict[str, torch.Tensor] = {}
    for prefix, (a_key, b_key) in pairs_a.items():
        a1 = parent_a[a_key].to(torch.float32)  # [r, in]
        b1 = parent_a[b_key].to(torch.float32)  # [out, r]
        a2 = parent_b[a_key].to(torch.float32)
        b2 = parent_b[b_key].to(torch.float32)
        if a1.shape != a2.shape or b1.shape != b2.shape:
            raise ValueError(f"Rank mismatch at {prefix}: {a1.shape} vs {a2.shape}")

        rank = a1.shape[0]
        delta = (1 - alpha) * (b1 @ a1) + alpha * (b2 @ a2)  # [out, in]

        q = min(rank + svd_oversample, min(delta.shape))
        u, s, v = torch.svd_lowrank(delta, q=q, niter=svd_niter)
        u, s, v = u[:, :rank], s[:rank], v[:, :rank]
        sqrt_s = torch.sqrt(torch.clamp(s, min=0.0))
        new_b = u * sqrt_s.unsqueeze(0)          # [out, r]
        new_a = (v * sqrt_s.unsqueeze(0)).T      # [r, in]

        blended[a_key] = new_a.to(parent_a[a_key].dtype).contiguous()
        blended[b_key] = new_b.to(parent_a[b_key].dtype).contiguous()

    # LoRA 以外のキー（存在すれば）は親 A から線形補間 or そのまま継承
    for key, value in parent_a.items():
        if key not in blended:
            blended[key] = (
                ((1 - alpha) * value + alpha * parent_b[key]) if key in parent_b else value
            )

    save_file(blended, os.path.join(out_dir, "adapter_model.safetensors"))
    _copy_config(parent_a_dir, out_dir)


def mutate_lora(
    in_dir: str,
    out_dir: str,
    ratio: float = 0.05,
    std: float = 0.01,
    seed: int | None = None,
) -> None:
    """テンソル単位の確率 ratio でガウスノイズ（相対スケール std）を付加する。

    ノイズはテンソルの実スケールに合わせるため、std はテンソルの標準偏差に
    対する相対値として扱う。seed 指定で再現可能。
    """
    os.makedirs(out_dir, exist_ok=True)
    weights = load_file(os.path.join(in_dir, "adapter_model.safetensors"))
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    mutated = {}
    for key, tensor in sorted(weights.items()):
        gate = torch.rand(1, generator=generator).item() if generator else torch.rand(1).item()
        if gate < ratio:
            scale = tensor.float().std().item() or 1.0
            noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32) * std * scale
            mutated[key] = (tensor.to(torch.float32) + noise).to(tensor.dtype)
        else:
            mutated[key] = tensor
    save_file(mutated, os.path.join(out_dir, "adapter_model.safetensors"))
    _copy_config(in_dir, out_dir)
