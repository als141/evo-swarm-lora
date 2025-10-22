from __future__ import annotations

import json
import os
import random
from typing import Dict

import torch
from safetensors.torch import load_file, save_file


def _load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_config(path: str, config: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)


def alpha_blend_lora(parent_a_dir: str, parent_b_dir: str, out_dir: str, alpha: float = 0.5) -> None:
    os.makedirs(out_dir, exist_ok=True)
    parent_a = load_file(os.path.join(parent_a_dir, "adapter_model.safetensors"))
    parent_b = load_file(os.path.join(parent_b_dir, "adapter_model.safetensors"))
    blended = {}
    for key, value in parent_a.items():
        if key in parent_b:
            blended[key] = ((1 - alpha) * value) + (alpha * parent_b[key])
    save_file(blended, os.path.join(out_dir, "adapter_model.safetensors"))

    config = _load_config(os.path.join(parent_a_dir, "adapter_config.json"))
    _write_config(os.path.join(out_dir, "adapter_config.json"), config)


def mutate_lora(in_dir: str, out_dir: str, ratio: float = 0.05, std: float = 0.01) -> None:
    os.makedirs(out_dir, exist_ok=True)
    weights = load_file(os.path.join(in_dir, "adapter_model.safetensors"))
    mutated = {}
    for key, tensor in weights.items():
        if random.random() < ratio:
            mutated[key] = tensor + torch.randn_like(tensor) * std
        else:
            mutated[key] = tensor
    save_file(mutated, os.path.join(out_dir, "adapter_model.safetensors"))

    config = _load_config(os.path.join(in_dir, "adapter_config.json"))
    _write_config(os.path.join(out_dir, "adapter_config.json"), config)
