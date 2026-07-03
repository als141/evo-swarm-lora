import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch
from safetensors.torch import load_file, save_file

from src.models.lora_ops import alpha_blend_lora, delta_blend_lora, mutate_lora

RANK = 4
IN_DIM = 32
OUT_DIM = 48


def _make_adapter(directory: Path, seed: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    weights = {}
    for layer in ("model.layers.0.q_proj", "model.layers.0.v_proj"):
        weights[f"base_model.model.{layer}.lora_A.weight"] = torch.randn(
            RANK, IN_DIM, generator=generator
        )
        weights[f"base_model.model.{layer}.lora_B.weight"] = torch.randn(
            OUT_DIM, RANK, generator=generator
        )
    save_file(weights, str(directory / "adapter_model.safetensors"))
    config = {"r": RANK, "lora_alpha": RANK * 2, "peft_type": "LORA"}
    (directory / "adapter_config.json").write_text(json.dumps(config))


def _delta(weights, layer):
    a = weights[f"base_model.model.{layer}.lora_A.weight"]
    b = weights[f"base_model.model.{layer}.lora_B.weight"]
    return b.float() @ a.float()


class TestDeltaBlend:
    def test_shapes_preserved(self, tmp_path):
        _make_adapter(tmp_path / "pa", seed=1)
        _make_adapter(tmp_path / "pb", seed=2)
        delta_blend_lora(str(tmp_path / "pa"), str(tmp_path / "pb"), str(tmp_path / "child"), alpha=0.5)
        child = load_file(str(tmp_path / "child" / "adapter_model.safetensors"))
        assert child["base_model.model.model.layers.0.q_proj.lora_A.weight"].shape == (RANK, IN_DIM)
        assert child["base_model.model.model.layers.0.q_proj.lora_B.weight"].shape == (OUT_DIM, RANK)

    def test_delta_is_interpolation(self, tmp_path):
        """子の ΔW が (1-α)ΔW1 + αΔW2 の最良 rank-r 近似に近いことを確認する。"""
        _make_adapter(tmp_path / "pa", seed=3)
        _make_adapter(tmp_path / "pb", seed=4)
        alpha = 0.3
        delta_blend_lora(str(tmp_path / "pa"), str(tmp_path / "pb"), str(tmp_path / "child"), alpha=alpha)

        pa = load_file(str(tmp_path / "pa" / "adapter_model.safetensors"))
        pb = load_file(str(tmp_path / "pb" / "adapter_model.safetensors"))
        child = load_file(str(tmp_path / "child" / "adapter_model.safetensors"))

        layer = "model.layers.0.q_proj"
        target = (1 - alpha) * _delta(pa, layer) + alpha * _delta(pb, layer)
        child_delta = _delta(child, layer)

        # 厳密 SVD による最良 rank-r 近似との誤差と比較して大差ないこと
        u, s, vh = torch.linalg.svd(target)
        best = u[:, :RANK] @ torch.diag(s[:RANK]) @ vh[:RANK, :]
        best_err = torch.norm(target - best)
        child_err = torch.norm(target - child_delta)
        assert child_err <= best_err * 1.05 + 1e-4

    def test_identical_parents_roundtrip(self, tmp_path):
        """同一の親同士のブレンドは元の ΔW をほぼ再現する（rank は保存される）。"""
        _make_adapter(tmp_path / "pa", seed=5)
        delta_blend_lora(str(tmp_path / "pa"), str(tmp_path / "pa"), str(tmp_path / "child"), alpha=0.5)
        pa = load_file(str(tmp_path / "pa" / "adapter_model.safetensors"))
        child = load_file(str(tmp_path / "child" / "adapter_model.safetensors"))
        layer = "model.layers.0.q_proj"
        assert torch.allclose(_delta(pa, layer), _delta(child, layer), atol=1e-3)

    def test_naive_blend_has_cross_term_error(self, tmp_path):
        """naive 方式は ΔW 補間から系統的に乖離する（交差項の存在確認）。"""
        _make_adapter(tmp_path / "pa", seed=6)
        _make_adapter(tmp_path / "pb", seed=7)
        alpha = 0.5
        alpha_blend_lora(str(tmp_path / "pa"), str(tmp_path / "pb"), str(tmp_path / "naive"), alpha=alpha)
        pa = load_file(str(tmp_path / "pa" / "adapter_model.safetensors"))
        pb = load_file(str(tmp_path / "pb" / "adapter_model.safetensors"))
        naive = load_file(str(tmp_path / "naive" / "adapter_model.safetensors"))
        layer = "model.layers.0.q_proj"
        target = (1 - alpha) * _delta(pa, layer) + alpha * _delta(pb, layer)
        naive_err = torch.norm(target - _delta(naive, layer))
        assert naive_err > 1.0  # ランダム親では交差項誤差が大きい


class TestMutate:
    def test_seed_reproducibility(self, tmp_path):
        _make_adapter(tmp_path / "pa", seed=8)
        mutate_lora(str(tmp_path / "pa"), str(tmp_path / "m1"), ratio=1.0, std=0.05, seed=123)
        mutate_lora(str(tmp_path / "pa"), str(tmp_path / "m2"), ratio=1.0, std=0.05, seed=123)
        m1 = load_file(str(tmp_path / "m1" / "adapter_model.safetensors"))
        m2 = load_file(str(tmp_path / "m2" / "adapter_model.safetensors"))
        for key in m1:
            assert torch.equal(m1[key], m2[key])

    def test_zero_ratio_is_identity(self, tmp_path):
        _make_adapter(tmp_path / "pa", seed=9)
        mutate_lora(str(tmp_path / "pa"), str(tmp_path / "m"), ratio=0.0, std=0.05, seed=1)
        original = load_file(str(tmp_path / "pa" / "adapter_model.safetensors"))
        mutated = load_file(str(tmp_path / "m" / "adapter_model.safetensors"))
        for key in original:
            assert torch.equal(original[key], mutated[key])

    def test_mutation_changes_weights(self, tmp_path):
        _make_adapter(tmp_path / "pa", seed=10)
        mutate_lora(str(tmp_path / "pa"), str(tmp_path / "m"), ratio=1.0, std=0.05, seed=2)
        original = load_file(str(tmp_path / "pa" / "adapter_model.safetensors"))
        mutated = load_file(str(tmp_path / "m" / "adapter_model.safetensors"))
        changed = any(not torch.equal(original[k], mutated[k]) for k in original)
        assert changed


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_delta_blend_endpoints(tmp_path, alpha):
    _make_adapter(tmp_path / "pa", seed=11)
    _make_adapter(tmp_path / "pb", seed=12)
    delta_blend_lora(str(tmp_path / "pa"), str(tmp_path / "pb"), str(tmp_path / "child"), alpha=alpha)
    pa = load_file(str(tmp_path / "pa" / "adapter_model.safetensors"))
    pb = load_file(str(tmp_path / "pb" / "adapter_model.safetensors"))
    child = load_file(str(tmp_path / "child" / "adapter_model.safetensors"))
    layer = "model.layers.0.q_proj"
    expected = _delta(pb, layer) if alpha == 1.0 else _delta(pa, layer)
    assert torch.allclose(expected, _delta(child, layer), atol=1e-3)
