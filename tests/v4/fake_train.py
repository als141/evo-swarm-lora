"""結合試験用のダミー学習: 例数を数えて、交叉が動く最小の LoRA safetensors を書く。"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

examples, out = sys.argv[1], Path(sys.argv[2])
n = sum(1 for line in open(examples) if line.strip())
out.mkdir(parents=True, exist_ok=True)
g = torch.Generator().manual_seed(n)
weights = {}
for layer in range(2):
    prefix = f"base_model.model.model.layers.{layer}.self_attn.q_proj"
    weights[f"{prefix}.lora_A.weight"] = torch.randn(4, 16, generator=g)
    weights[f"{prefix}.lora_B.weight"] = torch.randn(16, 4, generator=g)
save_file(weights, str(out / "adapter_model.safetensors"))
(out / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8, "peft_type": "LORA"}))
(out / "train_info.json").write_text(json.dumps({"n_examples": n, "final_loss": 0.0, "elapsed_s": 0}))
