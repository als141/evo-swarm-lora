"""v4 の LoRA 学習（bf16・A100）。社会学習（議論ログからの学習）で子個体を作る。

2026-07 実装（scripts/train_lora_persona.py）からの修正:
- QLoRA（4bit 量子化ベースで学習→bf16 ベースで推論）をやめ、bf16 ベースで直接学習する
- 損失は最後の assistant 発話（とその終端トークン）だけに掛ける（旧実装は全トークンに掛けていた）
- 長い議論プロンプトでも、損失を掛ける位置だけ lm_head を計算してメモリを節約する
- 親アダプタからの継続学習（ラマルク的継承）と、seed の完全固定
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tokenize_example(tokenizer, messages: List[dict], max_len: int) -> Optional[Dict[str, List[int]]]:
    """最後の assistant 発話だけにラベルを付けた input_ids / labels を返す（長すぎれば None）。"""
    assert messages[-1]["role"] == "assistant"
    prompt_ids = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True)
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        return None  # テンプレートの不整合（想定外）は学習に使わない
    # 末尾の改行トークン（<|im_end|> の後）は学習対象から外す
    end = len(full_ids)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    while end > len(prompt_ids) and full_ids[end - 1] != im_end:
        end -= 1
    if end <= len(prompt_ids) or end > max_len:
        return None
    input_ids = full_ids[:end]
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
    return {"input_ids": input_ids, "labels": labels}


def train_lora(examples: List[List[dict]], out_dir: str, parent_adapter: Optional[str] = None,
               seed: int = 0, lr: float = 1e-4, epochs: int = 1, rank: int = 16, alpha: int = 32,
               dropout: float = 0.05, tokens_per_step: int = 32768, max_len: int = 8192,
               warmup_ratio: float = 0.05, log=print) -> dict:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    data = []
    dropped = 0
    for messages in examples:
        tokenized = tokenize_example(tokenizer, messages, max_len)
        if tokenized is None:
            dropped += 1
        else:
            data.append(tokenized)
    if not data:
        raise ValueError("no usable training examples")

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if parent_adapter:
        model = PeftModel.from_pretrained(model, parent_adapter, is_trainable=True)
    else:
        config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=TARGET_MODULES,
                            bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
    model.enable_input_require_grads()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0, betas=(0.9, 0.999))

    total_tokens = sum(sum(1 for t in d["labels"] if t != -100) for d in data) * epochs
    total_steps = max(1, math.ceil(total_tokens / tokens_per_step))
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    lm_head = model.get_output_embeddings()
    backbone = model.get_base_model().model  # Qwen3Model（LoRA 層を含む）
    model.train()
    step, acc_tokens, acc_loss, seen_label_tokens = 0, 0, 0.0, 0
    history = []
    order = list(range(len(data)))
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(order)
        for idx in order:
            example = data[idx]
            input_ids = torch.tensor([example["input_ids"]], device="cuda")
            labels = torch.tensor(example["labels"], device="cuda")
            positions = torch.nonzero(labels[1:] != -100).squeeze(-1)  # 次トークン予測で損失を掛ける位置
            hidden = backbone(input_ids=input_ids).last_hidden_state[0]  # [T, H]
            logits = lm_head(hidden[positions]).float()
            target = labels[1:][positions]
            loss_sum = torch.nn.functional.cross_entropy(logits, target, reduction="sum")
            n_tok = int(target.numel())
            (loss_sum / tokens_per_step).backward()
            acc_tokens += n_tok
            acc_loss += float(loss_sum.detach())
            seen_label_tokens += n_tok
            if acc_tokens >= tokens_per_step:
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                history.append({"step": step, "loss": acc_loss / acc_tokens, "tokens": acc_tokens})
                if step % 5 == 0:
                    log(f"[train] step {step}/{total_steps} loss={acc_loss / acc_tokens:.4f} "
                        f"elapsed={time.time() - started:.0f}s")
                step += 1
                acc_tokens, acc_loss = 0, 0.0
    if acc_tokens > 0:  # 端数のステップ
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        history.append({"step": step, "loss": acc_loss / acc_tokens, "tokens": acc_tokens})

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    info = {
        "parent_adapter": parent_adapter, "seed": seed, "lr": lr, "epochs": epochs, "rank": rank,
        "alpha": alpha, "dropout": dropout, "tokens_per_step": tokens_per_step, "max_len": max_len,
        "n_examples": len(data), "n_dropped": dropped, "label_tokens": seen_label_tokens,
        "steps": len(history), "final_loss": history[-1]["loss"] if history else None,
        "elapsed_s": round(time.time() - started, 1), "history": history,
    }
    (Path(out_dir) / "train_info.json").write_text(json.dumps(info, indent=1))
    del model, optimizer
    torch.cuda.empty_cache()
    return info
