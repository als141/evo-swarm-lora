from __future__ import annotations

from typing import List

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


def load_base(device_map: str | dict | None = "auto", load_in_4bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
    }
    if load_in_4bit:
        kwargs.update(
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
            }
        )
    model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME, **kwargs)
    model.config.use_cache = True
    return model, tokenizer


def attach_lora(model, adapter_dir: str):
    PeftConfig.from_pretrained(adapter_dir)  # ensures adapter is valid
    return PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)


def generate(
    model,
    tokenizer,
    messages: List[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][tokens.input_ids.shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)
