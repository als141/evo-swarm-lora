import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def load_sft_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return load_dataset("json", data_files=path, split="train")


def format_sample(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a persona-specific LoRA adapter via QLoRA.")
    parser.add_argument("--data", required=True, help="Path to JSONL chat dataset.")
    parser.add_argument("--out", required=True, help="Directory to store the trained adapter.")
    parser.add_argument("--r", type=int, default=32, help="LoRA rank.")
    parser.add_argument(
        "--target",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        help="Target modules for LoRA injection.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--bsz", type=int, default=1, help="Per-device batch size.")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps.")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=8192,
        help="学習系列長の上限。TRLのSFTConfig.max_length既定1024では長CoTリプレイ例"
        "（run002で最長~7.4kトークン）が切り捨てられ能力保持の意味を失うため明示する。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    use_cuda = torch.cuda.is_available()
    if not use_cuda and os.environ.get("ALLOW_CPU_TRAINING") != "1":
        raise RuntimeError(
            "CUDA is not available. CPU へのサイレントフォールバックは禁止（実験が無効になる）。"
            "意図的に CPU で動かす場合は ALLOW_CPU_TRAINING=1 を設定すること。"
        )
    device_str = "cuda" if use_cuda else "cpu"
    print(f"[info] Loading base model on {device_str}.")

    # T4 (compute capability 7.5) は bf16 非対応のため fp16 へフォールバックする
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    load_kwargs = {"device_map": "auto"}
    if use_cuda:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = quant_cfg
        load_kwargs["torch_dtype"] = compute_dtype
    else:
        load_kwargs["device_map"] = None
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)
    if not use_cuda:
        model = model.to(torch.device("cpu"))
    # gradient_checkpointing と use_cache=True は競合するため学習時は無効化する
    model.config.use_cache = False

    dataset = load_sft_dataset(args.data)
    dataset = dataset.map(lambda ex: format_sample(ex, tokenizer), remove_columns=dataset.column_names)

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.r * 2,
        lora_dropout=0.05,
        target_modules=args.target,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        optim="paged_adamw_32bit" if use_cuda else "adamw_torch",
        gradient_checkpointing=True,
        packing=False,
        dataset_text_field="text",
        max_length=args.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    trainer.model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    size_mb = sum(
        os.path.getsize(os.path.join(args.out, file)) for file in os.listdir(args.out)
    ) / (1024 * 1024)
    print(f"Saved LoRA adapter to {args.out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
