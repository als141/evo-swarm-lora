import argparse
import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import torch

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
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        quantization_config=quant_cfg,
        device_map="auto",
    )

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
        bf16=True,
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        packing=False,
        dataset_text_field="text",
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
