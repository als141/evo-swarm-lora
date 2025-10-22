import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model weights.")
    parser.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507", help="Base model identifier.")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory.")
    parser.add_argument("--out", required=True, help="Directory to save the merged model.")
    args = parser.parse_args()

    base_model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base_model, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out)
    print(f"Merged model saved to {args.out}")


if __name__ == "__main__":
    main()
