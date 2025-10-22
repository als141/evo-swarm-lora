# GOAL
- data/sft_persona_{a,b,c}.jsonl を読み込み、QLoRAで3人格LoRAを adapters/ に書き出す。

# ACTIONS
- scripts/train_lora_persona.py の引数やデフォルトを調整し、短時間で学習が終わるようにする（データが少量のため）。
- 学習ログとLoRAサイズを標準出力に表示。

# ACCEPTANCE
- adapters/persona_a|b|c に adapter_config.json / adapter_model.safetensors が生成。
- 3人格で run_debate_local.py が動作。
