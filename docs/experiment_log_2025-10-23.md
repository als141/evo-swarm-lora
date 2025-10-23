# 実行ログ (2025-10-23)

## 実行概要
2025-10-23 に以下のコマンドを順番に実行し、最小構成の LoRA 学習・協調議論・進化コアを実機検証しました。

### 1. 依存同期
```
uv sync
```
- PyTorch 2.9 / Transformers 4.57 / TRL 0.24 / bitsandbytes 0.48.1 等をダウンロード。
- CUDA 12.x ランタイム群（合計 3GB 超）を取得し、`.venv` が生成されました。

### 2. LoRA 学習
```
uv run python scripts/train_lora_persona.py --data data/sft_persona_a.jsonl --out adapters/persona_a --epochs 1 --bsz 1 --grad_accum 1
uv run python scripts/train_lora_persona.py --data data/sft_persona_b.jsonl --out adapters/persona_b --epochs 1 --bsz 1 --grad_accum 1
uv run python scripts/train_lora_persona.py --data data/sft_persona_c.jsonl --out adapters/persona_c --epochs 1 --bsz 1 --grad_accum 1
```
- Qwen3-4B-Instruct-2507 を BitsAndBytesConfig 4bit NF4 でロード。
- TRL `SFTTrainer` (v0.24) に合わせて `processing_class=tokenizer` 等へ修正。
- 各実行で adapter_config.json + adapter_model.safetensors (≈267MB) を生成。
- 训练ログ例: `train_loss≈3.28`, `mean_token_accuracy≈0.56` 等が標準出力に記録。

### 3. 協調ディベート (ラウンド1)
```
uv run python scripts/run_debate_local.py --topic "消費税減税は日本経済にプラスか？" --rounds 1
```
- 3人格 LoRA を順次適用し、発話ログと投票結果をコンソール出力。
- 現状は各発話の先頭文字を投票に用いる簡易ロジックのため、最終回答が `"-"` になるなど改善余地を確認。

### 4. LoRA 交配・突然変異
```
uv run python scripts/evolve_loras.py --parents adapters/persona_a adapters/persona_b --child adapters/gen1_child_ab --alpha 0.5 --mut 0.02 --mut-std 0.01
```
- `persona_a` と `persona_b` の LoRA 重みを線形合成し、5% ノイズを付与。
- 結果として `adapters/gen1_child_ab/adapter_model.safetensors` が生成。

### 5. `.gitignore` 調整
```
(adapters/*, *.safetensors, out/, .cache/ などを ignore)
```
- 生成物が GitHub 制限 (100MB) を超えるため Git 管理から除外しました。

## 成果物
| 種別 | パス | 説明 |
| --- | --- | --- |
| LoRA アダプタ | `adapters/persona_{a,b,c}/adapter_model.safetensors` | 各人格の QLoRA 重み (267MB) |
| LoRA 設定 | `adapters/persona_{a,b,c}/adapter_config.json` | rank, α, modules 等 |
| 進化子アダプタ | `adapters/gen1_child_ab/adapter_model.safetensors` | A/B ハイブリッド |
| ディベートログ | コンソール出力 (未保存) | Round-0 発話・投票 | 
| ソース更新 | `scripts/train_lora_persona.py`, `.gitignore` | TRL v0.24 対応, 大容量 ignore |

※ safetensors は `.gitignore` により Git へは含まれていません。必要に応じて Git LFS / 外部ストレージへ保存してください。

## ネクストアクション
1. **投票ロジック改善**: `extract_vote` を構造化出力(JSON)解析に置き換え、最終回答の耐性を上げる。
2. **評価ハーネス**: `./scripts/eval_with_harness.sh` を persona_a/b/c/gen1 それぞれに対して実行し、`out/` に JSON を保存。
3. **適応度計測の自動化**: Solo / チーム性能・新規性計算を Notebook または `scripts/compute_fitness.py` などとして実装。
4. **vLLM Multi-LoRA サービング**: `docker compose up -d` → `scripts/ping_vllm_persona.py` を実行し、OpenAI 互換 API 経由で人格差異を確認。
5. **LangGraph v1 統合**: `prompts/codex/04_langgraph_integration.md` に従い、ディベートの状態遷移を LangGraph 化し、UI/監視を拡張。
6. **安全性と品質分析**: 低品質発話や幻覚検知、議論収束性の統計を取り、論文化に向けた定量評価指標を整備。

## 補足
- すべてのコマンドは Codex CLI で許可済み。GPU/HF キャッシュに依存するため Docker 環境との差分に留意。
- 強制終了されたコマンドは `uv sync` 実行中のタイムアウトのみで、その後再実行し成功済み。
- 実験ログは今後も日付別に追記していくことを推奨。
