# 進化的群知能による LoRA エージェント協調最適化フレームワーク

本リポジトリは **Qwen3-4B-Instruct-2507** をベースにした 3 エージェント構成の LoRA 学習・協調推論・進化的最適化・ベンチマーク評価を、ローカル GPU（A6000 相当）と **uv** による依存管理で再現可能にする研究用コードベースです。  
ローカル推論から vLLM Multi-LoRA サービング、進化ループ、LangGraph 連携の拡張までを段階的に実行するための手順と実装が含まれます。

---

## ハイライト
- **モデル**: Qwen/Qwen3-4B-Instruct-2507（Transformers ≥4.51, 256K コンテキスト）
- **学習**: QLoRA (4bit, bitsandbytes) + TRL SFTTrainer、rank 16–32 推奨
- **協調推論**: 3 人格 LoRA を切り替えた Multi-Agent Debate（Transformers ローカル → vLLM Multi-LoRA 拡張）
- **進化的最適化**: 適応度 = タスク性能 + 協調寄与 + 新規性、LoRA ブレンド＋突然変異で世代更新
- **評価**: lm-evaluation-harness / lighteval を OpenAI 互換 API（vLLM）で呼び出し比較
- **依存管理**: uv（Python 3.12）+ Docker（CUDA 12.4 ベース）で再現性確保

---

## 前提条件
- Ubuntu 22.04 以降、CUDA 12.4 対応 GPU（A6000 48GB を想定）
- `uv` インストール済み（`curl -LsSf https://astral.sh/uv/install.sh | sh`）
- Docker / Docker Compose v2（GPU プラグイン設定）
- Hugging Face アカウント（モデルキャッシュを共有する場合）
- ネットワーク帯域：PyTorch 2.9 + CUDA ランタイム群を初回ダウンロードするため 7GB 以上

---

## ディレクトリ構成
```
.
├─ adapters/                # 学習済み LoRA を格納（personas, 世代管理）
├─ data/                    # 人格ごとの SFT データ（JSONL）
├─ scripts/                 # 学習・議論・進化・評価スクリプト
├─ src/                     # モジュール（モデルラッパー・協調計算など）
├─ prompts/codex/           # Codex CLI 用タスクプロンプト
├─ Dockerfile               # GPU ありコンテナ（uv 同梱）
├─ docker-compose.yml       # trainer + vLLM Multi-LoRA サービス
├─ pyproject.toml           # 依存定義（uv 管理）
└─ uv.lock                  # uv 依存ロック
```

---

## セットアップ

### 1. 依存同期（uv）
初回は CUDA ランタイム群含め 3〜4 分かかります。
```bash
uv sync
```

### 2. Python 実行
uv で venv を管理しているため `uv run` を利用します。
```bash
uv run python -m compileall src scripts   # 静的チェック
uv run ruff check src scripts             # Lint
```

---

## SFT データと LoRA 学習

### 1. 人格データ編集
`data/sft_persona_*.jsonl` に {messages:[...]} 形式で会話を追記します。各人格 50〜200 ターンが目安。

### 2. LoRA 学習
```bash
uv run python scripts/train_lora_persona.py \
  --data data/sft_persona_a.jsonl \
  --out adapters/persona_a \
  --r 32 --epochs 1 --bsz 1 --grad_accum 16

uv run python scripts/train_lora_persona.py --data data/sft_persona_b.jsonl --out adapters/persona_b
uv run python scripts/train_lora_persona.py --data data/sft_persona_c.jsonl --out adapters/persona_c
```

出力: `adapter_config.json` と `adapter_model.safetensors` が `adapters/persona_*` に生成されます。  
GPU メモリが逼迫する場合は `--r 16` や `--grad_accum 32` で削減を検討。

---

## 協調ディベート（ローカル Transformers）

LoRA が揃ったら議論を実行します。
```bash
uv run python scripts/run_debate_local.py \
  --topic "消費税減税は日本経済にプラスか？" \
  --rounds 3
```
ログ出力: 各ターンの発話と信頼度重み付き投票の結果。  
`--rounds 0` を指定すれば実行前のドライラン（環境チェック）になります。

---

## 進化的最適化ループ（最小構成）

### 1. 適応度計算
`src/coop/fitness.py` に性能・協調寄与・新規性の計算が実装されています。  
Solo 精度や埋め込み距離は Notebook や追加スクリプトで計算し、JSON ログ化してください。

### 2. LoRA の交配・突然変異
```bash
uv run python scripts/evolve_loras.py \
  --parents adapters/persona_a adapters/persona_b \
  --child adapters/gen1_child_ab \
  --alpha 0.6 --mut 0.03 --mut-std 0.01
```
`adapters/gen1_child_ab` に子アダプターが生成されます。実験では世代ごとに `adapters/gen_{世代}/` を切ってバージョン管理します。

---

## vLLM Multi-LoRA サービング

### 1. 起動
```bash
docker compose up -d --build
```
- `trainer`: uv + GPU 環境。ローカルと同じコードベースをマウント済み。
- `vllm`: Qwen3-4B-Instruct-2507 を Multi-LoRA で公開（OpenAI 互換 API）。

### 2. 疎通チェック
```bash
uv run python scripts/ping_vllm_persona.py \
  --base-url http://localhost:8000/v1 \
  --persona persona_a \
  --prompt "次の数列の一般項は？ 2,4,8,16,..."
```
応答が人格ごとに差別化されることを確認してください。

---

## ベンチマーク評価

### lm-evaluation-harness
```bash
./scripts/eval_with_harness.sh
```
`out/persona_a_eval.json` に結果が出力されます。  
`MODEL_ARGS` を `persona_b` / `persona_c` / `genX` に切り替えて比較可能。

### LightEval
LightEval を利用する場合は `uv add lighteval` で追加し、OpenAI 互換エンドポイントを指定してください。

---

## LangGraph v1 への移行

`prompts/codex/04_langgraph_integration.md` に LangGraph v1 での移行タスクが定義されています。  
ステップ:
1. `uv add langgraph>=1.0.0`
2. Debate ノード（agent→metrics→judge）を LangGraph のステートマシンにマップ
3. `scripts/run_debate_local.py` と出力パリティを検証

---

## テストと検証
- **Ruff**: `uv run ruff check src scripts`
- **compileall**: `uv run python -m compileall src scripts`
- **Docker ビルド**: `docker compose build trainer`
- **実行ドライラン**: `uv run python scripts/run_debate_local.py --topic "ping" --rounds 0`
- **本番動作**: LoRA 学習 → Debate → Evolve → vLLM 疎通 → Harness 評価

---

## 実装メモ
- Python モジュールは `src/` に配置し、スクリプトからは `sys.path` でルートを追加
- LoRA ブレンドと突然変異は `safetensors` を直接操作
- Debate の投票は簡易的な信頼度加重多数決（構造化出力に拡張可能）
- 進化ループは JSON ログを残し、適応度の推移を可視化する（Jupyter 等）
- `prompts/codex/*.md` は Codex CLI で段階的に投入できるタスクシート

---

## 参考コマンド一覧
```bash
# 依存同期
uv sync

# LoRA 学習（例）
uv run python scripts/train_lora_persona.py --data data/sft_persona_a.jsonl --out adapters/persona_a

# Debate 実行
uv run python scripts/run_debate_local.py --topic "都市のヒートアイランド対策を議論" --rounds 3

# LoRA 交配
uv run python scripts/evolve_loras.py --parents adapters/persona_a adapters/persona_b --child adapters/gen1_ab --alpha 0.5 --mut 0.05

# vLLM 起動
docker compose up -d vllm

# OpenAI SDK 疎通
uv run python scripts/ping_vllm_persona.py --persona persona_c --prompt "高齢化社会の孤立感をどう和らげる？"
```

---

## 注意事項
- Qwen3-4B のダウンロードと CUDA ランタイムでストレージを大きく消費します（最低 40 GB を確保）
- bitsandbytes は GPU アーキテクチャに依存します。CUDA 12.4 + Ampere で動作確認済み
- FlashAttention-3 を利用する場合は別途ビルドが必要（任意）
- 学習データはプライバシーに配慮し、ライセンス遵守のもとで収集してください
- 進化ループで生成した LoRA は評価ログとともに保存し、再現性を維持してください

---

## ライセンスと引用
- ベースモデル: [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)（Apache-2.0）
- ライブラリ: Transformers, TRL, PEFT, bitsandbytes, vLLM, LangGraph（各ライセンスに従う）
- このリポジトリのコードは MIT ライセンス相当（必要に応じて追記してください）

---

## サポート
バグや質問は Issue で報告してください。  
大きな設計変更（LangGraph への完全移行など）はディスカッションを立てて議論することを推奨します。
