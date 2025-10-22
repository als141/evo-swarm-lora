# Codex CLI Agent Briefing

このドキュメントは Codex CLI で本リポジトリを操作するエージェント向けの実務ガイドです。全コマンドの実行が許可されているため、タスクを完遂するまで必要な操作を遠慮なく行ってください。

---

## プロジェクト概観
- テーマ: Qwen3-4B × LoRA × 3 エージェント協調 → 進化的最適化
- 依存管理: uv（Python 3.12, `.venv` 自動生成）
- GPU 要件: CUDA 12.4, A6000 1 枚を想定
- 主なロール:
  - `scripts/train_lora_persona.py` – QLoRA 学習
  - `scripts/run_debate_local.py` – 3 人格ディベート
  - `scripts/evolve_loras.py` – LoRA ブレンド＆突然変異
  - `scripts/ping_vllm_persona.py` – vLLM 疎通
  - `scripts/eval_with_harness.sh` – ベンチマーク起動

---

## これまでの実装プロセス
1. `pyproject.toml` を uv 仕様へ拡張（依存群 + dependency-groups）。
2. `data/`, `src/`, `scripts/`, `prompts/codex/` を作成し、人格データ・モデルラッパー・ユーティリティを実装。
3. QLoRA 学習スクリプト、議論ロジック、LoRA 進化オペレーション、OpenAI 互換疎通スクリプトを追加。
4. Dockerfile / docker-compose で trainer + vLLM Multi-LoRA サービスを定義。
5. `uv sync` 実行で PyTorch 2.9 + CUDA12 ランタイムを含む依存を取得し、`compileall` / `ruff` ですべてのモジュールを検証。
6. README を全面更新、Codex タスクプロンプトと本ガイドを整備。

---

## 主要コマンド
```bash
# 依存同期
uv sync

# Lint / 静的解析
uv run ruff check src scripts
uv run python -m compileall src scripts

# LoRA 学習
uv run python scripts/train_lora_persona.py --data data/sft_persona_a.jsonl --out adapters/persona_a

# 協調ディベート
uv run python scripts/run_debate_local.py --topic "議論トピック" --rounds 3

# LoRA 交配・突然変異
uv run python scripts/evolve_loras.py --parents adapters/persona_a adapters/persona_b --child adapters/gen1 --alpha 0.6 --mut 0.03

# vLLM 起動／疎通
docker compose up -d --build
uv run python scripts/ping_vllm_persona.py --persona persona_a

# ベンチマーク
./scripts/eval_with_harness.sh
```

---

## 実験・評価フロー
1. **SFT データ整備**: `data/sft_persona_*.jsonl` を編集し会話数を増やす。
2. **LoRA 学習**: 3 人格それぞれ `train_lora_persona.py` でアダプタ生成。
3. **ローカル Debate**: `run_debate_local.py` で協調推論ログ取得。
4. **適応度計算**: Notebook などでソロ性能・チーム性能・新規性を算出。
5. **進化ループ**: `evolve_loras.py` を繰り返し、`adapters/gen_*` を生成。
6. **vLLM 評価**: Multi-LoRA を OpenAI 互換 API で提供し、`eval_with_harness.sh` でスコアリング。
7. **LangGraph 拡張**（必要なら）: `prompts/codex/04_langgraph_integration.md` を参照し、状態遷移を移植。

---

## uv 運用のヒント
- デフォルトで `.venv` が作成される。常に `uv run ...` で仮想環境を利用。
- 依存追加は `uv add <pkg>`、削除は `uv remove <pkg>`。
- `uv lock` の差分をコミットし、再現性を担保する。

---

## テスト方針
- **形式検査**: `uv run python -m compileall src scripts` で構文と import を検証。
- **スタイル**: `uv run ruff check src scripts`。必要に応じ `--fix` を使用。
- **実行試験**: `uv run python scripts/run_debate_local.py --topic "ping" --rounds 0` で動的依存をチェック。
- **コンテナ検証**: `docker compose build trainer` → `docker compose up trainer` で GPU 利用可否を確認。
- **ベンチマーク**: `./scripts/eval_with_harness.sh` を通して JSON ログの生成を確認。

---

## 注意点
- 研究用途のため、LoRA ファイルや評価ログは世代ごとに保存し、メタデータ（乱数 seed, 訓練コマンド, ハードウェア構成）を記録。
- CUDA ランタイム群は容量が大きい。キャッシュ volume (`hf-cache`) を共有して重複ダウンロードを避ける。
- LoRA 交配では親子アダプタの rank が一致していることを確認。
- 特別な要求がない限り destructive コマンド（`git reset --hard` 等）は使用しない。

---

## 参考タスクプロンプト
- `prompts/codex/00_project_bootstrap.md` – 環境再構築
- `prompts/codex/01_train_three_loras.md` – LoRA 学習
- `prompts/codex/02_multilora_vllm.md` – vLLM Multi-LoRA
- `prompts/codex/03_evolutionary_loop.md` – 進化ループ
- `prompts/codex/04_langgraph_integration.md` – LangGraph 移行

---

## 最後に
全コマンド実行が許可されています。A/B 試験、追加データ収集、LangGraph への拡張など、新しい試みは `docs/` や Issue にメモを残しながら進めてください。必要に応じて README と本ガイドを更新し、後続のエージェントがスムーズに引き継げる状態を維持してください。
