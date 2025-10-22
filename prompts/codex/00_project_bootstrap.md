# GOAL
- Qwen3-4B + LoRA(QLoRA) + 3人格のマルチエージェント協調→協調的最適化の最小研究実装。
- uv / Docker 前提。A6000 1枚で実験可能。

# ACTIONS
1) 付与された pyproject.toml / Dockerfile / docker-compose.yml をベースに、venv同期とビルドを通す。
2) src/*, scripts/*, data/* を生成（不足があれば質問せず合理的に補完）。
3) run_debate_local.py が単独で動くことをCI風に検証用スクリプトを作る。

# ACCEPTANCE
- `uv sync` が成功。
- `python scripts/run_debate_local.py --topic "テスト"` がエラーなく終わる。
