# v4 実験環境イメージ（固定）

- URI: `us-central1-docker.pkg.dev/pro-plasma-510112-m7/evo-swarm/env-v4@sha256:41ab70107a39096377da1fef113d9ef9fbd5f1ce3b016cabfc979a2d13da8d71`
- タグ: `env-v4:20260929`（Cloud Build 5728bfbf-8794-4ced-94af-94ba744446db、2026-09-29）
- ベース: `vllm/vllm-openai:v0.8.5`
- 主要版: torch 2.6.0+cu124 / transformers 4.51.3 / vllm 0.8.5 / peft 0.15.2 / datasets 3.6.0
- 推論経路パッケージは `constraints.txt` で同梱版に固定（変更されればビルド失敗）
- 全パッケージ: `pip_freeze_env-v4_20260929.txt`

**規則: v4 の全実験（学習・評価・進化）はこの digest だけを使い、実験期間中は再ビルドしない。**
コードはイメージに含めず、ジョブ起動時に `CODE_DIR`（GCS 上の git SHA 別スナップショット）から取得する。
