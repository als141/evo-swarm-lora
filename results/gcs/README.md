# 実験データアーカイブ（GCS完全ミラー）

GCS `gs://evo-swarm-lora-usc1-research-501308/experiments/` のローカルミラー。
**このディレクトリと `artifacts_local/`（git外）があれば、Google Cloud なしで全実験データを参照できる。**
`.jsonl` と一部の大きな `.json` は gzip 圧縮済み（`zcat` / `gzip -d` で読む）。

## measurement environment（最重要の注意）

評価コンテナイメージ間に系統差がある（同一問題・同一設定で+6pt級、修論 §4.6.4）。
**異なる環境の数値を直接比較してはならない。**

- **旧環境（イメージ 2026-07-04 版）**: `run001/final_eval3/` の全63エントリ（実験1）
- **新環境（イメージ 2026-07-05 版）**: `run002/` 配下すべて（実験2・再測定・頑健性）

修論の最終比較は新環境のみ: `run002/{recheck_c2s1, remeasure_v1, robust_c7, robust_c2, g3_team_check, final_c7}`。
統計スクリプト: `scripts/analysis/final_stats_clean.py`（同一環境）, `final_stats.py`（実験1）。

## run001/（実験1: v1プロトコル）

| パス | 内容 |
|---|---|
| `final_eval3/` | 実験1最終評価63エントリ（7条件×3ベンチ×3シード）。per_item付き確定JSON |
| `evolution/run_log.json` | 進化ループ6世代の全ログ（適応度・Shapley値・選抜履歴） |
| `evolution/configs等` | 進化の設定。**アダプタ実体は `artifacts_local/adapters/run001_evolution/`** |
| `transcripts_demo/` | 議論の生会話デモ60問（プロンプト+3体の全発話） |
| `transcripts_math500_diag/` | 診断実験: MATH-500 40問×3条件の生会話120レコード（CoT圧縮の証拠、修論 §4.6.2） |
| `replay/replay_pool.jsonl` | 能力保持リプレイ36例（ベース自己生成・正解検証済み長CoT） |
| `configs/` | 評価バッテリー・パイロットの設定JSON |
| `pilot_debate_style/` | conditional vs standard 議論スタイルのA/Bパイロット（300問） |

## run002/（実験2: 処方の検証、すべて新環境）

| パス | 内容 |
|---|---|
| `data/` | run002 SFTデータ（ペルソナ60例+リプレイ36例×3人格） |
| `g1_solo_check/` | G1ゲート: 再学習後solo点検（3ベンチ200問、毀損完治の判定） |
| `g2_aggregation/` | G2ゲート: 集約ablation（多数決/重み付き/GenSelect、SGPQA150問） |
| `g3_team_check/` | G3ゲート: チームseed1本番（transcripts_team.json.gz=生会話全記録付き） |
| `final_c7/` | 新チームseed2,3の最終評価 |
| `robust_c7/`, `robust_c2/` | 頑健性確認seed4-6（新チーム/SC@9） |
| `recheck_c2s1/` | イメージ系統差の切り分け検証（SC@9 seed1再実行） |
| `remeasure_v1/` | 環境統制の再測定26エントリ（SC@9/ベース/旧チーム×3シード） |
| `*/llm_calls/*.jsonl.gz` | **全LLM呼び出しの完全記録**（入力プロンプト全文+生成全文+logprob confidence+設定）。1行=1呼び出し |
| `*/progress/` | （GCSのみ、ミラー除外）問題単位の再開キャッシュ。確定JSONと重複のため省略 |

## artifacts_local/（git外、ローカルのみ ~5.7GB）

| パス | 内容 |
|---|---|
| `adapters/run001_gen0/persona_{a,b,c}/` | 実験1のgen-0 LoRAアダプタ（rank32） |
| `adapters/run001_evolution/gen_*/` | 進化全世代の子アダプタ（最終チーム=gen_04/gen4_critic_child, gen_05/gen5_{pragmatist,explorer}_child） |
| `adapters/run002_replay/persona_{a,b,c}/` | 実験2のリプレイ再学習アダプタ（rank16、最終チームc7の実体） |

GCS原本: `gs://evo-swarm-lora-usc1-research-501308/experiments/`（課金停止後は削除される可能性あり。本ミラーが一次保存先）。
