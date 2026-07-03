# Agent Briefing (Codex CLI / Claude Code 共通)

このドキュメントは本リポジトリを操作するエージェント向けの実務ガイドです。全コマンドの実行が許可されているため、タスクを完遂するまで必要な操作を遠慮なく行ってください。

> **運用ルール(必須)**: 新しい発見・気付き・修正・変更・参照ソースが生じたら、その都度このファイル末尾の「研究ログ」セクションに日付付きで追記すること。環境設定(GCPプロジェクト・バケット・クォータ等)が変わった場合は「クラウド環境の現状」セクションも更新すること。

---

## プロジェクト概観
- テーマ: Qwen3-4B × LoRA × 3 エージェント協調 → 進化的最適化
- ベースモデル: `Qwen/Qwen3-4B-Instruct-2507`（ゲートなし・Apache-2.0。**HFトークン不要**）
- 依存管理: uv（Python 3.12, `.venv` 自動生成）
- GPU 要件: CUDA 12.4, A6000 1 枚を想定（※ドキュメント上の想定。**実際のローカルGPUは GTX 1660 6GB のみ**のため、学習・vLLMサービングはクラウド前提。下記「クラウド環境の現状」参照）
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

## クラウド環境の現状（2026-07-03 構築）

計算資源は GCP で提供。**予算上限 $300（課金アカウントはJPY建てのため ¥45,000 で予算アラート設定済み: 50/80/95/100% で通知）**。

| 項目 | 値 |
|---|---|
| GCPアカウント | gaku.masuda@starup01.jp |
| gcloud 構成 | `evo-swarm`（`gcloud config configurations activate evo-swarm` で切替） |
| プロジェクト | `research-501308`（課金アカウント 017D39-1B6DE0-7AD4AB にリンク済み） |
| リージョン | asia-northeast1（既定） |
| GCSバケット | `gs://evo-swarm-lora-research-501308`（asia-northeast1、成果物・LoRAアダプタ保存先） |
| GCSバケット(実験用) | `gs://evo-swarm-lora-usc1-research-501308`（us-central1。**Vertexジョブは同リージョンバケット必須**のため実験成果物はこちら） |
| 認証 | ユーザー認証 + ADC 設定済み（コードは `aiplatform.init()` / `storage.Client()` を引数なしで呼ぶADC前提） |
| 有効化済みAPI | aiplatform / storage / artifactregistry / compute / cloudbuild / billingbudgets |

**GPUクォータの実態（2026-07-03 時点）— 計画を左右する重要事実**:
- Compute Engine の GPU は `GPUS_ALL_REGIONS=0` → **GPU VM は立てられない**（増枠申請 or アカウントアップグレードが必要）
- Vertex AI Custom Training は **preemptible（Spot）のみ利用可**:
  - Spot **T4 ×1** — asia-northeast1 / us-central1 / us-west1（安価 ~$0.1/h。**bf16非対応なのでfp16を使う**こと）
  - Spot **A100 ×8** — us-central1（bf16可・高速）
  - Spot V100/P100 ×1 — asia-east1 / us-central1 / us-west1
  - オンデマンド（非Spot）GPU は全て 0
- 結論: 学習・評価は **Vertex AI の Spot ジョブ** として実行する。Spotはプリエンプトされうるため、チェックポイント保存と再開を前提に設計する。

環境変数（ハードコードなし。Vertex実行時は `AIP_*` が自動注入される）:
- `CLOUD_ML_PROJECT_ID=research-501308` / `CLOUD_ML_REGION=asia-northeast1`
- `AIP_MODEL_DIR` / `AIP_CHECKPOINT_DIR` / `AIP_TENSORBOARD_LOG_DIR`（gs:// URI）

---

## 研究ログ（随時追記・新しいものを上に）

### 2026-07-04: 進化ループ本番完了（6世代）— 開発セットでチーム精度+12pt
- **run001 進化ジョブ成功**（A100 Spot、約3.5h、6世代×3役割×2個体、適応度=厳密Shapley×fitness sharing、MMLU-Pro固定100問）。
- **結果**: チーム精度 gen0 0.56 → gen5 0.68（開発セット上、選抜ノイズ含む）。最終チームは全役割が進化産の子個体（gen4_critic_child / gen5_pragmatist_child / gen5_explorer_child）。
- **特筆すべき挙動**: gen0でpragmatist役はsolo精度最低(0.49)の個体がチーム精度最大(0.63)でShapley選抜された = 「協調寄与適応度が個体性能のみでは拾えない個体を選ぶ」という本研究の主張の実例。修論考察の材料。
- スモークテストで厳密Shapleyの効率性（Σφ=チーム精度）を実測確認済み。GSM8Kは8/8全問正解で飽和を実地確認（主ベンチから外した判断の裏付け）。
- **最終評価バッテリー投入**（63エントリ=7条件×3ベンチ×3シード、適応度セット100問はMMLU-Pro評価から除外済み）。完了後に統計分析→修論結果章へ。

### 2026-07-03: 【重要】VertexのGPUドライバはCUDA 12.2相当 — cu124以降のtorchはCPUフォールバックする
- **症状**: T4学習ジョブが極端に遅い+初回はRAM不足で失敗。ログに `CUDA initialization: The NVIDIA driver on your system is too old (found version 12020)` → torch cu124 が初期化できず**サイレントにCPU学習**していた。
- **対処**: (1) trainer イメージの torch を **cu118 ビルド**に変更（ドライバ12.2でネイティブ動作）。(2) `train_lora_persona.py` は CUDA 不可なら即エラーに変更（`ALLOW_CPU_TRAINING=1` でのみ CPU 許可）。(3) eval イメージ（vLLM は cu124/cu128 ビルド）は entrypoint で `/usr/local/cuda/compat` を LD_LIBRARY_PATH に追加+`nvidia-smi` ログ出力。**A100ホストのドライバも535系ならvLLMが同じ問題を踏む可能性**があり、スモークテストの nvidia-smi 出力で確認する。ダメなら vLLM の cu118 系イメージ/自前ビルドへ切替。
- **教訓**: ジョブ投入時は必ず「GPUが実際に使われているか」をログで確認する（`Loading base model on cuda` / nvidia-smi）。CPUフォールバックは静かに実験を無効化する。

### 2026-07-03: 修論Markdownドラフトの LaTeX テンプレ移植完了（thesis/）
- **移植**: `docs/thesis/01〜04_*.md` → `thesis/Sec1〜Sec4.tex`（原本の md は無変更）。Sec4 は実験設定（4.1〜4.5）＋「4.6 実験結果」プレースホルダ（TODOコメント）。Sec5（考察）/Sec6（結論）/abstract.tex は節見出し＋TODOのみの骨組み。ラベルはテンプレ規約（`cha:N` / `sec:N.M(.L)` / `tab:` / `eq:` / `alg:`）。句読点はテンプレに合わせ「，．」へ統一。
- **参考文献**: 各章末リストを `thesis/refer.bib` に統合（重複除去して **63エントリ**、key は著者年形式 例 `du2023improving`）。ダミーの Back1997 は削除。著者不明の arXiv エントリ6件・タイトル未確認1件（arXiv:2511.11040）は note に「要確認」を明記。
- **表紙**: タイトル「マルチエージェント議論におけるチームレベル適応度を用いたLoRAエージェント集団の進化的最適化」、著者 増田 学。年度・学籍番号・指導教員・英文タイトルは TODO コメント。
- **コンパイル環境の発見**: `platex`/`uplatex`/`pbibtex` は PATH になく `~/.TinyTeX/bin/x86_64-linux/`（TeX Live 2025）に存在（`/usr/local/texlive/2024` はほぼ空）。**jreport は pLaTeX 用クラスのため uplatex では JY1 エンコーディングエラー → platex を使うこと**。不足パッケージ algorithms/multirow/appendix は `tlmgr --repository https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2025/tlnet-final install ...` で導入（通常リポジトリは2026に進んでおりクロスリリース拒否）。
- **検証**: `platex → pbibtex → platex ×2 → dvipdfmx` でエラー0・未解決参照0・49ページの main.pdf 生成を確認。amsmath はテンプレ未読込のため数式は kernel 互換記法（eqnarray, \mbox, \mathop）で記述している点に注意。

### 2026-07-03: 実験方針の意思決定（ユーザー確認済み）と修論テンプレ展開
- **ユーザー決定**: (1) 修論は大学指定LaTeXテンプレ（`template/修論見本.zip`→jreport, Sec1〜6構成）に合わせる。`template/latex/` に正名展開し、作業コピーを `thesis/` に配置、Markdownドラフトの移植を実施中。 (2) 進化実験は**まず主実験のみ**（A1 solo適応度アブレーションは結果を見て判断）。 (3) **debateラウンドは1で固定**（文献根拠: 2502.19130等「ラウンド増は逆効果」）。
- **SFTデータ補記**: persona_a（批判的検証者）・persona_c（発散的探索者）も各60例で生成完了済み（Bと同構成: 英語算数20/英語4択10/日本語タスク15/日本語議論応答10/特殊5。数値答は独立再計算で全検証）。3ファイルとも1行1JSON・system固定・重複なしを機械検証済み。
- **研究倫理の方針**: 「有意な結果が出るまで」の要請は、開発イテレーション（適応度セット100問上のチューニング）と最終評価（固定プロトコル・K=3シード一発）の分離で対応。p-hackingを構造的に排除する。

### 2026-07-03: クラウド実験開始とインフラの学び
- **学習ジョブ失敗→修正**: n1-standard-8(30GB RAM)+T4 で「Replicas low on memory」により FAILED → **n1-highmem-8(52GB) に変更**して解決方向。
- **Vertexの制約**: baseOutputDirectory のバケットはジョブと同リージョン必須 → `gs://evo-swarm-lora-usc1-research-501308`(us-central1) を新設し実験成果物はここに置く。
- **vLLM更新**: 最新安定版 v0.24.0 (2026-06-29) を確認。eval イメージを ARG 化し、v0.8.5(既定) と v0.24.0(eval:v024) の両方をビルド。スモークで比較して本番採用を決める。
- **評価バグ修正**: run_eval.py の --task choices が旧タスク名のままで新ベンチが実行不能だった問題を修正（修論3-4章起草エージェントのコード照合で発見）。適応度セットと最終評価の重複排除 (--exclude-items-file) も実装。
- 修論ドラフト第1-4章を docs/thesis/ に起草完了（実験結果非依存部分）。2026年文献の書誌情報は最終稿前に原典確認要。

### 2026-07-03: 修論 第3章・第4章ドラフト起草
- `docs/thesis/03_method.md`（提案手法、数式・記号表・Algorithm 1 付き）と `docs/thesis/04_experimental_setup.md`（実験設定、比較8条件・ハイパラ表5点付き）を新規作成。research_design.md §3-§7 と実装（`src/evolve/loop.py`, `src/evalx/{shapley,debate,tasks}.py`, `src/models/lora_ops.py`, `scripts/run_evolution.py`, `scripts/run_eval.py`, `scripts/cloud/*`）に一致させて記述。未実施の実験結果への言及なし。
- **執筆中に確認した実装事実（論文に反映済み）**:
  - 世代あたり連合評価数は共有キャッシュで `7 + 4×非代表候補数` = 19（K=2）。キャッシュは世代内のみ（`CoalitionEvaluator` を毎世代再生成）。LLM呼び出しは R=1・100問で 6,600回/世代、6世代で約4.0×10⁴回。
  - debate 各ラウンドは対話履歴なしの単一ターン呼び出しで、**他者の直前発話のみ提示（自己発話は再提示しない）**。プロンプト文言 "keep or change your previous answer" と文脈の不一致に注意。
  - 適応度セット（MMLU-Pro seed 777・100問）と最終評価（seed 1-3）は**シード分離のみで排他抽出は未実装**。母集合約12,000問のため期待重複 ~4問/500問。厳密非重複には除外フィルタの実装が必要（論文では正直に「シード分離・偶発重複ありうる」と記述）。
  - SC@9 と debate（既定 rounds=1 → 6生成/問）の関係は「SC@9 がベースライン有利の保守的計算量マッチ」と明記。
- **発見したコードのギャップ（未修正・要対応）**: `scripts/run_eval.py` の `--task` argparse choices が `["gsm8k", "mmlu", "arc_challenge"]` のままで `tasks.py` の TASK_LOADERS（mmlu_pro/math500/supergpqa 含む、"mmlu" は存在しない）と不整合。battery 設定は `--task mmlu_pro` 等を渡すため**現状のままでは最終評価が argparse エラーで走らない**。choices を `sorted(TASK_LOADERS)` に修正すべき。

### 2026-07-03: 修論 第1章・第2章ドラフト起草
- `docs/thesis/01_introduction.md`（序論、本文約5,000字）と `docs/thesis/02_related_work.md`（関連研究、本文約10,000字）を新規作成。
- 素材は `docs/research_design.md` と `docs/literature_notes.md` のみに限定。数値は文献ノート記載のもののみ使用し、進行中の実験結果には一切言及していない。
- 第2章の構成: 2.1 MAD（原典→批判→成功条件、4B級で素のdebateが機能しない可能性の緊張関係を明示）/ 2.2 マージ理論（交差項問題・KnOTS）/ 2.3 進化的最適化（表2.1: Sakana→LoraHub→Model Swarms→GENOME→EvoPref→PopuLoRA 対比）/ 2.4 適応度設計理論（Shapley・共進化・QD・fitness sharing）/ 2.5 位置づけ（research_design §2 の表を発展させた表2.2）。
- **要フォロー**: 2026年文献（PopuLoRA 2605.16727 / EvoPref 2605.09777 / EvoMAS 2602.06511 / Meta-Team 2605.29790 / Cost of Consensus 2605.00914 / Demystifying MAD 2601.19921）の著者名・正式タイトルは暫定表記。最終稿前に原典確認と novelty 再検索を行い、参考文献リストと表2.2を更新すること。

### 2026-07-03: 研究設計の全面確定と実験基盤の再構築（docs/research_design.md 参照）
- **旧評価系の廃止**: `evaluate_debate.py` の fitness（投票スコア=confidence 0.6固定+引用数+回答文字数）は学術的に無効と判断。2025-11-01 の gen0→gen1「改善」は回答長の増加が主因であり、以後この結果は主張に使わない。
- **適応度の再設計（理論的根拠つき）**: fitness = 代表チーム文脈での**厳密Shapley値**（3エージェント=全7連合を実測、近似不要）× fitness sharing（Goldberg & Richardson 1987、乗法ペナルティ）。多様性の加重「加算」はQD文献（MAP-Elites）とアンサンブル統一理論（Wood+ JMLR23）の双方が批判するため不採用。理論的骨格は協調的共進化（Potter & De Jong 1994）+ PBT。
- **ベンチマーク変更**: GSM8KはQwen3-4Bで飽和(80-92%)→スモーク専用。主要評価は **MMLU-Pro(500問)/MATH-500 L4-5/SuperGPQA(300問)**。GPQA-DiamondはHFゲート付きで不採用。**SC@9（計算量マッチ）ベースラインが査読上必須**（Smit+ ICML24）。統計は Miller(2411.00640) 準拠: paired t + bootstrap + McNemar + Holm補正、K=3 seed。
- **LoRA交叉の理論的修正**: 従来の A/B 行列別補間は交差項 B1A2+B2A1 が混入（KnOTS 2410.19735）→ **ΔW空間ブレンド+ランダム化SVD再分解**を主方式に実装（`delta_blend_lora`）。naive方式はアブレーションA3として保持。テストで最良rank-r近似との一致を検証済み。
- **新実装**: `src/evalx/`（tasks/client/debate/shapley/stats）、`src/evolve/loop.py`（協調的共進化）、`scripts/run_eval.py`（solo/team/coalitions/scモード）、`scripts/run_evolution.py`。テスト41件 `tests/` に整備。`train_lora_persona.py` はT4向けfp16フォールバック+use_cache競合修正。
- **クラウド基盤**: Artifact Registry `evo-swarm`(us-central1) 作成。Vertex AIクォータ実測: **Spot A100×8 (us-central1) / Spot T4×1**。T4はvLLM Multi-LoRA非対応(CC8.0要件)のため**評価/進化はA100、学習のみT4**。イメージ2種（trainer / eval=vLLM v0.8.5ベース）を `cloud/cloudbuild.yaml` でビルド。ジョブ投入は `scripts/cloud/submit_job.sh`。
- **novelty check結論**: 「議論のチームレベル適応度×LoRA重み集団の世代交代進化」の交点は未報告（2026-07-03時点）。要引用の近接研究: GENOME(2503.01155)/PopuLoRA(2605.16727)/MAPoRL(2502.18439)/Model Swarms(2410.11163)/EvoMAS(2602.06511)。執筆直前に再検索すること。

### 2026-07-03: ペルソナB SFTデータセット生成（60例）
- `data/sft_persona_b.jsonl` を全面書き換え（旧2例を破棄）。ペルソナB=実務的意思決定者（結論先行・概算検証・コスト/実現可能性重視）。
- 内訳: 英語算数文章題20（GSM8K風、末尾 `ANSWER: <数値>`）/ 英語4択10（`ANSWER: <A-D>`）/ 日本語意思決定タスク15 / 日本語議論応答10 / 英語フェルミ推定5（`ANSWER: <数値>`）。全例で system prompt 固定。
- 検証済み: 厳密なJSONL 60行・3メッセージ構造・算数/4択/フェルミの全答を独立再計算で一致確認・userプロンプト重複なし・英語応答は約100-360語。生成/検証スクリプトはセッションscratchpadで実行（リポジトリには含めず）。

### 2026-07-03: MAD・マルチエージェント自己改善の文献調査（詳細レポートはセッション出力参照）
- **最重要関連研究**: (1) Subramaniam+ "Multiagent Finetuning" (ICLR2025, arXiv:2501.05707) = debate由来データで生成/批評エージェントを分化SFT、Phi-3(4B)でMATH 58.8→66.0%(5反復)。(2) Feng+ "Model Swarms" (ICML2025, arXiv:2410.11163) = LoRAエキスパート群をPSOで重み空間探索(+最大21%)。(3) PopuLoRA (arXiv:2605.16727, 2026-05) = LoRA個体群の突然変異・交叉による共進化self-play。→「SFT自己改善 vs 重み空間進化」という差別化はModel Swarms/PopuLoRAの存在で単独では成立しない。**チームレベル適応度（討論での協調・新規性）を目的関数にした進化**が本研究の差別化点になる。
- **警告（4Bクラスdebate）**: 7-8B級の同質debateは単体CoT/Self-Consistencyに勝てない・むしろ悪化する報告多数（arXiv:2502.08788, 2509.05396, 2605.00914）。原因は追従(sycophancy)・合意崩壊。対策: 多様な初期解＋confidence条件付き更新（arXiv:2601.19921）、投票中心の集約（arXiv:2508.17536, 2502.19130: 推論タスクは投票が+13.2%優位、議論ラウンド増は逆効果）。→ run_debate_local.py は「多数決＋confidence重み付け、ラウンド数少なめ」が文献的に正当。
- 小型モデルでもdebateトレースによるpost-training(MACA, arXiv:2509.15172)でGSM8K +27.6%等の報告あり。「弱いモデルほどdebateの矯正効果が大きい」(MADC: Qwen-3Bで+8.8%)と「弱いモデルが強いモデルを汚染する」(arXiv:2509.05396)が併存 → ペルソナ多様性の設計と集約方式が成否を分ける。

### 2026-07-03: GCP環境構築と要件調査
- **環境構築**: 上記「クラウド環境の現状」の通り、プロジェクト選定〜課金リンク〜API有効化〜バケット・予算アラート作成まで完了。研究本体（学習ジョブ）は未実行。
- **要件調査の結論**: 必要な外部認証は GCP(ADC) のみ。HFトークン不要（Qwen3-4Bはゲートなし）、OpenAI APIキー不要（`openai` パッケージはローカルvLLMへの互換クライアント、`api_key="EMPTY"`）、W&B完全未使用（実験記録は Vertex Experiments + `results/` のJSON）。
- **既知のギャップ（未修正）**:
  1. `google-cloud-storage` が `pyproject.toml` に未明記（`google-cloud-aiplatform` の推移的依存頼み）
  2. `scripts/evaluate_debate.py` と `data/debate_topics.json` が未コミット
  3. `.env.example` が存在しない
  4. README/AGENTS.md に Vertex 手順の記載がなかった（本更新で一部解消）
  5. README等の「PyTorch 2.9 / Transformers 4.51」表記と pyproject の `torch>=2.6.0` / Dockerfile の `torch==2.6.0+cu124` が不整合
  6. `train_lora_persona.py` の `gradient_checkpointing=True` が `use_cache=True` と競合しうる（実行時警告の可能性）
- **未コミットの変更内容**: `train_all_personas_vertex.py` にGCSアップロード機能追加、`train_lora_persona.py` にBitsAndBytes 4bit・CPUフォールバック・新TRL API対応、`run_debate_local.py` に投票のJSONパース+confidence処理、Dockerfile に venv PATH 追加。

---

## 最後に
全コマンド実行が許可されています。A/B 試験、追加データ収集、LangGraph への拡張など、新しい試みは `docs/` や Issue にメモを残しながら進めてください。必要に応じて README と本ガイドを更新し、後続のエージェントがスムーズに引き継げる状態を維持してください。
