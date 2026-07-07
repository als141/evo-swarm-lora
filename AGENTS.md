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

### 2026-07-07: 進捗スライド v2 を作成・納品（ユーザー改訂指示を反映）
- **成果物**: `slides/20260709_progress_v2/`（git push済み fb9e9d1）。全38枚（本編27＋付録3＋章扉/表紙/結び）、pptx＋PDF、全ページ口語ノート付き。PDF送付済み。v1（`slides/20260709_progress/`）は比較用に残置。
- **ユーザー改訂指示と対応**:
  1. **曖昧表現の具体化**: 「有名な研究」→ Duら（2023, GSM8K 77→85/MMLU 64→71）、Liangら（2023, 思考の退化）、Liら（2024, More Agents）、Smitら（ICML 2024）、「Talk isn’t Cheap」（2025）、Zhangら（2025）を出典チップ付きで明記。
  2. **フォント変更**: 源暎ゴシックP → **源暎エムゴv2（GenEiMGothic2）**。5ウェイト（Regular/Medium/Bold/Heavy/Black）を階層で使い分け（表紙・章扉=Black／スライド見出し=Heavy／小見出し・強調=Bold／本文=Regular/Medium）。`~/.local/share/fonts/genei-mgothic2/`（okoneya GenEiMGothic_v2.0.zip）。ファミリー名は「GenEi M Gothic v2 <weight>」。合成bold禁止（各ウェイトが独立ファミリーのため、deck_helpers.pyの `_apply_run`/`_BOLD_UP` でウェイト名に写像）。
  3. **実験条件ページを新設**: ①ベース単体 ②素の議論 ③多数決9回(SC@9) ④LoRAチーム(本命) を各カードで説明。
  4. **処方前後の比較を削除**: 旧チーム(c5)を全図・全文から除去。最終結果は**新環境3条件（ベース/SC@9/LoRAチーム=c7）**のみ。数値は不変（72.7/81.8/43.1・74.0/87.3/48.6・71.6/86.7/43.1）。
  5. **進化を正直に**（ユーザー選択「試みと分かったこと」）: 発見②「進化は“強化”ではなく“修復”だった」を新図 fig_evolution（進化寄与 MMLU−0.1/MATH+4.3/SGPQA−0.8pt、ΔW向きのみ回転）で提示。「最良チームは進化でなくリプレイで仕上げた」と明記。
  6. **処方＝リプレイ＋議論の工夫を手法に組込み**（ユーザーが中身を要望）: 手法⑤「性格を教えると賢さが削れる→リプレイ（復習）で補う」、手法⑥「議論を頑丈にする3工夫（匿名化/条件つき更新/重み付き投票）」。“処方”という語は前面に出さず“チームの鍛え方”として説明。
- **重要な事実整理（ユーザーの誤解を解いた）**: **新チーム(c7)=最良は「進化後」ではなくリプレイ再学習で作ったもの**。進化を経たのは成績の振るわない旧チーム(c5)。ユーザーは当初「進化後チーム」と認識していたが、実態（進化＝修復にとどまる／c7はリプレイ産）を説明の上、看板の進化は正直に「修復だった」と位置づける方針で合意。
- **失敗デモを差し替え**（一貫性）: 旧＝暗号復号先(10429, run001旧チーム) → 新＝**「夫婦が長続きする理由＝親友」(mmlupro-test-6410, run002新チームg3)**。正解「親友」が2票→R2で実務/探索が「“親友”は価値観一致の結果、より根本は“意見が合う”」という深読み(追従)で全滅。成功例(カントール8070)と同じ新チームのログで統一。
- **技術基盤の追加**: `deck_helpers.py` にウェイト写像（`FONT_BLACK/HEAVY/BOLD/MED/REG`、`_apply_run`）。`make_figures.py` は FP_BOLD/HEAVY を FontProperties で図中強調に使用、fig_final は3条件化、fig_evolution 新規。ビルド/PDF/検品手順は v1 と同じ（LibreOffice直叩き＋pdftoppm）。

### 2026-07-07: 研究進捗報告スライド（2026-07-09 発表用）を作成・納品
- **成果物**: `slides/20260709_progress/`（git管理・push済み aca8492）。本編26枚＋付録3枚のpptx（編集可能）とPDF、全ページに口語スピーカーノート付き。ユーザーへPDF送付済み。
- **狙い（ユーザー指示）**: ガチガチの進捗報告でなく「LLM同士が会話しているのがリアルタイムで分かるデモ」を主役に。専門用語は噛み砕く（LoRA/ファインチューニングは既知前提）。AIっぽくないデザイン・文章。費用情報は一切入れない。源暎ゴシック系フォント使用可。
- **デモ素材＝実会話ログ**（results/gcs の transcripts から発掘）: (1)成功例 `mmlupro-test-8070`（カントール集合、run002新チーム g3）＝実務のみ級数検算で正解H、批判/探索は誤答J→議論R2で2体がHに訂正し**少数派の正解が多数派を動かす**。(2)失敗例 `mmlupro-test-10429`（暗号の復号先、run001 demo）＝正解「受信側」が2票→R2で批判/探索が「WhatsAppは端末で復号」等の**深読み(追従)で不正解に転落**。検証可能領域で議論が効き/困難領域で崩れる、を実ログで体現。3ペルソナのsystem promptは付録に原文掲載。
- **構成**: 表紙／導入（一言＋見どころ）／章扉①背景（大→集団, Du議論で改善, 3つの批判=計算量マッチ/追従/同質性, 本研究の問い）／章扉②手法（全体像図, 3性格, Shapley=「抜けたら困る度」, 進化=交叉/変異/選択）／章扉③デモ（成功4枚+失敗3枚のビルドアップ）／章扉④実験結果（設定, 領域依存反転, 能力毀損と修復, 最終結果棒＋対戦判定）／章扉⑤考察結論（なぜ勝てない/どこで勝つ, 進化=修復＋評価環境の罠, 結論4点, 今後）／付録（数値表, ペルソナ原文, 用語辞典）。
- **技術基盤（後続の再利用可）**: 図=matplotlib（`make_figures.py`, 源暎ゴシックP登録・フラット学術配色）。スライド=python-pptx（`deck_helpers.py`にフォントea/latin/cs全指定・角丸カード・会話bubble・章扉・脚注ヘルパー、`build_deck.py`が本体）。PDF化=LibreOffice AppImage直叩き（`~/apps/squashfs-root/opt/libreoffice26.2/program/soffice`、AppRunラッパーは出力されないので不可、要 `HOME=/home/als0028`）。検品=`render.sh`（pptx→PDF→pdftoppmでページ別PNG）。源暎ゴシックP/Nは `~/.local/share/fonts/genei/`（okoneya.jp GenEiGothicP-1.1.zip）。
- **数値は新環境統一の最終値のみ使用**（final_stats_clean.py準拠）: ベース72.7/81.8/43.1・SC@9 74.0/87.3/48.6・旧チーム68.5/79.3/42.0・新チーム71.6/86.7/43.1。主張=処方+3.2pt有意/SC@9に−3.0pt有意負け/ベース互角/数学のみSC@9と互角。**負けも正直に記載**（方法論的貢献として提示）。領域依存Δ（実験1・旧環境内）=素の議論 MMLU−4.1/MATH+4.0/SGPQA+3.1pt。CoT圧縮=2,423→1,188字・MATH solo 83→70%（run001同一環境）。

### 2026-07-07: 【研究一式の納品完了】修論最終化・全文整合・データ完全ローカル化・GitHub push 済み
- **表紙確定**: 舛田 岳（「増田 学」は誤りだった）/ 新潟大学 自然科学研究科 電気情報工学専攻 情報社会デザイン科学コース / 学籍番号 F25C142E。残TODOは「年度」と「指導教員の職位確認」の2箇所のみ（ユーザー確認事項）。
- **全文整合性検査（専任エージェント）**: 24件の指摘を全件修正——貢献3点→4点、A1/A2/A3の「検証する」→「実装済み・未実施」への時制統一（Sec2/Sec3）、+3.5→+3.2pt統一、生成設定の進化ループ/最終評価分離明記、RQ表記の廃止、副次評価（LLM-as-judge/多様性推移）の未実施明記、ΔWノルム比1.00と26問flipをSec4.6.2へ正式記載、旧チーム呼称の初出定義、進化ループ生成条件のlimitation追加ほか。**主要数値の照合は全一致**（旧稿数値の誤残存なし）。
- **refer.bib**: 全63エントリの実在・書誌を原典照合済み（別エントリ参照）。本文側もSec2表2.2の「Meta-Team」→実題「Evolve as a Team」、Sec3のjudge2025bottleneck引用文を実論文内容（意思決定者の力）に整合。
- **最終コンパイル**: platex→pbibtex→platex×2→dvipdfmx でLaTeXエラー0・bibエラー0・未解決参照0、main.pdf 535KB。
- **データ完全ローカル化（GCS不要化）**: `results/gcs/` = 全実験データミラー（評価JSON・生会話・LLM完全記録・進化ログ・リプレイ、gzip圧縮812MB→199MB、git管理）+ データカタログREADME（環境系統差の注意付き）。`artifacts_local/` = 全LoRAアダプタ実体5.7GB（gen0/進化全世代/run002、git外）。
- **GitHub push完了**（origin/master、5コミット）: ①評価・学習基盤のコード変更 ②設計文書docs ③修論一式 ④実験データアーカイブ+分析スクリプト ⑤AGENTS.md+.gitignore。LaTeXビルド生成物はuntrack化。
- **コスト最終実測**（ユーザー提供コンソール値）: 総額¥39,879、無料クレジット残¥4,071。**残クレジットが少ないため、今後クラウド作業を再開する場合は課金アカウント付け替え（別アカウントのBilling Account User権限付与→`gcloud billing projects link research-501308 --billing-account=<新ID>`）を先に行うこと**。データは全てローカルにあるためGCSが消えても研究は無傷。
- **残タスク**: 表紙の年度・指導教員職位（ユーザー確認）のみ。研究の余地（ハイブリッドSC@3×3体、進化v3、A1アブレーション）はSec6.2と research_design_v3.md に保存済み。

### 2026-07-06: refer.bib 全63エントリの書誌全数点検が完了（実在未確認ゼロ・削除ゼロ）
- 全エントリを arXiv abs ページ / ACM DL / ACL Anthology / ICLR proceedings と照合。**63件すべて実在確認**。修正済み bib 全文は点検エージェントの報告として返却（refer.bib への反映は親セッションで実施）。citation key は \cite 互換のため全て維持。
- **プレースホルダ解決**: judge2025bottleneck (arXiv:2511.11040) の実体は「Key Decision-Makers in Multi-Agent Debates: Who Holds the Power?」(Qian Zhang+ 2025)。MADC・Truth Last の提案論文で、文献調査ログの「MADC: Qwen-3Bで+8.8%」と同一論文。
- **タイトル大幅相違の修正**: zhang2025debate は改題済み「Stop Overvaluing Multi-Agent Debate — We Must Rethink Evaluation and Embrace Model Heterogeneity」/ cost2026consensus「…: Isolated Self-Correction Prevails Over Unguided Homogeneous Multi-Agent Debate」/ demystifying2026mad「…: The Role of Confidence and Diversity」(Zhu+) / maca2025preference「Self-Improvement of Language Models by Post-Training on Multi-Agent Debate」(Samanta+) / PopuLoRA「Co-Evolving LLM Populations for Reasoning Self-Play」(Creus Castanyer+) / EvoPref「Multi-Objective Evolutionary Optimization Discovers Diverse LLM Alignments Beyond Gradient Descent」(Guo+、LoRAは題名に含まれない) / EvoMAS「Evolutionary Generation of Multi-Agent Systems」(Hu+, ICML 2026、team-level fitness は題名に含まれない) / metateam2026meta の実体は「Evolve as a Team: Collaborative Self-Evolution for LLM-based Multi-Agent Systems」(Hao+、Meta-Teamという論文名ではない)。
- **著者・venue の確定**: Smit, Andries P.→Andries（P.は誤り）/ Minut, S.→Adrian Robert / GENOME の第一著者 Zhang, W.→Zhang, Yiqun / SuperGPQA 著者は「M-A-P Team and Xinrun Du and others」/ talk2025cheap=Wynn, Satija, Hadfield (ICML 2025 MAS WS) / choi2025debate=NeurIPS 2025 Spotlight / wang2024mixture=ICLR 2025 採録 / yang2024model=ACM Computing Surveys 2026 採録 (DOI: 10.1145/3787849) / survey2025selfevolving=TMLR 2026、正式題「…: What, When, How, and Where to Evolve on the Path to Artificial Super Intelligence」(Gao, Huan-ang+)。
- **⚠️要フォロー**: Sec2 の本文・表2.2 で旧仮タイトル（「Meta-Team: Meta-Optimization of Agent Team Configurations」「EvoMAS: … with Team-Level Fitness」「EvoPref: … LoRA Optimization」「If Multi-Agent Debate Is the Answer…」等）に基づく紹介文があれば、実タイトル・実内容に合わせて修正すること。

### 2026-07-06: 最終化フェーズ（ユーザー指示: 現結果で確定、ハイブリッドは今後の課題へ）
- **ユーザー決定**: ハイブリッド実験(SC@3×3体=9生成)は実施せず「今後の課題」としてSec6.2冒頭に追記（本研究実測に基づく最有力の残された設計として記述）。
- **表紙修正**: 著者「舛田 岳」（増田学は誤り）、所属「新潟大学 自然科学研究科 電気情報工学専攻 情報社会デザイン科学コース」、学籍番号 F25C142E。
- **参考文献の全数実在確認**（エージェント実行中）: 63エントリ、「著者情報要確認」10件+タイトル不明1件(judge2025bottleneck)を重点点検。確認できないものは削除・平文化。**修論全文の整合性検査**も別エージェントで並行（数値矛盾・章参照・TODO残存・Sec2/Sec3の時制）。
- **研究データの完全ローカル化**: `results/gcs/` = GCS実験データのミラー（評価JSON全量・生会話transcripts・LLM完全記録llm_calls・リプレイ・進化ログ・パイロット。jsonl/大JSONはgzip圧縮で812MB→199MB、git管理）+ `results/gcs/README.md` = データカタログ（環境系統差の注意含む）。`artifacts_local/` = 全LoRAアダプタ実体~5.7GB（gen0/進化全世代/run002、.gitignore対象、ローカルのみ）。**GCS不要で全データ参照可能な状態が完成**。.gitignoreにLaTeXビルド生成物とartifacts_local/を追加。
- 残り: bib反映→再コンパイル→git整理コミット→push。

### 2026-07-06 深夜: 【研究の最終決着】新環境統一の全再測定完了 — 主結果=「処方を尽くしてもSC@9に有意負け(−3.0pt)」+「処方の効果は本物(+3.2pt)」
- **再測定26エントリ完走**（c2 SC@9・c1ベース・c5旧チーム各3シード、新イメージ統一。ジョブA=1297897618877186048/B=2496981019664580608）。**全条件が同一評価環境で揃った初のクリーン比較**（scripts/analysis/final_stats_clean.py）。
- **最終精度表**: c1ベース mmlu 0.727/math 0.818/sgpqa 0.431（3シード）、**c2 SC@9 0.740/0.873/0.486**（6シード）、c5旧チーム 0.685/0.793/0.420（3シード）、c7新チーム 0.716/0.867/0.431（6シード）。
- **主検定（6シード6,000問）: c7 vs SC@9 = −2.98pt、p<10⁻⁶ — 有意負けが確定**（ベンチ別: mmlu −2.4pt有意/math −0.7pt ns/sgpqa −5.4pt有意）。昨日の「+1.3pt(p=0.053)」や「MMLU+5.5pt有意勝ち」は全てイメージ系統差の産物だった。
- **副検定（3シード3,000問、Holm後有意）**: **c7 vs c5 = +3.20pt (p<10⁻⁵) — 処方（リプレイ+conditional+匿名化+weighted）の効果は本物**（mmlu+3.7/math+6.3/sgpqa+0.2）。c7 vs c1 = +0.27pt ns（**チームはベース水準まで完治**、mathのみ+3.8pt有意）。c5 vs c1 = −2.93pt有意（v1チームは実はベース以下だった）。c1 vs c2 = −3.10pt有意。
- **研究の最終結論**: (1) 4B級LoRAペルソナ・チーム議論は、能力修復・集約改善・プロトコル改良を尽くしても計算量マッチSC@9に勝てない（Smit+ ICML24の批判を最も統制された形で確認した負の結果）。(2) 診断に基づく処方は統計的に本物の+3.2ptを与え、壊れたチームをベース水準へ完治させた。(3) MATH-500ではチームがベースを+3.8pt有意に上回り、検証可能領域での議論の価値は維持。(4) 測定環境の系統差（イメージ間+4〜6pt、ベース素生成に強く効きLoRA付きにはほぼ効かない）の発見と統制は方法論的貢献。
- **コスト実測**（ユーザー提供のコンソール値）: 総額¥39,879（Compute Engine ¥31,642 + AI Platform ¥7,857 + Build/Storage/AR ¥381）、無料クレジット残 ¥4,071。**教訓: Vertex AI Custom JobのGPU実効レートはSpot VM単価の約2倍（A100で~$2.2/h）** — ジョブ時間×Spot単価の見積もりは半分に過小評価する。
- 残作業: 修論の結論最終化（Sec4.6新環境値差し替え+系統差小節、Sec5.4/5.5、Sec6、abstract）→コンパイル→完成。

### 2026-07-06 朝: 【重大】評価イメージ間の系統差を検出 — v1（旧イメージ）とv2（新イメージ）の測定は直接比較不能
- **経緯**: 頑健性確認(seed4-6)でc7vsSC@9の+1.3pt(p=0.053)が消失(5シードで−0.68pt)。ただしc7は6シードフラットなのにSC@9だけs4-6で+7pt跳ねる不自然さ→唯一の系統差=SC@9のみ新旧イメージ跨ぎ(s1-3=7/4版、s4-6=7/5版)。
- **切り分け検証**: c2_sc9_mmlu_pro_s1を新イメージで再実行(recheckジョブ6277445043586334720)。**同一問題402問で旧0.716 vs 新0.776 = +6.0ptの系統差が確定**(予測一致率86.6%=生成自体が変化)。問題セット同一・生成設定同一・vLLM本体v0.8.5同一を検証済み→イメージ再ビルド時の依存ライブラリ差(Dockerfile.evalの追加pipはdatasets/httpxのみだが、ベースイメージ内の何かが変動)が原因と推定。**vLLMはseed固定でも実行環境・並列状態で非決定的**(同一設定で12-14%の予測が変わる)ことも実証。
- **影響**: v1全測定(c1-c6、旧イメージ)とc7(新イメージ)の比較は全て汚染。「c7が旧チームに+3.5pt有意勝ち」も大部分が環境差の疑い。**同一イメージ内の比較のみ有効**: 新環境同士(c7 vs c2 s4-6)ではc7が負けている(−3.7pt級)。
- **教訓(最重要)**: (1) インフラ(イメージ)更新を跨ぐ測定は比較不能。イメージ更新時は主要ベースラインの再測定をセットにする。(2) 頑健性確認(追加シード)が系統誤差を検出した——これを省いていたら誤った結論(+1.3pt優位)で修論を書いていた。(3) 中間暫定値の報告は問題順の偏りにも注意(259問時点+11.5ptは誇張、同一問題比較では+4.2pt)。
- **対処方針**: 主要条件を新イメージで再測定し全比較を同一環境に統一する(**ユーザー承認済み: 「主要3条件を再測定」**): c2 SC@9×3ベンチ×s1-3(mmlu s1はrecheck流用のため8)+c1 ベースsolo×9+c5 進化後チーム×9(v1プロトコル忠実再現=standard/多数決/ラベルあり)=26エントリ。**投入済み: ジョブA(c2+c1)=1297897618877186048 / ジョブB(c5)=2496981019664580608**、出力先 experiments/run002/remeasure_v1/。完走~5-7時間(SC@9律速)。再測定後: c7vsC2は6シード新環境(s1-3再測+s4-6既存)、c7vsc5/c1は3シード新環境で最終統計→修論結論の最終化。

### 2026-07-06 未明: 修論の結果・考察・結論・概要を執筆完了（コンパイル検証済み）— 認証再失効中のオフライン作業
- **認証が再失効**（07-05朝のログインから約17時間、組織の定期reauthポリシー）。頑健性ジョブ2本はクラウド側で継続中（失効前の観測: c7側3/9・c2側1/9完了）。認証復旧待ちのバックグラウンド監視を起動し、オフラインで修論を執筆。
- **thesis/Sec4.tex**: §4.6実験結果を全面執筆——実験1（7条件×3ベンチ確定表+検定+損失分解表oracle付き）と実験2（G0-G3ゲート+c7最終表+主検定p=0.053、6シード差し替えTODOコメント付き）の二段構成。§4.3比較条件表を実態に合わせ更新（A1未実施を正直に記載し実験2条件7‡を追加）。§4.4統計設計をMcNemar exact中心+頑健性追加+開発/評価分離の実態に更新。
- **thesis/Sec5.tex 考察を全面執筆**: 5.1 2軸トレードオフ（多様性利得×能力毀損）による領域依存性の統一説明+マルチンゲール定理への領域依存反例 / 5.2 損失分解→処方の因果（CoT圧縮2423→1188字の機構、リプレイのパレート改善、集約損失の未回収部分も正直に） / 5.3 進化=選抜でなく修復（ΔWノルム比1.00、勝者の呪い、リプレイが上位互換） / 5.4 SC@9比較の最終形 / 5.5 限界（p=0.053の扱い、A1未実施、外的妥当性）。
- **thesis/Sec6.tex 結論**（4つの知見+今後の課題5項目）と **abstract.tex 概要**（目的+手法+実験1/2の結果+貢献3点）を執筆。
- **コンパイル検証**: platex→pbibtex→platex×2→dvipdfmx でエラー0・未定義参照0・main.pdf生成成功。
- 残り: 6シード最終値の差し替え（TODOコメント箇所）、Sec1-3の整合点検（研究課題の記述と結論の対応）、参考文献の追補（Debate or Vote等の新規引用は現状本文で平文言及）、図表（進化推移図等）の検討。

### 2026-07-05: 【最終決着 — c7全9点確定・本検定完了】旧全条件に有意勝ち、SC@9には+1.3pt優位だがp=0.053のボーダー
- **c7（run002チーム）3シード確定**: MMLU-Pro 0.754/0.706/0.706（平均0.722）/ MATH-500 0.855/0.855/0.860（0.857）/ SuperGPQA 0.430/0.413/0.423（0.422）。全エントリNone率0%。
- **主検定（プール3000問 McNemar exact）: c7 vs SC@9 = +1.3pt（212 vs 173）、p=0.0527** — 点推定は優位だが両側5%にわずかに届かず（Holm後も同様）。**昨夜のc5（−2.2pt、p=0.004有意負け）からの統計的逆転は c7vsc5=+3.5pt p<1e-5 で確定**。
- **ベンチ別主検定**: MMLU-Pro **+5.5pt p<1e-5 有意勝ち**（154vs72） / MATH-500 −0.5pt p=0.68 同等 / SuperGPQA **−4.4pt p=0.0008 有意負け**（48vs88）。領域構造の最終形=「知識推論はチームが強い・数学は同等・院級科学は超大量サンプリングが強い」。
- **副検定（プール、Holm後すべて有意）**: c7 vs 旧チームc5 **+3.5pt** / vs ベース **+4.3pt** / vs 素の議論 **+4.6pt** / vs プロンプトペルソナ **+3.5pt**（全てp<1e-4）。**処方（リプレイ修復+conditional+匿名化+weighted集約）の効果はv1の全手法に対して統計的に確定**。
- **解釈と選択肢**: 主検定p=0.053は効果量+1.3ptに対する3000問の検出力不足。(a) 頑健性確認としてseed4-6を追加（+3000問、$10-15、一晩）— 同効果量ならp≈0.01級で決着。optional stopping批判を避けるため修論には「3シード計画+頑健性追加」を透明に記載する。(b) 現状で修論へ—主張は「旧手法から有意改善・SC@9と同等以上（点推定+1.3pt優位、p=0.053）」。→ユーザーに提示。
- 統計スクリプト: scripts/analysis/final_stats_c7.py に恒久化済み。
- **頑健性確認を投入（(a)を採用）**: c7 seed4-6 = **4233074695550795776** / SC@9 seed4-6 = **8224108390331187200**（各9エントリ、計+3000問で6シード6000問プールに拡張。出力先 experiments/run002/robust_c7/ と robust_c2/）。同効果量+1.3ptが維持されればp≈0.005級で決着。**修論には「主実験3シード（p=0.053）+事前計画を明記した頑健性追加3シード」として透明に記載**（optional stopping批判への対応: 追加は効果の消失/維持の確認が目的であり、6シード合算と3シード単独の両方を報告する）。完走見込み: 明朝（SC@9側は9生成/問で重め）。

### 2026-07-05: 【G3 seed1で2ベンチ確定・SGPQA暫定も優勢 — 総合合格見込み】最終評価c7(seed2,3)投入・進化v3はスキップ決定
- **G3確定値（seed1、v1と同一問題セット）**: MMLU-Pro **0.754**（500問None0。v1 c5 0.714比+4.0pt / SC@9 0.688比**+6.6pt**）/ MATH-500 **0.855**（200問None0。SC@9 0.855と同値・素の議論0.860に−0.5pt=同等）/ SuperGPQA **確定0.430**（300問None0。**暫定0.512@121は前半上振れで確定は大幅下**。SC@9 0.457比−2.7pt負け、ただしv1 c5 0.387比+4.3pt改善でG2 seed9の0.433-0.440と一貫）。**G3総合=プールで合格**: seed1プール1000問で run002チーム 0.677 vs SC@9 0.652 = **+2.5pt**（MMLU大勝が牽引、MATH同等、SGPQA小負け）。「全ベンチ同等以上」は未達（SGPQA−2.7pt）だが主検定のプール優位はseed1で実測。教訓: 暫定値の報告は50%完了以降に限る（121/300のSE±4.5ptを軽視した）。
- **run002チーム（c7）の構成確定**: リプレイ入りアダプタ（rank16/2ep/lr1e-4/seq8192）× conditional議論 × 発言匿名化 × logprob重み付き投票（weighted）。チーム6生成/問 < SC@9の9生成。
- **最終評価投入 = 6608512392756199424**（c7のseed2,3×3ベンチ=6エントリ、出力先 experiments/run002/final_c7/）。seed1はG3の値を流用（同一設定・同一プロトコル）。完走後: final_stats.pyへc7を追加し、**主検定=プール3000問でc7vsSC@9**（McNemar+Holm）、副=ベンチ別・c7vsc5・c7vsc1。
- **進化v3はスキップ決定**（理由: v1実測で進化の寄与は「SFT毀損の修復」のみ=MATH+4.3pt/他±0。リプレイが同じ修復をより安価・確実に果たした今、進化の限界利益は薄く、時間リスク（3世代2-3h+選抜ノイズ）が利益を上回る。修論では「進化=重み平均化による損傷修復機構。処方としてはリプレイSFTが上位互換」と整理し、v1の進化実験は機構解明の実験として位置づける）。ユーザーへ報告済みの方針に沿い、反対があれば再開可能（進化v3設計はresearch_design_v3.md §3.4に保存済み）。

### 2026-07-05: 【G1全合格 — 能力毀損は完治、一部ベース超え】G2集約ablation投入
- **G1最終確定（seed9・200問・solo）**: MMLU-Pro 0.695/0.695/0.665（基準0.63✅）/ **MATH-500 0.775/0.845/0.840**（基準0.75✅、run001の0.645-0.685から+13〜16pt修復、pragmatist 0.845はベース0.830超え）/ **SuperGPQA 0.440/0.405/0.425**（基準0.37✅、run001の0.327-0.343から**+7〜11pt修復**、critic 0.440はベース0.436同等以上）。None率≤4.5%健全。**リプレイSFT（36例38%混合+rank16/2ep/lr1e-4/seq8192）は3ベンチ全てで毀損を完治させ、ペルソナ維持と能力保持の両立に成功** — 設計v3の第一柱P1が実測立証。
- **G2投入 = 1614970384048717824**（run002チームでSuperGPQA 150問 seed9、conditional+匿名化を共通とし集約のみ{majority/weighted/genselect+judge=ベースモデル}の3エントリ、新evalイメージ=LLM完全記録d1618f64入り）。判定基準: weighted/genselect ≥ majority+3ptで新集約採用。**このジョブからllm_calls/に全入出力が記録される**。
- 次段: G2集計→G3（チーム1シード: MMLU-Pro500+SGPQA300+MATH200、G2最良集約）→合格なら進化v3→最終評価。
- **【G2確定】** SuperGPQA 150問 seed9・run002チーム・conditional+匿名化: majority **0.433** / **weighted 0.440（最良・採用）** / genselect 0.433（いずれもNone 0）。集約の上積みは+0.7ptと小さい（仮説: soloが修復され議論後の割れ自体が減少+割れた問題は4B judgeでも選びきれない難問）。ただし絶対値は run001チーム s1 0.387 → **0.433-0.440 = +4.6〜5.3pt** で、SC@9 s1 0.457 に−1.7ptまで接近。genselect の +3pt 基準は未達（採点型でなく比較選択でも4Bには荷が重い可能性、修論では「集約はweighted採用・genselectは効果なし」を正直に報告）。
- **【G3投入 = 5639851444778565632】** run002チーム×weighted集約×conditional×匿名化、seed1の3ベンチ本番（MMLU-Pro 500 / MATH 200 / SGPQA 300、--save-transcripts付き、v1確定値と直接比較可能）。判定基準: MMLU-Pro vs SC@9 s1 0.688・vs c4 s1 0.708 / MATH vs 素の議論 s1 / SGPQA vs SC@9 s1 0.457。所要~2-3h。

### 2026-07-05: 【G0合格】リプレイ生成完走 → 学習系列長バグ発見・修正 → run002準備完了
- **リプレイ生成**: T4版はSpotプリエンプト頻発+再確保待ちで進捗せずキャンセル → **A100v2版（8032727396400496640）が~15分で完走**。旧A100版失敗はvLLM起動直後の即死でフレークと判断（同構成で本実験が多数回正常起動）。**T4でもvLLM cu124はfp16・LoRAなしなら起動可**という新知見（compat libs経由、「T4はvLLM不可」は Multi-LoRA に限る話に訂正）。
- **G0検収 全合格**: 36例（math L5×24=geometry16/int.algebra5/precalc3 + MMLU-Pro val×12）、CoT平均6,503字・中央値4,794字（ベース自然長2,423字超の長導出）、全例正解検証済み・重複なし。
- **【重大バグ発見・修正】TRL SFTConfig.max_length 未指定（既定1024）**: このままではリプレイ長CoT（実測トークン長 最長7,435、4096超が8例）が**学習時に1024で切り捨てられ、リプレイの意味が消滅**するところだった。run001の学習も1024で切られていた（旧データは779字≈短いので実害なし）。修正: train_lora_persona.py に `--max-seq-len`（既定8192、SFTConfig.max_length へ）、train_all_personas_vertex.py に中継引数を追加。**trainerイメージ再ビルド**（cloud/cloudbuild.trainer-only.yaml 新設、build 8f516062）。教訓: フレームワークの暗黙デフォルト（特に切り捨て系）はデータの性質が変わるたびに再点検する。
- **run002データ**: make_run002_datasets.py で 3ペルソナ×96例（既存60+リプレイ36、ペルソナsystem付与、重複除外、構造検証）を合成し gs://evo-swarm-lora-usc1-research-501308/experiments/run002/data/ へアップ済み。
- **run002学習ジョブ**: T4版5144244379535278080は**OOMで失敗**（8192トークン系列の最初のステップで6.26GiB追加確保不可。T4 14.6GiBでは4bit+rank16+grad checkpointingでも長系列が乗らない）→ **A100版 1054131492952735744 を再投入**（長CoTを切らない方針を優先、コスト差+$1、YAML=$SP/train_run002_a100.yaml）。教訓: max-seq-len を上げる変更はGPUメモリ再見積とセット。アダプタ出力先: gs://.../experiments/run002/training/model/adapters/persona_{a,b,c}。次段=G1 solo点検（3ベンチ200問、検収 MATH≥0.75/SGPQA≥0.37/MMLU≥0.63）。
- **evalイメージ再ビルド起動**（build 1b98a4b8）: G2集約ablation用に新実装（--aggregation weighted/genselect、--anonymize-debate、--judge-model、chat_scored logprob confidence）を焼き込む。G1と並行で仕上げて時間節約。
- **run002 A100版学習 SUCCEEDED**（~30分で3ペルソナ完走、アダプタ3点確認済み）。evalイメージビルドもSUCCESS。**G1 solo点検バッテリー投入 = 6391460001340194816**（run002の3アダプタ×3ベンチ200問solo、seed9、configs/g1_battery.json、出力先 experiments/run002/g1_solo_check/）。検収基準: MATH≥0.75 / SGPQA≥0.37 / MMLU≥0.63（200問SE±3.4pt考慮で−2pt緩め判定可）。
- **【G1中間: リプレイ修復は本物】** MMLU-Pro solo 確定 0.695/0.695/0.665（基準0.63クリア、run001同等以上）。**MATH-500 solo（暫定、explorer は157/200時点）= critic 0.775 / pragmatist 0.845 / explorer 0.892** — run001の0.645-0.685から**+13〜+22ptの大修復**、基準0.75を全員クリア、pragmatist/explorerはベースc1(0.830, seed1-3)を超えている可能性。seed差(9 vs 1-3)による±数ptの誤差はあるが+13pt超はseedでは説明不能。criticのNone率4.5%はやや高め（許容内）。SuperGPQA solo は実行待ち。**数値シナリオの前提（毀損の修復可能性）が最初の実測で裏付けられた**。
- **【ユーザー指示: LLM入出力の完全記録】** client.py に `_CallLogger` 実装 — 環境変数 `EVALX_LOG_DIR` 指定時、**全LLM呼び出しの入力メッセージ（プロンプト/コンテキスト全文）+生成全文+confidence+設定** を calls_<pid>_<秒>.jsonl へ追記（ProgressCache と同じ gcsfuse 実績方式、プロセス=batteryエントリ単位でファイル分割、スレッドセーフ）。submit_job.sh の battery/evolution ブランチに EVALX_LOG_DIR=<出力先>/llm_calls を追加 → **G2以降の全ジョブで生入出力が一次資産として GCS に残る**。テスト2件追加（全58件通過）。evalイメージ再々ビルド=d1618f64。既存資産の透明性: v1のチーム議論はデモ60問+診断120レコードの transcripts あり、solo/SC の生発話は記録前で無し（per_item の予測・正誤のみ）。分析スクリプト3本（final_stats.py 等）を scripts/analysis/ へ恒久化（scratchpad は一時領域のため）。

### 2026-07-05: 設計v3の確信度評価（ユーザー確認済み）とv3実験のスケジュール
- **ユーザーへの回答として明示した確信度の内訳（後続セッションはこの判断枠組みを引き継ぐこと）**:
  - ① リプレイによる能力修復（G1）= **確信度 高（~8割）**。根拠: 因果の鎖が全て実測（SFTデータ779字→生会話でCoT 2423→1180字圧縮を直接確認→MATH L5に劣化集中）。「壊した原因の除去」であり新規の魔法ではない。
  - ② 集約改善（G2）= **確信度 中〜高（~6-7割）**。根拠: oracle 0.517 vs チーム0.387（回収すべき正解はチーム内に実在）+ MATHで議論集約がoracle到達済み（0.813/0.815）という自データ内の成功例。不確実要素: GenSelectが4B・訓練なしでどこまで効くか→G2 ablationで先に測る。
  - ③ プールでSC@9有意超え = **五分五分より少し上（55-60%）**。リスク: リプレイがペルソナ多様性を薄めチームゲインが減る副作用（実測するまで不明）。
  - 設計思想: 「絶対上がる」と言い切る代わりに**ゲートの階段（G1: MATH solo≥0.75等 / G2: 集約+3pt / G3: SC@9同等以上）で、上がらない変更は本採用させない構造**にした。届かなくても各処方の損失削減量の測定自体が修論の柱。
- **v3実験スケジュール（07-05 10:15時点の見積）**: G0リプレイ生成=進行中（A100、間もなく完走）→ run002学習=本日昼投入・~14時完了 → G1 solo点検=本日夕方 → G2集約ablation=本日夜 → G3チーム確認=本日深夜 → （合格時）進化v3=今夜〜明朝 → 最終評価バッテリー=07-06 → 統計確定=**07-06夜〜07-07**（Spot待ちバッファ込み）。コスト$26-31/残予算~$140。

### 2026-07-05: 【v3本実験 全63エントリ完走・最終確定】総合はSC@9に−2.2pt有意負け、パイロットA/B完了、リプレイT4版稼働中
- **最終精度表（3シード平均）**: MMLU-Pro: c4 **0.684** > c5 0.683 > c2 0.667 > c6 0.661 > c1 0.639 > c3p 0.624 > c3 0.598 / MATH-500: c3 **0.870** > c2 0.862 > c3p 0.860 > c1 0.830 > c5 0.813 > c4 0.770 > c6 0.698 / SuperGPQA: c2 **0.467** = c3 0.467 ≈ c3p 0.466 > c1 0.436 > c4 0.407 > c5 0.399 > c6 0.364。**seed1で見えた「c3p首位0.487」は3シードで消滅（シードノイズ）**——SGPQAでは SC@9=素の議論=プロンプトペルソナが同着+3pt。
- **プール3000問の検定（McNemar exact+Holm）**: **c5 vs SC@9 = −2.2pt p=0.0041（Holm後も有意）→ 総合で計算量マッチに有意負けが確定**。ベンチ別: MMLU-Pro +1.6pt(ns) / MATH −4.8pt(p=6e-5) / SGPQA −6.8pt(p=1e-5)。c5 vsベース +0.8pt(ns)=ベース同等。進化寄与(c5vsc4)プール+0.6pt(ns)、MATHのみ+4.3pt有意修復。gen0vsベース プール+0.2pt(ns)（MMLU+4.5有意/MATH−6.0有意/SGPQA−2.9 p=0.065）。素の議論vsベース: MATH+4.0有意/SGPQA+3.1有意/MMLU−4.1有意悪化→プール±0。**領域依存の三すくみ構造が3シードで最終確定**。
- **SGPQAの構図修正**: gen0チームはSGPQAでベースとほぼ同等（−2.9pt ns）まで戻していた（solo毀損−7.2ptをチーム化+4.3ptでほぼ相殺）。MATHは毀損−13ptが大きすぎてチーム化でも未達。進化の修復はMATHのみで発現。
- **パイロットA/B（MMLU-Pro 300問 seed555）**: conditional 0.690 vs standard 0.680（+1.0pt, p=0.72 ns、flip 17vs14）。**有意ではないが方向は正・悪化なし** → run002では文献裏付け（訂正方向バイアス介入 2508.17536）+無害性からconditional採用。ただし単独では+1pt級の小効果、集約改善（weighted/genselect）との併用が本命。
- **ジョブ状態**: 本実験3本SUCCEEDED（計63エントリ）/ パイロットA/B・診断SUCCEEDED（transcripts_math500_diag/に生会話40問×3条件あり、分析はこれから）/ **リプレイA100版FAILED**（原因未調査）・**T4版RUNNING**（entrypoint迂回のfp16構成が機能、replay/はまだ空=生成中）。
- 統計スクリプト: $SP/v3check/final_stats.py（7条件×3ベンチ精度表+6ペア検定ベンチ別/プール+Holm+パイロット判定を一括出力）。

### 2026-07-05: 設計v3の集約実装完了（logprob重み付き投票+GenSelect裁定+匿名化）— 全56テスト通過
- **client.py**: `ChatResult`+`chat_scored()` 追加 — vLLMのlogprobsからmean/tail confidence（幾何平均確率、tail=末尾64トークン）を計算。既存`chat()`はラッパー化で挙動不変。
- **debate.py**: (1) `weighted_vote()` logprob重み付き投票（confidence None は重み1.0） (2) `genselect_adjudicate()` 候補を匿名・順序ランダム化して1コンテキスト比較選択、`BEST: <n>`解析、失敗時None（呼び出し側でフォールバック） (3) `_debate_user_prompt(anonymize=, shuffle_seed=)` 発言者ラベル匿名化+提示順シャッフル (4) `run_debate(aggregation=, anonymize=, judge_model=)` — genselectは票が割れた問題のみ裁定発動・失敗時weightedへ。**既定値は全てv3プロトコル同一動作**（後方互換: chatのみ実装のFakeClientでも動くようwant_conf時のみchat_scored使用）。DebateRecordにconfidences/aggregation/adjudicated追加。
- **run_eval.py**: `--aggregation {majority,weighted,genselect}` / `--anonymize-debate` / `--judge-model` 追加。progressキャッシュは集約方式でラベル分離（従来キャッシュと衝突しない）。transcriptsにconfidences/adjudicated記録。
- テスト6件追加（匿名化のラベル置換・weighted_voteの重み論理・genselectの選択解析とシャッフル整合・解析不能時None）→ **tests/ 全56件通過**。ruff/compileall通過。
- これでG2集約ablation（150問×{majority, weighted, genselect}×同一チーム）がジョブ投入だけで実行可能。
- **インフラ待ち**: gcloud OAuth失効でユーザーの `gcloud auth login` 待ち（Vertexジョブはサーバ側で継続中）。認証復旧後: 本実験最終確認(前回50/63)→リプレイレース処理→run002投入。

### 2026-07-05: 再設計向け文献調査（5テーマ・約50本）完了 → 設計v3確定版（docs/research_design_v3.md + literature_notes_v3.md）
- (1) **集約損失対策** = GenSelect型裁定（2507.17797 / 2602.09341: 多数決比+5pt絶対、多数決が割れた問題のみ起動・匿名化・順序ランダム化必須）+ logprob重み付き投票（DeepConf 2508.15260 / CISC 2502.06233）。**verbalized confidenceは4Bで80-100%に飽和し不可**（2502.11028/2604.01457）。4Bは採点型judge不可・比較選択型なら可（2606.19544）。
- (2) **RSA（2509.26626）が同一モデルQwen3-4B-Instruct-2507で集約ループの大幅改善を実証**（SuperGPQA含む）— 集約損失13pt回収の直接的裏付け。debate round1はRSAの1ステップと同型。
- (3) **P2自己生成リプレイは文献最善筋と同型**（2506.09428/2505.13811/SDFT 2402.13669）。比率より「ベース自身の生成分布か」が支配的。38%混合は妥当レンジ。rank16/2epも裏付け（LoRA忘却はrank・学習量と共に増、LR感度支配: 2405.09673）。
- (4) **選抜ノイズ対策** = IRT/識別力厳選の適応度セット（tinyBenchmarks 2402.14992: 厳選100問で誤差2%）+ successive halving（2404.00069）+ 同点持ち越し（oracle選抜の正バイアスは構造的: 2606.26836）。実装は c1実測per_itemで正解率30-70%帯を優先する簡易IRT近似。
- (5) **sycophancy対策** = 発言匿名化が訓練不要で有効（2510.07517: sycophancy>self-bias、匿名化で均等化）。条件付き更新は「訂正方向バイアス介入」（Debate or Vote 2508.17536, NeurIPS25 Spotlight: 議論単体はマルチンゲール=期待改善ゼロの定理）の実装形として正当。
- (6) **修論の論述素材**: MATH-500の素の議論+4.0pt有意は「マルチンゲール定理への領域依存の反例」（検証可能領域では更新の無情報性が破れる）。novelty: チームShapley適応度×LoRA集団進化は依然未報告、最近接=ACPO 2602.09341（要引用対比）。DARE drop率は0.3-0.4が安全（DELLA 2406.11617最適0.4）、マージは2-3個維持（利得は凹: 2505.21226）。
- **設計v3確定**: §3.2集約（匿名化+conditional+logprob投票+GenSelect裁定、チーム≈6.4生成<SC@9の9生成）、§3.4進化（識別力厳選120問+MI理論裏付き多様性項+SH+DARE0.3-0.4）、公平比較にSC@9+CISC / SC@7+GenSelect対照を追加。文献約50本はliterature_notes_v3.mdに整理（2026年IDは最終稿前に原典再確認要）。
- インフラ: セッションAPI上限（6時リセット）で夜間停止が発生。gcloud OAuthも失効しユーザーに再ログイン依頼中（認証待ちバックグラウンド監視あり）。scripts/make_run002_datasets.py 作成済み（ペルソナ60例+リプレイ36例合成、lint/compile通過）。

### 2026-07-05: 【設計v3起草】損失3分解の発見 — 敗因は「能力毀損」と「集約の失敗」に分解可能、docs/research_design_v3.md 作成
- **ユーザー指示**: 「ゼロから何が足りなかったか洗い出し、議論FW・進化・評価・ペルソナまで再設計したv3を、絶対に精度を上げる自信が持てるまで詰める」→ 法医学的分析+実装点検+文献調査（エージェント並行実行中）でv3起草。
- **決定的な新分析（oracle上限=3体中1体でも正解）**: SuperGPQA s1で **oracle 0.517 vs 実測チーム 0.387 = 集約損失13.0pt**（毀損した3体ですらベース0.407を+11pt超える正解保有率）。MMLU-Pro: oracle 0.800 vs 0.714 = 8.6pt。**MATH: 0.815 vs 0.813 = ほぼゼロ（検証可能領域では議論が完璧に集約）**。議論1ラウンド自体の寄与は+1.2〜2.4ptのみ（多数決→議論後）。→ チーム誤差 = L1能力毀損 + L2集約損失 + L3探索損失 に分解。多数決は「正解が少数派」で構造的に負ける。
- **難易度層別（SuperGPQA）**: 素の議論は **hard帯+17.3pt**（0.493 vs c1 0.320）、easy/middle+2〜4pt。議論の価値はベース精度が低い帯域に集中。c5はhardで0.293と最弱=毀損がhardで最も痛い。ペアflip: c3p正解・c5誤答48問 vs 逆18問。
- **実装点検の確認事実**: (1) ペルソナ=日本語1文の性格プロンプト、SFT応答は平均779字の短文のみ→8192トークン級長CoT能力を短文で上書き=毀損の直接原因 (2) confidence抽出はrun_debate_local.pyに実装済みだがevalx評価系に未接続 (3) 進化演算子はα∈[0.3,0.7]内挿+小変異のみ=親の凸包の内側しか探索不可（ΔWノルム1.00不変の「方向回転」と整合） (4) rank32/alpha64/lr2e-4/3ep は60例に過剰。
- **docs/research_design_v3.md 起草**: 原理=P1能力アンカー(リプレイ+rank16/lr1e-4/2ep) P2検証型集約(confidence表明+**judge集約**=3体の解をベースモデルが裁定、チーム7生成<SC@9の9生成) P3戦略ペルソナ(導出/検証/具体例、run003ゲート付き) P4進化v3(hard層化120問再サンプル適応度+誤り相関ペナルティ+successive halving+DARE/TIES外挿) P5実測ゲート階段G0-G5。**数値シナリオは全て実測量の再結合**: 保守的仮定（毀損半減+集約損失1/3回収）でプール0.66-0.67 vs SC@9 0.643=+2〜3pt有意圏。コスト$26-31。対照にSC@7+judge(=8生成)を追加し「judge利得がチーム固有か」を切り分ける設計。
- 修論は二段構え: 実験1=v1の診断（領域依存・損失分解・勝者の呪い・追従実例）、実験2=v3処方の因果検証（各ゲートの合否として報告）。

### 2026-07-05 深夜点検: SuperGPQA中間値 — LoRAチームに逆風・プロンプトペルソナが首位、リプレイをT4で先行実行へ
- **SuperGPQA seed1（300問、確定+暫定）**: **c3'プロンプトペルソナ 0.487（確定・首位）** > c3素の議論 0.467（確定） > c2 SC@9 0.461（280問暫定） > c1ベース 0.407（確定） > **c5進化後チーム 0.381（160問暫定・ベース以下）** > c6進化後solo(critic) 0.343（300問分暫定、None5）。
- **読み**: SuperGPQAはMATH型の「能力律速」領域。ペルソナSFT毀損 −6.4pt（c6 0.343 vs c1 0.407）はチーム化の +3.8pt では埋まらない。一方 **c3'が+8.0ptで首位 = MMLU-Proと真逆**（MMLU-Proではc3'はチームに−5.9pt有意負け）。→ 領域構造は「検証可能性」1軸でなく **「多様性利得」×「能力毀損被害」の2軸トレードオフ**として整理すべき。MMLU-Pro=利得大/被害小→LoRAペルソナ勝ち、MATH=利得中/被害大→素の議論勝ち、SuperGPQA=利得大/被害大→**プロンプトペルソナ勝ち**（毀損ゼロで多様性だけ買う）。実用処方: 領域に応じてペルソナの実装方式を切替、または重みペルソナ+能力リプレイ（P2）で両取り。
- **プール影響試算**: このままc5のSuperGPQAが0.38-0.40なら3000問プールでSC@9に約−2pt級で逆転される公算。**SC@9超えの成否はP2リプレイの成功にほぼ収斂**（MATH/SuperGPQAの毀損修復が必須条件）。
- **アクション**: リプレイ生成はベースモデルのみでLoRA不要 → **T4版ジョブを自作YAML（entrypoint迂回・fp16・LoRAなし・max-model-len 16384）で投入 = 9073445938416058368**。A100版 5943936778602807296 はT4版RUNNING確認後にキャンセル（同一出力パス二重書き込み防止）。A100 Spot供給が本実験3本で飽和しPENDING滞留のため、T4の潤沢な供給を使う判断。eval イメージ(vLLM cu124)のT4動作は初試行（compatライブラリ頼み。失敗しても数分でFAILED・コスト微小）。
- 進捗: final_eval3 45/63完了（MATH-500とMMLU-Proは全42確定済み。残り18は全てSuperGPQA）。本実験3ジョブRUNNING健全（progress最終更新=点検時刻）。パイロットA/B・診断はA100待ちPENDING継続。

### 2026-07-05: MATH-500全確定・プール統計・機構分析 — 領域依存性が本研究の中心構造に
- **MATH-500最終確定（3シード600問対）**: c3素の議論 **0.870** > c2 SC@9 0.862 > c3'プロンプト 0.860 > c1ベース 0.830 > **c5進化後 0.813** > c4 gen0 0.770 > c6進化後solo 0.698。検定: **進化の修復 +4.3pt p=0.0007**（3シード再現+4.5/+4.5/+4.0）/ 進化後vsベース −1.7pt p=0.27（**ベース水準まで完治**）/ 素の議論vsベース +4.0pt p<0.0001（数学では議論自体が有効）。
- **2ベンチ統合プール（2100問対）**: SC@9 0.723 ≥ **c5 0.720**（p=0.83引き分け）> c4 0.709 > c1 0.694 > c3' 0.691 > c3 0.676（最下位）。c5vsc3素 +4.5pt p=1e-5 / c5vsベース +2.7pt p=0.005。**「素の方が良い」は全体では偽**（MATH-500限定の現象）。本当の強敵はSC@9で現状完全タイ。
- **中心的発見=領域依存性**: 議論の律速要因が (a)検証可能領域(MATH)=個体能力律速→素の議論が効きペルソナSFTの能力毀損が致命傷 (b)検証困難領域(MMLU-Pro)=多様性律速→重み焼き込みペルソナのみ有効(素は有意悪化)。ペルソナSFT=「能力を代償に多様性を買う取引」で、進化はこの**取引条件をパレート改善**（MATH+4.3pt修復・MMLU-Proコストゼロ）。矛盾する先行研究(議論は効く/効かない)を検証可能性の軸で整理できる。
- **機構検証**: gen0と進化後アダプタのΔW Frobeniusノルムは全役割で完全一致(比1.00)→進化は大きさでなく**方向を回転**させ、交叉平均がSFT損傷成分を打ち消したと解釈（ノルム保存の相対スケール変異+ブレンドの帰結）。
- **SuperGPQA開始**: c1ベース0.407（公表42.8と整合・None2.3%=健全）。SC@9との決着はc4/c5のSuperGPQA値次第。
- **投入済みジョブID**: 診断(math500生会話40問×3条件)=8710272849715134464 / パイロットA/B(conditional vs standard、300問seed555)=8938267580850765824 / リプレイ生成=5943936778602807296（いずれもSpot空き待ち→本実験完走後に自動起動）。
- **SC@9超えの定量的道筋**: 必要マージン≈プール+1.5-2pt。P2リプレイ成功でMATHチーム0.86-0.87なら+1.3pt、P1条件付き更新でMMLU-Proの壊し26問半減なら+0.7pt、合算で有意圏。SuperGPQAがMMLU-Pro型ならさらに押す。
- **未コミット変更**: client.py(timeout環境変数化600s) / debate.py(ANSWER_FORMAT_MATH+DEBATE_UPDATE_INSTRUCTIONS 2スタイル) / tasks.py(二重ANSWER prefix除去) / run_eval.py(--debate-style) / make_replay_data.py(新規) / tests(+11件) / docs/research_design_v2.md(新規)。standard文言はテストで不変性保証。

### 2026-07-04: MMLU-Pro本検定の結果と法医学的分析 → v2はパイロット主導に転換
- **本検定（1500問対、McNemar exact+Holm）**: ✅有意: LoRAチームvsベース +4.4pt (p_adj=0.001) / vsプロンプトペルソナ +5.9pt (p<0.0001) / 素の議論の悪化 −4.1pt (p=0.0035)。❌非有意: **チームvsSC@9 +1.6pt (p_adj≈0.34, CI −0.6〜+3.7)** / 進化寄与 ±0 (p=1.0)。→「底上げは本物、計算量マッチ超えは未達」が正確な現在地。SC@9超えには約+1ptの上積みが必要（不一致ペア数から必要マージン≈2.2pt）。
- **法医学的分析の発見**:
  1. c4vsc5は予測一致80%・flip95対94で完全相殺 → 進化は「動いたが選抜が無情報」。
  2. 進化ログ実測: 同一個体の測定チーム精度が世代間で±4-7pt揺れる（適応度100問のSE±4.9pt）。dev軌跡0.56→0.68は**勝者の呪い**（毎世代4候補×3役割の最大値選抜がノイズ上振れを拾う）。1-2pt差の選抜は統計的に不可能（必要n≈2600問/判断）。
  3. MATH-500劣化はLevel5(0.82→0.55)とIntermediate Algebra/Precalculus/Geometryに集中 = 長い導出能力の選択的破壊（SFTの短いCoTスタイル刷り込みが原因の疑い。診断ジョブで生テキスト確認中）。
  4. 議論が正解を壊す実例確認（mmlupro-test-3450: 正解Dの2対1多数がround1で崩壊しF採用。追従による多数派崩れ）。500問統計では+44/−26。
- **対策の優先順位（パイロット→本採用のゲート方式）**: (P1) 条件付き更新プロンプト（自分の誤りを特定できた場合のみ変更可）で−26問型を半減狙い → **--debate-style conditional 実装済み・A/B 300問(seed555)投入**。(P2) 能力保持リプレイ: ベースモデル自己生成の正解長CoT（MATH訓練L4-5優先分野厚め+MMLU-Pro val）を混ぜ rank16/2ep 再学習 → **scripts/make_replay_data.py 実装済み**。(P3) 進化v2は「大差候補の生成（DARE大変異）+successive halving（60問足切り→300問確認）+3世代で検証+2pt未達なら中止」のゲート付き。standardスタイルの文言不変性は回帰テストで保証（v3再現性維持）。

### 2026-07-04: 設計v2の策定（docs/research_design_v2.md）— 汎化する進化への再設計
- **v3実測の診断**: D1 ペルソナSFTがMATH-500のsolo能力を破壊（0.84→0.65-0.69、リプレイなし60例/rank32/3epが原因）。D2 進化が固定100問MMLU-Pro適応度に過適合（dev+12pt→test±0）。D3 全問debateは高コスト（round0で64%は3体一致）。
- **v2の5変更**: (A) 能力リプレイ30例混合+rank16/2ep+MATHガード40問 (B) マルチタスク適応度120問を世代ごと層化再サンプル (C) 検証150問で世代選択（val最良世代を採用） (D) 突然変異をDARE式ドロップ+リスケールに、交叉にTIES符号合意オプション (E) round0全員一致なら議論スキップ+confidence重み付き投票 → 世代数10・候補3/役割へ増強。
- **文献裏付け**: DARE 2311.03099 / TIES 2306.01708 / MERGE³ 2502.10436（縮小セット適応度）/ SID 2510.06843（議論early-exit -40%トークン）/ ConfMAD系 2509.16839 / LoRA-Loop 2507.13568（リプレイ）。
- **コスト見積**: リプレイ生成$1-2 + 再学習<$1 + 進化v2 $6-8 + 最終評価27エントリ$15-20 = **計$25-30**。c1/c2/c3/c3pはベース条件でv3の値を再利用可。
- **修論構成**: v3=実験1（チーム効果実証+故障診断）、v2=実験2（処方の効果測定）。v2の上積みが小さくても因果分析の研究として成立する二段構え。

### 2026-07-04: 【重大バグ】MATH-500の回答形式指示がletter用に落ちていた → 修正・イメージ再ビルド・再測定
- **症状**: v3のmath500が全条件で精度5-7%（base soloで0.075。このモデルなら60%前後のはず）。None率0%なのに不正解ばかり。per_item点検で予測が 'c' や 'answer:202' など。
- **原因**: `debate.py` の `answer_format_instruction()` が number/letter の二分岐で、**answer_type="math" が letter に落ち「ANSWER: <letter>」と指示**していた。数学問題でモデルが選択肢文字を答える。副次バグとして「ANSWER: ANSWER: X」の二重prefix時に letter 抽出が単語 ANSWER の 'A' を拾う事故も発見。
- **修正**: (1) ANSWER_FORMAT_MATH（"ANSWER: <final simplified answer>"）を追加し明示的な3分岐+未知型はValueError。(2) extract_answer で tail の二重 "ANSWER:" prefix を剥がす。(3) 回帰テスト6件追加（tests/test_evalx.py、計45件全通過）。
- **影響範囲**: **math500の全測定が無効**（進化はMMLU-Proのみ・letter型のGSM8K/MMLU-Pro/SuperGPQAとnumber型は無影響）。v1のmath500も同バグを含んでいた（そもそもv1は全体無効）。3バッテリージョブをユーザー承認の上キャンセル、evalイメージ再ビルド（build af9b340a）→ 汚染math500成果物（final_eval3のc1_base_solo_math500_s1.jsonとprogress/*math500*）を隔離 → 再投入。再投入ジョブは新イメージ（EVALX_HTTP_TIMEOUT既定600s）でタイムアウト失敗9件も自動回収される。
- **MMLU-Pro最終確定（500問×3シード平均）**: c1 base 0.639 / c2 SC@9 ~0.671(暫定) / c3 素の議論 ~0.598(暫定) / c3' ~0.627(暫定) / **c4 gen0チーム 0.684** / **c5 進化後チーム 0.683** / c6 進化後solo最良 0.661。**c4≈c5でMMLU-Proの進化寄与はゼロ〜僅少（開発セット+12ptは過適合の疑い濃厚）**。ただしチーム化効果（c4/c5 > c2 SC@9 > c1 > c3）は3シードで一貫し堅い。
- 教訓: 「None率だけでなく、**既知の公表値との絶対水準の乖離**も健全性チェックに含める」（今回None率0%で一見健全に見えた）。answer_typeのような列挙分岐は else に落とさず明示網羅+未知型エラーにする。

### 2026-07-04: v3最終評価の中間結果（MMLU-Pro）と180秒タイムアウト問題
- **MMLU-Pro seed1 確定値（500問・None率≤0.4%）**: c1 base solo **0.660** / c4 gen0チーム **0.708** / **c5 進化後チーム 0.714** / c6 進化後solo 0.652-0.698（最良critic 0.698）。暫定（残り数問）: c2 SC@9 0.688 / c3 素の議論 0.639 / c3' プロンプトペルソナ 0.655。
- **読み**: (1) c5がSC@9を+2.6pt上回り首位、(2) チーム0.714 > 最良個体0.698（協調ゲイン実在）、(3) 素の議論はベース単体より悪化=文献どおり、(4) **⚠️進化寄与 c5−c4 は+0.6ptと小さい**（開発セット+12ptに対し汎化は僅か）。seed2暫定: c1 0.618 / c2 0.659 / c3 0.585 / c3' 0.595 / c5 0.662 — seed2は全体に難しく c5≈c2。**「進化の上積み」はシード・ベンチ平均と検定次第**。「LoRAチーム化で+5pt・SC@9同等以上」は堅い。
- **【障害】OpenAIクライアント timeout=180s が高並列時に不足**: 32並列でGPU飽和中の最難問（8192トークン級生成）が180s×3リトライを使い切り `RuntimeError: Request timed out` → parallel_mapが例外伝播しrun_eval.pyごと落ちる。エントリ末尾（490/500前後）で failed が5件（c2_s1/c2_s2/c3_s1/c3_s2/c3p_s1）。**進捗キャッシュは残る**ため、3ジョブ完了後に同一設定でモップアップジョブを1本再投入すれば残り数問だけ低負荷再実行で完走できる（並走はprogressキャッシュ書き込み競合リスクがあるので不可）。恒久対策: client.py のタイムアウトを `EVALX_HTTP_TIMEOUT`（既定600s）で環境変数化済み（次回イメージビルドから有効）。
- 教訓: 「バッテリー式評価はエントリ失敗を握りつぶして先へ進む設計のため、battery_summary.json と完了JSON数の突合を定期点検に含める」こと。

### 2026-07-04: 【重要】最終評価v1は無効（max_tokens=512切り捨て）→ Qwen公式仕様準拠のv3で再測定
- **発見**: v1最終評価でベース系条件の回答抽出失敗率44-84% vs LoRAチーム5-30%と非対称。原因はmax_tokens=512でベースモデルの長いCoTが切り捨てられ回答行に到達しないこと（LoRA群は短い回答+ANSWER形式をSFT済みで影響小）。**v1の比較結果（base 0.27 vs LoRA team 0.64等）は使用禁止**。
- **Qwen3-4B-Instruct-2507公式仕様の確認結果**: ネイティブコンテキスト262,144 / **推奨出力長16,384** / 推奨サンプリング temp0.7・top_p0.8・top_k20・min_p0。旧設定はtop_p0.9・top_k未設定で非準拠だった。
- **v3設定（最終形）**: max_tokens=8192、VLLM_MAX_MODEL_LEN=32768（debateは他者解全文埋め込みのため）、top_p=0.8、top_k=20（vLLM拡張extra_body経由）。回答抽出もmarkdown太字耐性を追加。出力先はfinal_eval3/（旧キャッシュ汚染回避）。
- **limitation（修論に明記）**: 進化ループ自体は512トークン設定で実行済み。進化内部は全個体がLoRA（None率5-6%）のため公正だが、進化時と最終評価時の生成条件が異なる。
- 教訓: 「新しい実験系では、まず抽出失敗率(None率)の条件間非対称性を必ず確認する」— 精度差より先に見るべき健全性指標。

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
