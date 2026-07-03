# 第4章 実験設定

本章では、第3章の提案手法を検証するための実装、ベンチマークと評価プロトコル、比較条件、統計設計、およびハイパーパラメータを述べる。本章の記述はすべて公開実装（`scripts/train_lora_persona.py`、`scripts/run_evolution.py`、`scripts/run_eval.py`、`scripts/run_eval_battery.py`、`scripts/cloud/`）と一致する。未実施の実験結果には言及しない。

## 4.1 実装

### 4.1.1 ベースモデルと QLoRA 学習

ベースモデルには **Qwen3-4B-Instruct-2507**（Apache-2.0、Hugging Face 上でゲートなし公開）を用いる。4B 級を選ぶ理由は、（i）同規模帯では「素の debate が性能を悪化させうる」ことが報告されており [Zhang+ 2025; arXiv:2605.00914]、チームレベル選択圧が debate を機能させるかという RQ の検証に適した困難条件であること、（ii）公表ベンチマーク値（MMLU-Pro 69.6、SuperGPQA 42.8）に改善余地が残ること、（iii）進化ループで多数の LoRA を単一 GPU 上に同時サービングできる計算規模であることによる。

世代 0 の 3 個体（critic / pragmatist / explorer）は、役割別 SFT データ（各 60 会話例、JSONL 形式、チャットテンプレート適用）を用いた **QLoRA** [Dettmers+ 2023] で学習する。設定は以下のとおりである。

- **量子化**: ベースモデルを 4bit NF4（double quantization 有効）でロードし、計算 dtype は GPU に応じて bfloat16（A100 等）または float16（T4、compute capability 7.5 のため bf16 非対応）へ自動フォールバックする。
- **LoRA 構成**: rank $r = 32$、$\alpha_{\mathrm{LoRA}} = 2r = 64$、dropout 0.05、対象モジュールは全層の `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` の 7 種。
- **最適化**: 学習率 $2 \times 10^{-4}$、エポック数 3（Vertex AI ジョブ投入設定。学習スクリプト単体の既定値は 1）、デバイスあたりバッチサイズ 1、勾配累積 16、`paged_adamw_32bit`、gradient checkpointing 有効（学習時は KV キャッシュを無効化して競合を回避）、packing 無効。
- **再現性**: グローバルシード 1234（Python / NumPy / PyTorch / CUDA を統一固定）。学習コマンド・シード・ハードウェア構成は `metadata.json` として成果物に同梱する。

各役割のペルソナは、SFT により重み空間に焼き込まれると同時に、推論時にも短い役割記述（例: 批判的検証者「厳密な検証を重視する批判的思考家。反証・例外・境界条件に敏感」）をシステムプロンプトとして与える。

### 4.1.2 vLLM Multi-LoRA サービング

全推論（進化ループの連合評価・最終評価とも）は vLLM [Kwon+ 2023] の OpenAI 互換 API を通じて行う。サーバは `--enable-lora`、`--max_loras 8`、`--max_lora_rank 64`、`--max-model-len 32768`、tensor 並列 1 で起動し、ベースモデルを 1 つだけ GPU に常駐させたまま、リクエストの `model` フィールドで LoRA アダプタを問い合わせ単位に切り替える。進化ループで世代ごとに生成される子アダプタは、動的 LoRA ロード API（`/v1/load_lora_adapter`、`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`）でサーバ再起動なしに登録する。

生成パラメータは全条件共通で temperature 0.7、top-p 0.9、最大 512 トークン、シード指定付きとする（エージェント×ラウンド、および Self-Consistency のサンプル添字ごとに決定的にオフセット。第3章 3.2 節）。クライアントは問題単位で 16 並列のリクエストを発行し、vLLM サーバ側の連続バッチングで GPU 利用率を確保する。API 呼び出しは最大 3 回の線形バックオフ付きリトライで保護する。

### 4.1.3 Vertex AI Spot 構成とプリエンプション耐性

計算は GCP Vertex AI Custom Training の **Spot（プリエンプティブル）ジョブ**として実行する（プロジェクトのクォータ上、オンデマンド GPU VM は利用不可）。工程ごとの構成は次のとおり。

| 工程 | マシン構成 | 理由 |
|---|---|---|
| QLoRA 学習（世代 0） | `n1-standard-8` + NVIDIA T4 ×1（Spot） | 4bit 量子化により 6GB 級で学習可能。fp16 フォールバックで対応 |
| 進化ループ / 最終評価 | `a2-highgpu-1g`（A100 40GB ×1、Spot、us-central1） | vLLM の Multi-LoRA カーネル（Punica/Triton）が compute capability ≥ 8.0 を要求し T4 では動作しないため |

成果物（アダプタ・評価 JSON・世代ログ）は GCS バケットに Cloud Storage FUSE マウント（`/gcs/...`）経由で保存する。Spot はいつでもプリエンプトされうるため、すべての評価ランナーは**問題単位の逐次 JSONL キャッシュ**を持ち、再実行時には評価済み問題 ID をスキップして途中から再開する。最終評価は複数条件を 1 ジョブにまとめるバッテリードライバで実行し、完了済みエントリの JSON が存在すれば条件ごとスキップする。進化ループも世代ごとに全ログを JSON へ書き出し、任意の世代から再開可能である。

## 4.2 ベンチマークと評価プロトコル

### 4.2.1 ベンチマーク選定

最終評価は以下の 3 ベンチマークで行う。いずれも Hugging Face 上でゲートなしに取得できる（認証トークン不要の方針を維持する）。

| ベンチマーク | 抽出方法 | 問題数 | 選定根拠 |
|---|---|---|---|
| MMLU-Pro [Wang+ 2024] | test 分割から固定ランダム抽出 | 500 | ベースの公表値 69.6 で飽和しておらず、MAD による改善報告と接続可能。10 択で当て推量の影響が小さい |
| MATH-500 [Lightman+ 2023] Level 4–5 | Level 4 以上に限定して抽出 | 200 | 数学推論は MAD の改善効果が最大と報告される領域。高難度に限定して天井効果を回避 |
| SuperGPQA [Du+ 2025] | 固定ランダム抽出 | 300 | ベースの公表値 42.8 と最も余地が大きい。大学院水準の広分野知識推論 |

GSM8K はベースモデルで 80–92% と飽和しているため配管確認（スモークテスト）専用とし、主張には用いない。GPQA-Diamond はデータセットが HF ゲート付きであるため不採用とする。

### 4.2.2 生成型評価と回答抽出

評価は lm-evaluation-harness 型の選択肢対数尤度方式ではなく、**生成型（generative）評価**で統一する。solo 条件と debate 条件を同一プロトコル・同一抽出ロジックで比較するためには、両者とも自由生成テキストから回答を取り出す必要があるからである。全エージェントに厳密な書式指示

```
Think step by step. Then give your final answer on the last line
in exactly this format:
ANSWER: <letter | number>
```

を与え、出力末尾の `ANSWER:` 行を正規表現で最優先抽出する。`ANSWER:` 行が欠落した場合に限り、文脈条件付きのフォールバック（選択肢タスクでは "answer is (X)" 等のパターン、数値タスクでは最後の数値、数式タスクでは最後の `\boxed{}`）を適用する。数式回答は LaTeX 表記の正規化（`\frac`→`/`、`\sqrt`→`sqrt()`、数値の同値判定 12.0 = 12 等）を経て比較する。抽出不能な出力は無効票（多数決から除外）とし、単独評価では不正解として扱う。**すべての比較条件で同一の抽出・正規化・採点コードを共有**し、条件間の差が抽出ロジックの差に混入しないことを保証する。

### 4.2.3 進化ループ内の適応度セット

進化ループの選択圧（第3章の $v(S)$ 計測）には、MMLU-Pro test 分割（約 12,000 問）から専用シード（777）で抽出した**固定 100 問**を全世代共通で用いる。100 問という規模は、tinyBenchmarks [Polo+ 2024] が示した「キュレートされた 100 例でフルベンチマークのスコアを平均誤差約 2% で推定できる」という結果に基づく。適応度セットは世代間の**順位付け専用**であり、その絶対値は主張に用いない。最終評価とはシード系列を分離して抽出するため（最終評価はシード 1–3）、標本の系統的な共有はないが、母集合が共通のため偶発的な重複は起こりうる（期待重複は 500 問抽出あたり約 4 問、1% 未満）。適応度セットへの選択圧の過適合（進化版の「テストへの学習」）が疑われる場合に備え、最終評価がこの 100 問を含まない形でも集計できるよう問題 ID を全ログに記録する。

## 4.3 比較条件

最終評価は以下の 8 条件で行う（研究設計書 §6.2 の 7 条件に、プロンプトペルソナの寄与を分離する条件 3′ を追加）。全条件で同一の問題集合・同一シード・同一抽出ロジックを用い、問題単位で対応づけ（paired）が可能な設計とする。debate 条件のラウンド数は $R = 1$ に統一する。

| # | 条件 | 構成 | 目的 |
|---|---|---|---|
| 1 | ベース solo CoT | ベースモデル単体、ペルソナなし | 絶対的な基準線 |
| 2 | **Self-Consistency@9** | ベースモデルから 9 サンプル→多数決 | **計算量マッチの非協調ベースライン**。3 エージェント×最大 3 生成（round 0 ＋ 2 ラウンド）＝ 9 生成に相当。既定 $R=1$ では debate 側は 6 生成/問であり、SC@9 はベースライン側に計算量で有利な保守的設定。「MAD は計算量を揃えた SC に勝てない」という批判 [Smit+ 2024] への直接の対処 |
| 3 | ベース×3 温度サンプリング debate | 同一ベースモデル 3 体（ペルソナプロンプトなし、シードオフセットのみ） | サンプリング多様性のみの debate（Self-MoA 型対照） |
| 3′ | ベース×3 プロンプトペルソナ debate | 同一ベースモデル 3 体＋役割プロンプト | プロンプトによるペルソナ付与の寄与を分離。RQ2（重み焼き込み vs プロンプト）の対照 |
| 4 | gen-0 LoRA チーム debate | SFT 直後の 3 ペルソナ LoRA | 進化前のチーム性能。条件 5 との差が進化の寄与 |
| 5 | **進化後チーム debate（主要条件）** | 提案手法（Shapley＋sharing、$G=6$ 世代）の最終代表チーム | 本研究の主張の中核 |
| 6 | 進化後 LoRA solo ×3 | 条件 5 の各個体を単独評価 | チーム効果（協調による上乗せ）と個体改善の分離 |
| 7 | A1: solo 適応度進化チーム debate | 適応度を $v(\{c\})$ に置換して進化させたチーム | **新規性の直接検証**。個体性能のみを選択圧とする GENOME 型 [Zhang+ 2025] との比較であり、チームレベル選択圧の寄与を分離する |

このほか進化ループ側のアブレーションとして、A2（fitness sharing 無効）と A3（交叉を naive な A/B 別補間に置換）を実行フラグとして実装している（第3章参照）。

## 4.4 統計設計

統計処理は Miller [2024] の評価統計 5 原則（標準誤差の併記・クラスタ構造の考慮・リサンプルの扱い・paired 分析・検出力設計）に準拠する。

- **シードと反復**: 主要比較は $K = 3$ シード（1, 2, 3）で反復し、各シードの生成サンプルを**問題内リサンプル**として扱う。すなわち問題 $j$ の条件 $A$ に対するスコアをシード間平均 $\bar{s}_{A,j}$ とし、以後の検定は問題を単位とする。
- **主検定**: 問題単位の**対応あり t 検定**（条件間差分 $\bar{s}_{A,j} - \bar{s}_{B,j}$ の平均がゼロという帰無仮説）を行い、**paired bootstrap**（$B = 10{,}000$、問題単位の再標本化）による 95% 信頼区間を併記する。
- **単一シード比較**: シード反復を伴わない補助比較には、2 値の対応ありデータに適した **McNemar 検定**を用いる [Dror+ 2018]。
- **多重比較補正**: 主要な仮説検定は「条件 5 vs 条件 2」「条件 5 vs 条件 7」の 2 比較 × 3 ベンチマーク = 6 検定であり、**Holm–Bonferroni 法** [Holm 1979] で family-wise error rate を 5% に制御する。それ以外の比較は探索的分析と位置づけ、補正済み p 値を主張に用いない。
- **検出可能効果量**: 対応あり検定（$\alpha = 0.05$、検出力 80%）の目安は +10pt → $N \approx 180$、+7pt → $N \approx 360$、+5pt → $N \approx 700$ である。3 ベンチマーク統合（$N = 500 + 200 + 300 = 1{,}000$ 問、$K=3$ シードによる分散低減を含む）では **+4〜5pt 程度の効果が検出可能**であり、Du らが報告する MAD の効果量（7〜15pt）に対して十分な感度を持つ。
- **再現性の記録**: 全実行についてシード・モデル revision・vLLM バージョン・生成パラメータ・問題 ID リストを結果 JSON に記録する。

副次評価として、議論品質の LLM-as-judge 評価（Gemini 2.5 Flash、pairwise + 提示順スワップ + 引き分け許容、50–100 サンプル）と、世代を通じた行動多様性（回答不一致率）の推移分析を行うが、これらは主要な統計的主張には用いない。

## 4.5 ハイパーパラメータ

以下の値はすべて実装のデフォルト値（コマンドライン引数の既定値、および Vertex ジョブ投入スクリプトの設定値）と一致する。

**表 4.1: QLoRA 学習（世代 0）**

| 項目 | 値 |
|---|---|
| ベースモデル | Qwen/Qwen3-4B-Instruct-2507 |
| 量子化 | 4bit NF4、double quantization、計算 dtype bf16（T4 では fp16） |
| LoRA rank $r$ / $\alpha_{\mathrm{LoRA}}$ / dropout | 32 / 64 / 0.05 |
| 対象モジュール | q, k, v, o, gate, up, down の各射影（全層） |
| 学習率 / エポック | $2 \times 10^{-4}$ / 3（Vertex 投入設定; スクリプト既定 1） |
| バッチサイズ / 勾配累積 | 1 / 16 |
| 最適化器 | paged_adamw_32bit（gradient checkpointing 有効、packing 無効） |
| SFT データ | 役割別 60 会話例 × 3 役割 |
| シード | 1234 |

**表 4.2: 議論・生成**

| 項目 | 値 |
|---|---|
| temperature / top-p / 最大トークン | 0.7 / 0.9 / 512 |
| 議論ラウンド数 $R$ | 1（round 0 の独立回答 ＋ 1 更新ラウンド） |
| 集約 | 最終ラウンド多数決（同数はシード付き乱数、無効票は除外） |
| シードオフセット | エージェント×ラウンド: $\mathrm{seed} \cdot 10^4 + 100i + \rho$; SC サンプル: $\mathrm{seed} \cdot 10^3 + k$ |
| クライアント並列数 / リトライ | 16 / 3 回（線形バックオフ） |

**表 4.3: 進化ループ**

| 項目 | 値 |
|---|---|
| 世代数 $G$ | 6 |
| サブ集団構成 | 3 役割 × $K = 2$（エリート ＋ 子）= 6 LoRA |
| 適応度 | 厳密 Shapley 値 × fitness sharing（`--fitness-mode shapley`） |
| 適応度セット | MMLU-Pro 100 問、シード 777（最終評価とシード分離） |
| sharing $\sigma_{\mathrm{share}}$ | 0.3 |
| 交叉 | ΔW 空間ブレンド ＋ ランダム化 SVD 再分解（`delta`） |
| 混合比 $\alpha$ | $\mathcal{U}(0.3, 0.7)$ から世代・役割ごとに抽出 |
| SVD | oversampling $+8$、power iteration 4 回、rank 32 に切詰め |
| 突然変異率 $\rho_{\mathrm{mut}}$ / 幅 $\sigma_{\mathrm{mut}}$ | 0.3 / 0.02（テンソル標準偏差に対する相対値） |
| 世代 0 変異体 | $\rho_{\mathrm{mut}} = 1.0$、$\sigma_{\mathrm{mut}} = 0.02$ |
| 進化シード | 1234 |

**表 4.4: 最終評価**

| 項目 | 値 |
|---|---|
| ベンチマーク | MMLU-Pro 500 / MATH-500 (Level 4–5) 200 / SuperGPQA 300 |
| シード | $K = 3$（1, 2, 3。問題抽出・生成・同数決着に共通使用） |
| Self-Consistency | $k = 9$ |
| bootstrap 反復 | $B = 10{,}000$ |
| 多重比較補正 | Holm–Bonferroni（2 主要比較 × 3 ベンチ） |

**表 4.5: サービング・インフラ**

| 項目 | 値 |
|---|---|
| vLLM | `--enable-lora --max_loras 8 --max_lora_rank 64 --max-model-len 32768`、TP=1 |
| 動的ロード | `/v1/load_lora_adapter`（`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`） |
| 学習ジョブ | Vertex AI Spot、n1-standard-8 + T4 ×1 |
| 進化・評価ジョブ | Vertex AI Spot、a2-highgpu-1g（A100 40GB ×1）、us-central1 |
| 成果物保存 | GCS（Cloud Storage FUSE マウント）、問題単位 JSONL 逐次追記による再開設計 |

## 参考文献（第4章）

- Cobbe, K. et al. (2021). *Training Verifiers to Solve Math Word Problems (GSM8K).* arXiv:2110.14168.
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314 (NeurIPS 2023).
- Dror, R., Baumer, G., Shlomov, S., & Reichart, R. (2018). The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing. *ACL 2018*.
- Du, X. et al. (2025). *SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines.* arXiv:2502.14739.
- Hendrycks, D. et al. (2021). *Measuring Mathematical Problem Solving with the MATH Dataset.* arXiv:2103.03874 (NeurIPS 2021).
- Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*, 6(2).
- Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM).* arXiv:2309.06180 (SOSP 2023).
- Lightman, H. et al. (2023). *Let's Verify Step by Step.* arXiv:2305.20050（MATH-500 はこの評価分割に由来）.
- Miller, E. (2024). *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations.* arXiv:2411.00640.
- Polo, F. M. et al. (2024). *tinyBenchmarks: Evaluating LLMs with Fewer Examples.* arXiv:2402.14992 (ICML 2024).
- Rein, D. et al. (2023). *GPQA: A Graduate-Level Google-Proof Q&A Benchmark.* arXiv:2311.12022.
- Smit, A. P. et al. (2024). *Should We Be Going MAD? A Look at Multi-Agent Debate Strategies for LLMs.* arXiv:2311.17371 (ICML 2024).
- Wang, X. et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* arXiv:2203.11171 (ICLR 2023).
- Wang, Y. et al. (2024). *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark.* arXiv:2406.01574 (NeurIPS 2024).
- Zhang, Y. et al. (2025). *If Multi-Agent Debate Is the Answer, What Is the Question?* arXiv:2502.08788.
- Qwen Team (2025). *Qwen3-4B-Instruct-2507* モデルカード（Hugging Face）および Qwen3 Technical Report. arXiv:2505.09388.
- （7–8B 同質 MAD の非効率性の報告）arXiv:2605.00914.

> **注**: 提出前に全文献の原典・書誌情報を確認すること（docs/literature_notes.md 冒頭の注意書き参照）。
