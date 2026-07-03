# 文献ノート（修論・関連研究章の素材）

2026-07-03 の 4 本の文献調査の統合。各文献の位置づけと本研究との関係を記す。
執筆時は必ず原典を確認すること。

## A. 進化的モデルマージ / LoRA 進化（直接の競合系統）

| 文献 | 出典 | 要点 | 本研究との差分 |
|---|---|---|---|
| Evolutionary Model Merge (Akiba+, Sakana AI) | arXiv:2403.13187, Nat. Mach. Intell. 2025 | CMA-ESでマージレシピ（層別係数・推論経路）を探索。EvoLLM-JP 7Bが70B級を超過 | 既存フルモデルのレシピ探索。集団の世代交代なし・協調評価なし |
| LoraHub (Huang+) | arXiv:2307.13269, COLM 2024 | LoRA群の合成係数を勾配フリー最適化(CMA-ES系)。few-shotで未知タスク適応 | 単一係数ベクトルの1点解。A/B別補間の交差項問題を内包 |
| Model Swarms (Feng+) | arXiv:2410.11163, ICML 2025 | LoRAエキスパート群をPSOで重み空間協調探索。200例で+21% | 粒子群であり交叉・突然変異・世代交代なし。効用は個体単位 |
| GENOME/GENOME+ (Zhang+) | arXiv:2503.01155 | LoRA重み=遺伝子とし crossover/mutation/selection/succession。最大40個体 | **最近接**。ただし適応度は個体タスク性能のみ。本研究のアブレーションA1に相当 |
| PopuLoRA (Creus Castanyer+) | arXiv:2605.16727 (2026-05) | 共有ベース上のLoRA集団を敵対的self-playで共進化。same-rank交叉演算子 | 相互作用が敵対的（出題者vs解答者）。協調議論でない |
| EvoPref | arXiv:2605.09777 | LoRA 32個体をNSGA-IIで200世代、多様性アーカイブ | アライメント目的・個体評価 |
| Mergenetic (Minut+) | arXiv:2505.11427, ACL 2025 Demo | 進化的マージの汎用ライブラリ(pymoo+MergeKit+lm-eval) | 実装基盤の参考 |

## B. マージ手法の理論（交叉演算子の設計根拠）

- **Model Soups** (Wortsman+ 2022, arXiv:2203.05482): 重み平均の原点。linear mode connectivity 前提
- **Task Arithmetic** (Ilharco+ 2022, arXiv:2212.04089): ΔW のタスクベクトル演算
- **TIES-Merging** (Yadav+, NeurIPS 2023, arXiv:2306.01708): 符号衝突の解決。干渉が線形和劣化の主因
- **DARE** (Yu+, ICML 2024, arXiv:2311.03099): Δのランダムdrop+rescale
- **KnOTS** (Stoica+, ICLR 2025, arXiv:2410.19735): **LoRA由来ΔWは整合が低く素朴平均が劣化**。SVDで共通基底に整列してからマージ（本研究のΔW空間交叉+SVD再分解の直接の根拠）
- **LoRA Soups/CAT** (Prabhakar+, COLING 2025, arXiv:2410.13025): 連結(rank合計)が線形マージに勝る（math +43%）
- **交差項問題**: A/B別補間は ((1-α)B1+αB2)((1-α)A1+αA2) = (1-α)²B1A1+α²B2A2+α(1-α)(B1A2+B2A1)。BA=(BR)(R⁻¹A) の再パラメータ化不変性も破る

## C. Multi-Agent Debate（協調実行の系統）

- **Du+ 2023** (arXiv:2305.14325, ICML 2024): MAD原典。3体×2ラウンドで GSM8K 77→85, MMLU 63.9→71.1。要約提示が有効、反省単体は無効
- **Liang+ DoT** (arXiv:2305.19118, EMNLP 2024): Degeneration-of-Thought。多様な視点の対立がDoT回避に必要（本研究のペルソナ多様性の動機）
- **Multiagent Finetuning** (Subramaniam+, arXiv:2501.05707, ICLR 2025): debate由来データで生成/批評を分化SFT。Phi-3 MATH 58.8→66.0 (5反復)。多様性維持が改善持続と相関。**「勾配模倣 vs 選択圧」の対比相手**
- **MoA** (Wang+, arXiv:2406.04692, ICLR 2025) / **Self-MoA** (Li+, arXiv:2502.00674): 品質拮抗時のみ混合が有効 → 協調寄与適応度で品質と多様性を両立する動機
- **More Agents Is All You Need** (Li+, arXiv:2402.05120): サンプリング+多数決のスケール則
- **MAPoRL** (Park+, arXiv:2502.18439, ACL 2025): 議論の協調品質を報酬にマルチエージェントRL共訓練。**進化ではなくRL**
- **MACA** (arXiv:2509.15172): debateトレースの選好学習で小型モデル GSM8K +27.6%

### MAD への批判（必ず引用し設計で対処）
- **Should we be going MAD?** (Smit+, arXiv:2311.17371, ICML 2024): MADはSC等に安定して勝てない。**計算量マッチSCベースライン必須**
- **If MAD is the Answer...** (Zhang+, arXiv:2502.08788): 評価不備の系統的指摘。異質性の活用を提案（本研究の設定を支持）
- **Talk Isn't Always Cheap** (arXiv:2509.05396): sycophancyによる正→誤の伝染
- **The Cost of Consensus** (arXiv:2605.00914): 7-8B同質MADは非効率・不安定
- **Debate or Vote** (Choi+, arXiv:2508.17536): MAD利得の大半は多数決で説明。信念更新はマルチンゲール
- **Voting or Consensus?** (Kaesberg+, arXiv:2502.19130, ACL 2025 Findings): 推論タスクは投票+13.2%、議論ラウンド追加は逆効果 → ラウンド1-2・多数決集約の根拠
- **Demystifying MAD** (arXiv:2601.19921): 初期回答多様性とconfidence条件付き更新が成功条件

## D. 適応度設計の理論（提案手法章の根拠）

- **Data Shapley** (Ghorbani & Zou, ICML 2019, arXiv:1904.02868): 公理的貢献配分（効率性・対称性・null player・線形性の一意解）
- **Shapley in ML survey** (Rozemberczki+, IJCAI 2022): 2^n コストと近似。**3体なら厳密計算可 = 本研究の強み**
- **COMA** (Foerster+, AAAI 2018, arXiv:1705.08926): counterfactual baseline による credit assignment
- **Difference Reward/WLU** (Wolpert & Tumer 1999, cs/9908014): 反事実貢献の源流
- **Cooperative Coevolution** (Potter & De Jong, PPSN 1994): 役割別サブ集団+協力者評価。**本研究の骨格**
- **PBT** (Jaderberg+ 2017, arXiv:1711.09846): exploit/explore の集団学習。進化ループ手続きの根拠
- **Novelty Search** (Lehman & Stanley 2011): 目的の欺瞞性。多様性圧の動機
- **MAP-Elites** (Mouret & Clune 2015, arXiv:1504.04909): **多様性は適応度に加算せずアーカイブ分離** — 加重和批判の根拠
- **Fitness Sharing** (Goldberg & Richardson 1987): 乗法ペナルティによるニッチ化（本研究が採用）
- **Ambiguity Decomposition** (Krogh & Vedelsby, NeurIPS 1994): E = Ē − Ā。多様性→チーム性能の数学的根拠（RQ3の分析枠組み）
- **Unified Theory of Diversity** (Wood+, JMLR 2023, arXiv:2301.03962): 多様性はbias/varianceと並ぶ第3次元でトレードオフ管理対象。無条件加算への最強の反証
- **NSGA-II** (Deb+ 2002): 加重和の限界（非凸Pareto前線不可・重み恣意性）。アブレーションの参考

## E. 評価方法論（実験章の根拠）

- **Adding Error Bars to Evals** (Miller, arXiv:2411.00640, Anthropic): SEM併記・クラスタSE・K回リサンプル・paired分析・検出力分析の5原則
- **Hitchhiker's Guide** (Dror+, ACL 2018): McNemar/bootstrap/t検定の使い分け
- **tinyBenchmarks** (arXiv:2402.14992, ICML 2024): 100問でフルスコア誤差~2% → 進化ループ内適応度セットの根拠
- 検出力の目安 (paired, α=0.05, power 80%): +10pt→N≈180, +7pt→N≈360, +5pt→N≈700
- **Qwen3-4B-Instruct-2507 公表値**: MMLU-Pro 69.6 / MMLU-Redux 84.2 / GPQA 62.0 / SuperGPQA 42.8 / AIME25 47.4 / IFEval 83.4。GSM8K系は飽和（旧4Bで80-92%）
- **LLM-as-judge**: pairwise + swap + tie許容。Geminiは高得点飽和・自系列選好の報告 → 副次評価に限定
- **vLLM T4制約**: CC7.5はbf16不可・FlashAttention不可・Multi-LoRAカーネル非対応(Issues #1157/#3197/#4246/#20259)

## F. 自己進化エージェントのサーベイ（序論の背景）

- A Survey of Self-Evolving Agents (arXiv:2507.21046)
- A Comprehensive Survey of Self-Evolving AI Agents (arXiv:2508.07407) — MASE パラダイム
- Model Merging survey (Yang+, arXiv:2408.07666, ACM Comput. Surv. 2026)
