# 再設計（v3）向け文献調査レポート（2025H2〜2026H1）

**調査日**: 2026-07-05 / **対象**: Qwen3-4B-Instruct-2507 × LoRAペルソナ3体 × 進化的最適化研究の再設計
**注**: 2026年のarXiv ID（2601〜2606）はプレプリントであり、修論引用前に原典の最終版・採録状況を要確認。数値はabstract/本文から原文ママで転記。

## エグゼクティブサマリ（設計判断に直結する結論）

1. **「素の議論はゲインのほぼ全てが多数決由来」が理論・実証両面で確定**（arXiv:2508.17536, NeurIPS 2025 Spotlight: 信念軌跡がマルチンゲール＝議論単体は期待正解率を改善しない）。議論を活かすには「訂正方向へ信念更新をバイアスする介入」が必須で、我々のP1（条件付き更新）はこの介入の一形態として文献的に正当。
2. **集約損失13ptへの最有力打ち手は「生成的選択（GenSelect型）」と「confidence重み付き投票」**。RSA（arXiv:2509.26626）は**我々と同一モデルQwen3-4B-Instruct-2507**で集約ループにより DeepSeek-R1/o3-mini(high) 級に到達したと報告。4B単体のLLM-as-a-judge採点は非推奨（信頼できるのは概ね14B〜32B以上）だが、「N候補比較→選択」形式なら小型でも機能し、1.7BでもRL訓練で多数決超え（arXiv:2602.02143）。
3. **verbalized confidenceは4B級では信用不可**（80-100%に飽和; arXiv:2502.11028, 2604.01457）。**logprob/エントロピー由来のtoken-level confidence**（DeepConf arXiv:2508.15260, 逆エントロピー投票 arXiv:2511.02309）は訓練不要で即適用可能。
4. **ペルソナSFTの能力毀損対策は「ベースモデル自己生成データによるリプレイ」が2025-26年の主流**で、我々のmake_replay_data.py（P2）は文献の最善筋と同型（arXiv:2506.09428, 2505.13811, 2402.13669）。比率の普遍的相場はなく、「リプレイがベース自身の分布か」が比率より支配的。rank低減・epoch削減も方向として正しい（arXiv:2405.09673）。より根治的には**重みを触らないactivation steering**でペルソナ注入する選択肢（arXiv:2512.07092）。
5. **選抜ノイズ・勝者の呪いは既知の構造問題**（oracle選抜は正に偏る; arXiv:2606.26836）。対策は (i) IRTベースの縮約評価セット（tinyBenchmarks arXiv:2402.14992: 厳選100問でMMLU誤差2%以内、MERGE³ arXiv:2502.10436: 適応度計算50×削減）、(ii) successive halving/racing、(iii) 世代選択をval分離（実装済み）。ランダム100問（SE±5pt）からIRT厳選への置換が費用対効果最大。
6. **多様性は「提案の質を落とさない範囲でのみ」有効**（Self-MoA arXiv:2502.00674: 混合が質を下げると単一最良モデルのアンサンブルに負ける）。我々の「検証可能領域＝個体能力律速/検証困難領域＝多様性律速」の発見は、この文献群（2502.00674 vs 2601.19921/2511.07784）の緊張関係を領域軸で整理するものとして位置づけられる。

---

## テーマ1: 判定者（judge）型の解集約

### 主要論文

| 論文 | arXiv ID | 要点・定量結果 |
|---|---|---|
| GenSelect: A Generative Approach to Best-of-N（NVIDIA） | 2507.17797（原型はAIMO-2優勝解 2504.16891） | N候補を並べてlong reasoningで比較選択。pointwiseスコアリング・pairwise比較を上回り、「majority votingベースラインを大幅に改善」。特に**候補数が少ない時に多数決より有利** |
| Learning Generative Selection for Best-of-N | 2602.02143 | **1.7Bモデル**をDAPO（RL）で選択タスク訓練。AIME24/25・HMMT25・LiveCodeBenchで「prompting・majority-votingベースラインを一貫して上回る」。より強いモデルの出力選択にも汎化 |
| Auditing Multi-Agent LLM Reasoning Trees Outperforms Majority Vote and LLM-as-Judge | 2602.09341 | 推論木のパス探索＋Anti-Consensus Preference Optimization (ACPO)。5設定で**多数決に対し最大+5pt絶対、LLM-as-judgeに対し最大+3pt絶対** |
| JUDGEBOARD / MAJ | 2511.15958 | 単独judgeではSLMとLLMに大きな性能差。ただし**複数SLMのMulti-Agent Judging (MAJ)** はMATHで大型モデルjudgeと同等以上 |
| Judging the Judges（position bias体系研究） | 2406.07791 | judge15種×約150,000評価。position biasは**候補間の品質差が小さいほど悪化**。指標: repetition stability / position consistency / preference fairness |
| Self-Preference Bias in LLM-as-a-Judge | 2410.21819 | judgeは自己出力（＝自分のperplexityが低い出力）を過大評価 |
| Reliability without Validity（大規模judge信頼性評価） | 2606.19544 | 人間判断との整合が強いのは**概ね≥32B**（例外的にQwen2.5-14B）。それ未満の小型モデルはjudgeプロンプトの追加ガイダンスを活用できない |

### 4B級への適用可能性
- **スコアリング型judge（rubric採点・pairwise裁定）としての4Bは信頼性不足**（2606.19544, 2511.15958）。
- 一方**「全候補を1コンテキストに並べて比較選択する」GenSelect形式は比較優位を活かせるため小型でも機能**し、RL訓練すれば1.7Bでも多数決超え（2602.02143）。訓練なしプロンプトのみの場合は効果が落ちる点に注意。
- 複数の4Bをjudge側でも協調させるMAJ（2511.15958）は我々の3体構成と親和的。

### 設計への示唆
- **採用**: 議論の最終集約に「3体の解（＋タイブレーク用にベース解）を匿名化・順序ランダム化して並べ、1回のGenSelect型比較選択で裁定」を追加。多数決とハイブリッド（多数決が2-1で割れた時のみ選択を起動）にすればコスト増最小で、oracle 51.7%と38.7%の間の「1体だけ正解」ケースを直接狙える。
- **必須の対策**: (1) position bias→候補順序ランダム化 or 2順序評価の一致採用（品質差が小さい候補ペアほどバイアス大: 2406.07791）、(2) self-selection bias→どの解が誰の(自分の)ものか隠す匿名化（2410.21819, 2510.07517）。
- **回避**: 4B単体にスコア（0-10点等）を出させる採点型集約。verbalized score は較正されない（テーマ2参照）。

---

## テーマ2: 小型LLM（3-8B）マルチエージェント議論の成功条件

### 主要論文

| 論文 | arXiv ID | 要点・定量結果 |
|---|---|---|
| Debate or Vote | 2508.17536 | MADを多数決と議論に分解。**性能ゲインの大半は多数決由来**。議論は信念のマルチンゲールを誘導＝議論単体は期待正解率を改善しない（定理）。ただし「訂正方向への信念更新バイアス」介入は有効。NeurIPS 2025 Spotlight |
| Demystifying Multi-Agent Debate: The Role of Confidence and Diversity | 2601.19921 | vanilla MADは多数決に劣ることを確認。処方は2つ: **(1) diversity-aware初期化**（Ncand=10から異なる答えの数を最大化するよう貪欲にN個選抜、訓練不要）、**(2) confidence-modulated更新**（0-10の確信度をRLで較正+他者確信度で条件付け）。Qwen2.5-7B/Llama-3.1-8Bで多数決比 **GSM8K +2.4pt / CommonsenseQA +2.0pt / HellaSwag +1.9pt** |
| Can LLM Agents Really Debate?（Knight-Knave-Spy統制実験） | 2511.07784 | 成功の支配要因は**個体の推論力とグループ多様性**。失敗機構は「**多数派圧力による独立訂正の抑制**」＝誤った合意を覆せない。妥当性志向の推論が改善を最も強く予測 |
| Peacemaker or Troublemaker: How Sycophancy Shapes MAD | 2509.23055 | MAD特化のsycophancy定義と測定指標を初提案。過剰な同調は議論を早期崩壊させ**単体エージェントより低精度**に。sycophancyはゼロでも過剰でもなく**中間の最適域**が存在 |
| Measuring and Mitigating Identity Bias in MAD via Anonymization | 2510.07517 | 議論をidentity重み付きベイズ更新として定式化。Identity Bias Coefficient (IBC)で測定した結果**sycophancyがself-biasより遥かに優勢**。**発言の匿名化**（誰の発言か消す）だけで自己/他者の重みが均等化。訓練不要・モデル非依存 |
| DebUnc | 2407.06426 | 不確実性メトリクスで発言力を制御。**テキストで確信度を伝えるよりattentionスケーリングで直接重み付けする方が有効** |
| SID: Multi-LLM Debate Driven by Self Signals | 2510.06843 | モデルレベル確信度で高確信エージェントは議論を早期離脱、attentionで議論内容を圧縮。精度を上げつつトークン大幅削減 |
| 較正の実態: Mind the Confidence Gap / Wired for Overconfidence | 2502.11028 / 2604.01457 | 小型LLMのverbalized confidenceは**80-100%に集中する過信**。過信は内部回路に由来し、モデルサイズ縮小で悪化 |
| MarsRL（solver/verifier/corrector分離） | 2511.11373 | 役割分離（解く/検査する/直す）＋RLのパイプライン。役割非対称化の代表例（RL前提） |

### 4B級への適用可能性
- 上記の主要結果は7-8Bで確認されており、4Bでは「素の議論の悪化」がさらに強く出ると予想される（我々のMMLU-Pro c3 −4.1ptは文献と整合）。
- diversity-aware初期化・匿名化・早期離脱は**訓練不要で4Bに即適用可**。confidence-modulated更新のRL較正はコスト高（採用するならlogprobベース代替）。

### 設計への示唆
- **採用（優先度順）**: (1) **発言の匿名化**（役割名・人格ラベルを議論文面から外す）。(2) **P1条件付き更新の継続**——2508.17536の「訂正方向バイアス介入」・2511.07784の「多数派圧力の抑制」の実装形。(3) **round0全員一致時の議論スキップ＋高確信エージェントの早期離脱**（SID）。(4) 初期解多様性の最大化＝ペルソナLoRAの存在意義を「diversity-aware初期化のweight空間版」として理論武装（2601.19921が直接の裏付け）。
- **回避**: verbalized confidence（0-10自己申告）をそのまま重みに使うこと（4Bでは飽和）。多ラウンド化（round=1固定は引き続き正しい）。solver/verifier/judgeの完全分離をRLなしで導入すること（MarsRL型はRL前提であり、SFTのみでverifierペルソナを作ると能力毀損コストの方が大きい）。

---

## テーマ3: LoRA SFTの能力毀損（破滅的忘却）対策

### 主要論文

| 論文 | arXiv ID | 要点・定量結果 |
|---|---|---|
| LoRA Learns Less and Forgets Less | 2405.09673 (TMLR) | 忘却量は**rankとともに増加**、LoRAはfull FTより忘れないが学習も少ない（同一トレードオフ曲線上）。LoRAは**学習率に極めて敏感**。推奨: 全モジュール適用・α=2r・full FTの約10倍のLR |
| Improved SFT to Mitigate Catastrophic Forgetting | 2506.09428 | **ベースモデルの指示分布を再構成した自己生成データ**を混合。公開SFTデータ（第三者データ）混合ベースラインを上回りつつタスク性能も改善 |
| Context-Free Synthetic Data Mitigates Forgetting | 2505.13811 | **モデル自身の無条件（文脈なし）生成**をリプレイに使うとKLダイバージェンスの不偏推定に相当。OLMo-1Bのzero-shot性能とR1-Distill-Llama-8Bの**推論性能の保持**を確認 |
| SDFT: Self-Distillation Bridges Distribution Gap | 2402.13669 (ACL 2024) | タスクデータの応答を**モデル自身の分布で書き直して**から学習。忘却緩和＋タスク性能同等以上。忘却の主因は「タスクデータとモデル分布のギャップ」 |
| OPLoRA | 2510.13003 | 凍結重みをSVDし、LoRA更新を**top-k特異部分空間の直交補空間に両側射影で制限**。LLaMA-2-7B/Qwen2.5-7Bで忘却を大幅低減しタスク性能は競争的。干渉量メトリクスρ_kを提案 |
| GeRe: General Samples Replay | 2508.04676 | 固定の少量一般テキストリプレイ＋閾値マージン損失で十分な忘却抑制（比率チューニング不要を主張） |
| The Geometry of Persona | 2512.07092 | 人格特性は**直交線形部分空間**にある→重み更新なしのベクトル操作でZero-Shot Personality Injection（凍結Qwen-2.5、心理指標MSE 0.011）。**SFTのalignment tax自体を回避** |
| （周辺）O-LoRA / LoRA-Loop / on-policy replay | 2310.14152 / 2507.13568 / 2605.29495 | タスク間直交化・リプレイ設計の2025-26年の派生群 |

### リプレイ比率の相場
普遍的な最適比率は**未確立**。継続事前学習系では1-5%で足りるという報告がある一方、SFT系の実務では10-50%混合が広く使われ、GeRe（2508.04676）は「固定少量で十分」、2506.09428/2505.13811/2402.13669は**比率より「リプレイがベースモデル自身の生成分布であること」が支配的**と示す。我々の36例/96例（約38%）はレンジ内で妥当。

### 設計への示唆
- **採用**: (1) **P2リプレイ再学習はそのまま進める**——「ベース自己生成の正解長CoTを混ぜてrank16/2ep」は2506.09428＋2405.09673の推奨と完全整合。数学L4-5の長い導出を厚めに入れる設計は「短CoTスタイル刷り込み＝分布ギャップ」というSDFTの因果仮説とも合致。(2) rank32→16・3ep→2epの低減（忘却はrank・学習量とともに増える）。(3) 学習率をrank変更時に再チューニング（LoRAはLR感度が支配的）。(4) 余力があれば**OPLoRA式のtop-k特異方向保護**をtrain_lora_persona.pyに追加しアブレーション（「進化がSFT損傷方向を打ち消した」というΔW回転仮説の対照実験としても価値がある）。
- **検討（研究の差別化として面白い）**: ペルソナを**SFTでなくsteering vectorで注入**した条件c3''を追加すれば、「多様性を重みに焼くこと自体の必要性」を切り分けられる（2512.07092）。ただしvLLM Multi-LoRA運用と両立しない（推論時フック要）ため、工数次第。
- **回避**: 第三者一般データ（例: 公開Alpaca系）だけのリプレイ（自己生成に劣る: 2506.09428）。rank・epoch据え置きでのリプレイのみ対応。

---

## テーマ4: 重み空間モデルマージ/進化の最新（2026前半まで）

### 主要論文

| 論文 | arXiv ID | 要点・定量結果 |
|---|---|---|
| DARE | 2311.03099 | delta重みをdrop率pで落とし1/(1-p)でrescale。SFT deltaなら90-99% dropでも性能維持だが、**極端なdrop率では顕著に劣化**し、deltaが大きい（＝よく学習した）モデルには不向き。実務はp∈{0.1..0.9}のグリッド探索が標準 |
| DELLA-Merging | 2406.11617 | 大きさに基づく確率的drop（MAGPRUNE）。彼らの探索での**最適trim/drop率は0.4** |
| Why Do More Experts Fail? | 2505.21226 | マージ利得は**厳密に凹**＝モデル数に対し収穫逓減、有効パラメータ空間の飽和が律速（Gaussian Width解析）。最適なマージ数閾値の存在を証明。対策としてRHT（再パラメータ化重裾）提案 |
| GENOME/GENOME+ | 2503.01155 | 集団ベース進化（交叉・突然変異・選択＋継承・アンサンブル）。12データセットで既存マージ/適応法を一貫して上回り、**初期集団最良LLM比 最大+54.8%**（gemma-2-2b級・少数ショット適応度設定） |
| Darwin Family | 2605.14386 | 14次元マージゲノム＋層重要度診断と進化探索を混合するMRI-Trust Fusion。**4B〜35B**で親モデル超え、Darwin-27B-OpusはGPQA Diamond 86.9%（1,252モデル中#6）。勾配学習なし・多世代再帰進化 |
| Evo-L2S（多目的進化マージ） | 2604.06465 | 精度×出力長の**パレート最適化**として進化マージを定式化。1.5B-14Bで推論トレース長50%超削減・精度維持。**エントロピーベースのサブセットサンプリングで適応度推定を激安化** |
| MERGE³ | 2502.10436 (ICML 2025) | 縮約データセット＋**IRT能力推定**で適応度計算を**50×削減**、単一GPUで進化マージ |
| tinyBenchmarks | 2402.14992 (ICML 2024) | **IRTで厳選した100問**でMMLU（14K問）の精度を**誤差2%以内**で推定（140×削減）。※ランダム100問ではなく厳選が肝 |
| Rethinking LLM Evaluation (EssenceBench) | 2510.10457 | GAで評価サブセット圧縮。HellaSwag 10K→**25問**でモデル順位を5%シフト以内に保存 |
| Don't Always Pick the Highest-Performing Model | 2602.08003 | アンサンブル選択を**相互情報量最大化**として定式化。LLM誤りは強相関（Gaussian copulaで誤り床を導出）→高性能モデルだけ足しても飽和。貪欲MI選択がMEDMCQA/MMLU/IMDBで同予算ベースライン超え |
| Self-MoA | 2502.00674 | **単一最良モデルの自己アンサンブルが異種混合MoAに勝つ**（AlpacaEval 2.0 +6.6%、MMLU/CRUX/MATH平均+3.8%）。混合の利益は提案品質に極めて敏感 |
| （選抜ノイズ）Capability Frontier / 二段選抜 | 2606.26836 / 2404.00069 | oracle選抜はノイズ最大値を取るため**正に偏る**（勝者の呪いの一般形）。successive halvingはadapter選抜で評価数を2×以上削減しつつ同等性能 |

### 設計への示唆
- **採用**: (1) **適応度セットをランダム100問→IRT/情報量ベース厳選に置換**（tinyBenchmarks方式）。同じ100問でもSEを大幅に下げられ、勝者の呪い（実測±4-7pt揺れ）の直接対策になる。世代ごと層化再サンプルと併用可。(2) **successive halving（60問足切り→300問確認）は文献裏付けあり**のまま進める。最終世代選択のval分離も2606.26836と整合。(3) **DARE drop率は0.4以下の穏当な値から**（DELLA最適0.4。0.9級の大dropはSFT deltaが小さい場合の性質であり、LoRA ΔWに直接高dropを適用する根拠は弱い。KnOTSの指摘どおりSVD整列後に適用が安全）。(4) **適応度への多様性項は「誤り相関（MI）ベース」への理論強化が可能**（2602.08003が「チーム適応度が個体性能の和でない」ことの情報理論的裏付け）。(5) マージは**2-3個の少数交叉を維持**（2505.21226: 多数マージは凹に飽和）。
- **回避**: 世代内1-2pt差での選抜（SH+IRTでも解消しきれない差は「同点」と扱い複数candidateを次世代へ持ち越す）。適応度評価の問題数を増やすだけの対策（コストが立たない。厳選が先）。

---

## テーマ5: Self-Consistencyを超えるtest-time compute集約

### 主要論文

| 論文 | arXiv ID | 要点・定量結果 |
|---|---|---|
| DeepConf: Deep Think with Confidence（Meta） | 2508.15260 | **token-level confidence（logprob由来）で低品質トレースを除外**した上でconfidence重み付き多数決。AIME 2025でDeepConf@512が**最大99.9%**（GPT-OSS-120B）、生成トークン**最大84.7%削減**。**訓練・チューニング不要**、vLLM統合済み、Qwen3系でも検証 |
| CISC: Confidence Improves Self-Consistency | 2502.06233 (ACL 2025 Findings) | 自己評価confidenceで重み付き多数決。9モデル×4データセットでほぼ全構成でSC超え、**必要サンプル数を平均40%以上削減**。VecCISC（2605.08070）はさらにトークン47%削減 |
| RSA: Recursive Self-Aggregation | 2509.26626 | 候補推論チェーン集団を「部分集合ごとに集約→改良集団を生成」で反復進化。**Qwen3-4B-Instruct-2507がDeepSeek-R1やo3-mini(high)に肉薄**（AIME-25, HMMT-25, Reasoning Gym, LiveCodeBench-v6, SuperGPQA）。並列・逐次スケーリング両方を上回る |
| AggLM: The Majority is not always right | 2509.06870 | 集約を明示的推論スキルとして**RLVRで訓練**（1.7B級アグリゲータ）。ルールベース（多数決）・報酬モデルベースラインを超え、**少数派正解の回収**が可能。異なる・より強いモデルの解にも汎化。多数決より少トークン |
| The Sequential Edge: Inverse-Entropy Voting | 2511.02309 | 同一計算量で**逐次リファインが並列SCを95.6%の構成で上回る**（最大+46.7%）。訓練不要の**逆エントロピー重み付き投票**が97%のケースで最適 |
| 補助: Optimal LLM+PRM aggregation / CGES / Best-of-Majority | 2510.13918 / 2511.02603 / 2510.03199 | PRMスコアとLLM合意信号の理論的最適結合 / confidence誘導の早期打ち切り / pass@kに対するminimax最適戦略（多数決→上位選抜のハイブリッド） |

### 設計への示唆
- **採用**: (1) **チーム集約とSCベースラインの両方に「logprobベースconfidence重み付き投票」を導入**。round0の3体解＋議論後解の集約を単純多数決からDeepConf式（低confidenceトレース除外＋重み付き投票）へ。実装はvLLMのlogprobs取得のみで訓練不要。(2) **RSA型ハイブリッド**: 「議論」を「他者解を材料に各自が集約・改良解を出す」操作として再解釈すると、我々のdebate round1はRSAの1ステップと同型。集約プロンプトを「批判・投票」から「複数候補の統合・再導出」へ寄せるだけで集約損失の回収余地がある。(3) **「同一計算量で両者に最良集約を許した比較」**（SC@9+CISC vs チーム+重み付き投票+GenSelect裁定）を最終評価に含めると査読耐性が上がる。
- **回避**: 逐次リファイン一辺倒への転換（2511.02309は魅力的だが、我々の貢献はチーム多様性×進化にある。逐次はSC強化ベースラインとして言及に留める）。報酬モデル/PRM導入（4B規模の予算でPRM運用はコスト過大、2510.13918は理論参照のみ）。

---

## 我々の実測課題(a)〜(f)への処方の対応表

| 課題 | 処方（文献裏付け） | 優先度 |
|---|---|---|
| (a) ペルソナSFTの−7〜13pt毀損 | ベース自己生成リプレイ約38%＋rank16/2ep（2506.09428, 2505.13811, 2405.09673）。オプション: OPLoRA射影（2510.13003）、steering vector注入への転換（2512.07092） | **P2実装済み・最優先で完走** |
| (b) チームゲイン+5〜12ptが毀損を埋めない | 多様性の源泉を保ちつつ(a)で毀損を消す。「多様性はSFT、能力はリプレイで保持」＝パレート改善の続行（Self-MoA 2502.00674の品質感度が理論的warning） | 高 |
| (c) SC@9とタイ | チーム側集約の強化: logprob重み付き投票（2508.15260, 2502.06233）＋GenSelect型裁定（2507.17797）＋議論スキップで浮いた計算を候補数増へ（2510.06843）。公平比較の枠組み再設計 | **高（+1.5-2ptの主要供給源）** |
| (d) 集約損失13pt（oracle 51.7%→38.7%） | 多数決が2-1/1-1-1に割れた問題のみGenSelect裁定を起動（2602.09341: MV比+5pt絶対の報告）。匿名化・順序ランダム化を併用 | **高** |
| (e) 選抜ノイズSE±5pt・勝者の呪い | IRT厳選適応度セット（2402.14992, 2502.10436）＋successive halving（2404.00069）＋同点candidateの持ち越し。oracle選抜の正バイアスは構造的（2606.26836）なのでval分離は継続 | 高 |
| (f) 追従（sycophancy） | 発言匿名化（2510.07517）＋条件付き更新P1（2508.17536の「訂正バイアス介入」）＋中庸の非同調が最適という設計原理（2509.23055） | 中（P1のA/B結果待ち） |

## 修論での位置づけに使える論述素材

- 「素の議論が効かない」ことは2508.17536（マルチンゲール定理）・2601.19921で理論・実証とも確立済み。**我々のMATH-500で素の議論が+4.0pt有意に効いた発見は、この一般論への領域依存の反例**として提示できる（検証可能領域では自己訂正が働き、マルチンゲール仮定＝更新の無情報性が破れる、という解釈）。
- 「重み焼き込みペルソナの価値」は2601.19921のdiversity-aware初期化（プロンプト操作）と対比し、「初期多様性をweight空間で恒久化し、進化で能力毀損だけを除去する」枠組みとして差別化できる。GENOME（2503.01155）・Darwin（2605.14386）はいずれもタスク精度適応度であり、**チームレベルShapley適応度の進化は依然として未報告**（2026-07-05時点の本調査でも交点の直接競合は見つからず。ただし2602.09341のACPO=反合意選好最適化が最近接なので要引用・対比）。

## 参照した主要arXiv ID一覧（テーマ横断・番号順）

2311.03099 (DARE), 2402.13669 (SDFT), 2402.14992 (tinyBenchmarks), 2404.00069 (二段選抜SH), 2405.09673 (LoRA Learns Less), 2406.07791 (Judging the Judges), 2406.11617 (DELLA), 2407.06426 (DebUnc), 2410.21819 (Self-Preference Bias), 2502.00674 (Self-MoA), 2502.06233 (CISC), 2502.10436 (MERGE³), 2502.11028 (Confidence Gap), 2503.01155 (GENOME), 2504.16891 (AIMO-2/OpenMathReasoning), 2505.13811 (Context-Free Replay), 2505.21226 (Why More Experts Fail), 2506.09428 (Improved SFT), 2507.17797 (GenSelect), 2508.04676 (GeRe), 2508.15260 (DeepConf), 2508.17536 (Debate or Vote), 2509.06870 (AggLM), 2509.14034 (Confidence Expression MAD), 2509.16839 (Roundtable Policy), 2509.23055 (Peacemaker or Troublemaker), 2509.26626 (RSA), 2510.06843 (SID), 2510.07517 (匿名化MAD), 2510.10457 (EssenceBench), 2510.13003 (OPLoRA), 2510.13918 (LLM+PRM最適集約), 2511.02309 (Inverse-Entropy Voting), 2511.02603 (CGES), 2511.07784 (Can LLM Agents Really Debate), 2511.11373 (MarsRL), 2511.15958 (JUDGEBOARD), 2512.07092 (Geometry of Persona), 2601.19921 (Demystifying MAD), 2602.02143 (Learning GenSelect), 2602.08003 (MIアンサンブル選択), 2602.09341 (Auditing Reasoning Trees), 2603.09892 (MSSR), 2604.01457 (Wired for Overconfidence), 2604.06465 (Evo-L2S), 2604.23178 (judgeバイアス緩和評価), 2605.08070 (VecCISC), 2605.14386 (Darwin Family), 2605.29495 (On-Policy Replay), 2606.19544 (Reliability without Validity), 2606.26836 (Capability Frontier)
