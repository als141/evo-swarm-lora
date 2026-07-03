# 第1章 序論

## 1.1 背景

### 1.1.1 単一モデルの拡大から小型モデル集団の協調へ

大規模言語モデル（LLM）の性能は、パラメータ数・データ量・計算量の拡大によって向上してきた。しかし、モデル規模の拡大は学習・推論コストの急激な増大を伴い、単一モデルの能力を際限なく引き上げるアプローチには経済的・工学的な制約が存在する。この制約を背景として、近年、「個の拡大」に代わる方向性として「集団の協調」への関心が高まっている。すなわち、単一の巨大モデルにすべての能力を担わせるのではなく、複数のモデル（あるいは複数のインスタンス）を相互作用させ、その集合としての振る舞いから単体を超える性能を引き出すという考え方である。実際、複数のサンプル生成と多数決という最も素朴な協調形態であっても、エージェント数の増加に伴い性能が単調に向上するスケール特性が報告されている（Li ら (2024)、arXiv:2402.05120）。また、配備済みモデルの重みやプロンプト、エージェント構成そのものを実行後に継続的に改善していく「自己進化エージェント」という枠組みが体系化されつつあり、モデルを静的な成果物ではなく進化し続ける集団として扱う視点が整理されている（arXiv:2507.21046; arXiv:2508.07407）。

この潮流は、小型モデルにとって特に重要な意味を持つ。数十億パラメータ級（以下、~4B 級と表記する）の小型モデルは、単体では最先端の巨大モデルに及ばないものの、ローカル環境や安価な計算資源上で複数個を同時に運用できるため、「複数の特化した小型モデルの協調」によって単体性能の限界を補うという設計が現実的な選択肢となる。さらに、パラメータ効率的微調整手法である LoRA（Low-Rank Adaptation）を用いれば、単一のベースモデルの上に複数の軽量アダプタを載せ替えるだけで多様な「人格（ペルソナ）」や役割を持つエージェント群を構成でき、メモリコストを抑えたまま集団を形成できる。本研究はこの構図、すなわち「共有ベースモデル + 役割別 LoRA アダプタ群」による小型モデル協調を対象とする。

### 1.1.2 マルチエージェント議論の登場とその限界

複数の LLM エージェントを協調させる代表的なプロトコルが、マルチエージェント議論（Multi-Agent Debate; MAD）である。Du ら (2023) は、複数の LLM インスタンスが互いの回答を参照しながら複数ラウンドにわたり回答を更新し、最終的に合意された回答を採用する手続きにより、3 エージェント × 2 ラウンドの構成で GSM8K の精度を 77 から 85 へ、MMLU を 63.9 から 71.1 へ改善できることを示した（arXiv:2305.14325）。また Liang ら (2023) は、単一モデルの自己反省が同一の思考に固執する「思考の退化（Degeneration-of-Thought）」に陥ることを指摘し、異なる視点を持つエージェント間の対立的議論がその回避に有効であることを示した（arXiv:2305.19118）。

しかし、その後の検証研究は MAD の効果に重大な留保を付けている。Smit ら (2023) は、MAD が Self-Consistency などの単純なアンサンブル手法に対して計算量を揃えた比較では安定して勝てないことを示した（arXiv:2311.17371）。さらに、議論の中で他者の回答に同調して正答を誤答へ書き換えてしまう追従性（sycophancy）による誤りの伝染（arXiv:2509.05396）、同一モデルの複製からなる同質なエージェント集団では議論が非効率かつ不安定になるという報告（arXiv:2605.00914）が続いている。Zhang ら (2025) は既存 MAD 研究の評価上の不備を系統的に指摘したうえで、MAD を機能させる鍵はエージェント間の異質性の活用にあると論じた（arXiv:2502.08788）。すなわち、MAD の成否は「議論に参加するエージェント集団の質と多様性」に強く依存するにもかかわらず、既存研究の多くはエージェント側を固定し、プロトコル（ラウンド数・集約方式・プロンプト）の工夫のみで改善を図ってきたのである。

## 1.2 研究課題

以上の背景から、本研究は次の三つの課題を出発点とする。

**(a) 既存の MAD 研究はエージェントを固定し、協調の質そのものを最適化していない。** MAD の性能はエージェント集団の構成に依存するにもかかわらず、既存研究の主流は与えられたモデルを所与として議論プロトコルを調整するものであった。議論への貢献度を報酬としてモデル自体を訓練する試みは MAPoRL（Park ら (2025)、arXiv:2502.18439）などの強化学習ベースの研究に限られ、「議論チームへの貢献」を選択圧としてエージェント集団（の重み）を世代交代的に進化させる研究は、著者の知る限り報告されていない。

**(b) 既存の LoRA 集団進化は個体性能のみを適応度とする。** LoRA アダプタの重みを遺伝子とみなし、交叉・突然変異・選択によって集団を進化させる研究系統が近年急速に発展している（GENOME：Zhang ら (2025)、arXiv:2503.01155；PopuLoRA：Creus Castanyer ら (2026)、arXiv:2605.16727 など）。しかしこれらの適応度は個体単独のタスク性能（あるいは敵対的 self-play の勝敗）であり、「チームの一員としてどれだけ協調に寄与するか」を測る適応度は用いられていない。個体として優秀なエージェントの集合が優れたチームになるとは限らない、というアンサンブル学習の古典的知見（Krogh & Vedelsby (1995)）に照らせば、これは本質的な欠落である。

**(c) 多様性を適応度に加算する素朴な設計には理論的問題がある。** チームの多様性が重要であるならば、「性能 + λ×多様性」の形で適応度に多様性項を加算すればよいと考えたくなる。しかし、品質多様性（Quality-Diversity）研究は多様性を目的関数に混ぜるのではなくアーカイブ構造で分離して保持すべきことを示しており（MAP-Elites：Mouret & Clune (2015)、arXiv:1504.04909）、アンサンブル多様性の統一理論もまた、多様性は bias・variance と並ぶトレードオフ管理の対象であって無条件に加算してよい量ではないことを示している（Wood ら (2023)、arXiv:2301.03962）。したがって、協調と多様性を扱う適応度設計には、加重和に代わる理論的に正当化された構成が必要である。

## 1.3 提案手法の要点と本研究の貢献

本研究は、上記の課題に対し、**マルチエージェント議論のチームレベル適応度——厳密 Shapley 値によって測られる協調寄与——を選択圧として、ペルソナ LoRA 集団を進化的に最適化する枠組み**を提案する。対象は Qwen3-4B-Instruct を共有ベースモデルとする 3 役割（批判的検証者・実務的意思決定者・発散的探索者）の LoRA エージェント集団であり、協調的共進化（Potter & De Jong (1994)）の枠組みに従って役割別サブ集団を維持しながら、世代ごとに (i) 各候補 LoRA を代表チームに組み込んだ議論評価、(ii) Shapley 値に基づく貢献度の算出、(iii) ΔW 空間での交叉と突然変異による次世代生成、を繰り返す。

適応度の中核には協力ゲーム理論の Shapley 値を採用する。Shapley 値は効率性・対称性・null player・線形性の公理を一意に満たす貢献配分であり（Ghorbani & Zou (2019)、arXiv:1904.02868）、一般には連合数が指数的に増えるため近似を要するが（Rozemberczki ら (2022)）、本研究の 3 エージェント構成では全 7 連合の議論評価によって**近似なしの厳密計算**が可能である。また、課題 (c) を踏まえ、多様性は適応度への加算項とせず、fitness sharing（Goldberg & Richardson (1987)）による乗法ペナルティ——回答不一致率で測った行動距離に基づき、同一ニッチに密集した個体の適応度を割り引く——として組み込む。

本研究の貢献は以下の 3 点である。

1. **厳密 Shapley 値による協調寄与を LoRA 集団進化の適応度に組み込む初の定式化。** 議論チームへの限界貢献を、3 エージェント構成において全連合の実測から近似なしに計算可能な Shapley 値として定式化し、これを選択圧とする進化的最適化の枠組みを提示する。Shapley 値は個体単独の性能（単独連合の価値）を内包するため、性能項と協調項の恣意的な加重和を必要としない点で、既存の適応度設計と理論的に区別される。

2. **協調的共進化の枠組みによる MAD チーム最適化手法の提案。** Potter & De Jong (1994) の役割別サブ集団と協力者評価の考え方を LLM エージェント集団に移植し、役割内・役割間の ΔW 空間交叉（LoRA 因子の素朴な補間が持つ交差項問題を回避するため、ΔW = BA を明示的に構成して補間し SVD で低ランクに再分解する方式；Stoica ら (2024)、arXiv:2410.19735 の知見に基づく）と fitness sharing を組み合わせた世代交代アルゴリズムを設計する。

3. **4B 級小型モデルにおける「議論が機能する条件」の実証分析。** 7〜8B 級の同質エージェントによる素の議論は非効率・不安定であり（arXiv:2605.00914）、評価設計の不備を正すと MAD の優位が消えるという批判（arXiv:2502.08788）も踏まえると、4B 級では素の議論は機能しない可能性が高い。本研究は、この不利な条件下でチームレベル選択圧が議論を機能させうるかを、計算量を揃えた Self-Consistency ベースライン、個体性能のみを適応度とするアブレーション（GENOME 型）、およびプロンプトのみのペルソナ付与との統制比較によって検証し、性能と行動多様性の世代推移から議論が機能する／しない条件を分析する。

なお、本研究の実験は現在進行中であり、本章では実験結果には言及しない。

## 1.4 本論文の構成

本論文の構成は以下の通りである。第 2 章では、マルチエージェント議論、モデルマージと LoRA 合成の理論、進化的モデル最適化、協調の定量化と適応度設計の理論という 4 つの研究系統を整理し、本研究の位置づけを明確化する。第 3 章では、提案手法——Shapley 値に基づくチームレベル適応度、fitness sharing、ΔW 空間交叉を含む協調的共進化アルゴリズム、および議論プロトコル——を定式化する。第 4 章では、ベンチマーク・比較条件・統計的検定を含む実験設計を述べる。第 5 章では実験結果とその分析を報告し、第 6 章で考察と限界を論じ、第 7 章で結論と今後の展望を述べる。

---

## 参考文献（第1章）

- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., Mordatch, I. Improving Factuality and Reasoning in Language Models through Multiagent Debate. ICML 2024 (arXiv:2305.14325), 2023.
- Liang, T. et al. Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate. EMNLP 2024 (arXiv:2305.19118), 2023.
- Li, J. et al. More Agents Is All You Need. arXiv:2402.05120, 2024.
- Smit, A. P. et al. Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs. ICML 2024 (arXiv:2311.17371), 2023.
- Zhang, H. et al. If Multi-Agent Debate is the Answer, What is the Question? arXiv:2502.08788, 2025.
- Talk Isn't Always Cheap: Understanding Failure Modes in Multi-Agent Debate. arXiv:2509.05396, 2025.
- The Cost of Consensus: Inefficiency and Instability of Homogeneous Multi-Agent Debate. arXiv:2605.00914, 2026.
- Park, C. et al. MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning. ACL 2025 (arXiv:2502.18439), 2025.
- Zhang, W. et al. GENOME: Nature-Inspired Population-Based Evolution of Large Language Models. arXiv:2503.01155, 2025.
- Creus Castanyer, R. et al. PopuLoRA: Adversarial Co-Evolution of LoRA Populations. arXiv:2605.16727, 2026.
- Krogh, A., Vedelsby, J. Neural Network Ensembles, Cross Validation, and Active Learning. NeurIPS 1994, 1995.
- Mouret, J.-B., Clune, J. Illuminating Search Spaces by Mapping Elites. arXiv:1504.04909, 2015.
- Wood, D. et al. A Unified Theory of Diversity in Ensemble Learning. JMLR 2023 (arXiv:2301.03962), 2023.
- Potter, M. A., De Jong, K. A. A Cooperative Coevolutionary Approach to Function Optimization. PPSN III, 1994.
- Ghorbani, A., Zou, J. Data Shapley: Equitable Valuation of Data for Machine Learning. ICML 2019 (arXiv:1904.02868), 2019.
- Rozemberczki, B. et al. The Shapley Value in Machine Learning. IJCAI 2022, 2022.
- Goldberg, D. E., Richardson, J. Genetic Algorithms with Sharing for Multimodal Function Optimization. ICGA 1987, 1987.
- Stoica, G. et al. Model Merging with SVD to Tie the Knots (KnOTS). ICLR 2025 (arXiv:2410.19735), 2024.
- A Survey of Self-Evolving Agents. arXiv:2507.21046, 2025.
- A Comprehensive Survey of Self-Evolving AI Agents. arXiv:2508.07407, 2025.
