# 第2章 関連研究

本章では、本研究に関連する研究を 4 つの系統——(1) マルチエージェント議論、(2) モデルマージと LoRA 合成の理論、(3) 進化的アプローチによるモデル最適化、(4) 協調の定量化と適応度設計の理論——に整理し、最後に本研究の位置づけを述べる。

## 2.1 マルチエージェント議論

### 2.1.1 原典と初期の成果

マルチエージェント議論（Multi-Agent Debate; MAD）の原典とされるのが Du ら (2023) の研究である。複数の LLM インスタンスがまず独立に回答を生成し、続くラウンドで他エージェントの回答を参照しながら自身の回答を批判的に更新する、という単純な手続きにより、3 エージェント × 2 ラウンドの構成で GSM8K を 77 から 85 へ、MMLU を 63.9 から 71.1 へ改善できることが示された（arXiv:2305.14325）。同研究では、他者回答を要約して提示することが有効である一方、単一エージェントの自己反省のみでは改善が得られないことも報告されており、「他者からの異なる情報」が改善の源泉であることが示唆されている。並行して Liang ら (2023) は、単一モデルの自己反省が自身の初期解に固執する「思考の退化（Degeneration-of-Thought; DoT）」を指摘し、異なる立場を割り当てられたエージェント間の対立的議論が DoT の回避に必要であると論じた（arXiv:2305.19118）。この知見は、議論に参加するエージェントに明示的な視点の多様性——本研究の文脈ではペルソナの多様性——を与えるべきであるという本研究の設計動機の直接の源流である。

議論を実行時のプロトコルとしてだけでなく訓練信号の源として使う方向も発展した。Subramaniam ら (2025) は、議論の軌跡から生成役と批評役を分化させて自己教師あり微調整（SFT）を行う Multiagent Finetuning を提案し、Phi-3 の MATH 精度を 5 反復で 58.8 から 66.0 へ改善するとともに、エージェント間の多様性の維持が反復改善の持続と相関することを示した（arXiv:2501.05707）。また MACA は議論トレースからの選好学習により小型モデルの GSM8K を +27.6% 改善した（arXiv:2509.15172）。さらに Park ら (2025) の MAPoRL は、議論における協調品質を報酬としてマルチエージェント強化学習で複数モデルを共訓練する枠組みを提案しており（arXiv:2502.18439）、「協調の質を最適化信号にする」という点で本研究に最も思想が近い。ただし MAPoRL は勾配ベースの RL であり、重み空間の集団に対する世代交代型の進化ではない（この対比は 2.5 節で詳述する）。

このほか、複数モデルの出力を階層的に統合する Mixture-of-Agents（Wang ら (2024)、arXiv:2406.04692）に対して、Li ら (2025) は単一の強いモデルの複数サンプルを統合する Self-MoA が異種混合を上回る場合が多いことを示し、混合が有効なのは構成モデルの品質が拮抗している場合に限られると分析した（arXiv:2502.00674）。品質と多様性はトレードオフの関係にあり、多様性は品質を犠牲にしない範囲でのみ価値を持つ——この知見は、品質（貢献度）と多様性を単一の適応度に統合する本研究の設計課題を先取りしている。

### 2.1.2 批判研究：MAD は本当に有効か

MAD の初期の成功報告に対しては、その後、系統的な批判が積み重ねられている。第一に、Smit ら (2023) は複数の MAD 戦略を計算量を統制して比較し、MAD が Self-Consistency（複数サンプルの多数決）等の単純な手法に安定して勝てないことを示した（arXiv:2311.17371）。これは MAD の効果検証において計算量を揃えたベースラインが不可欠であることを意味する。第二に、Zhang ら (2025) は既存 MAD 研究の評価設計（ベースラインの弱さ、ハイパーパラメータの非統制等）を系統的に指摘し、公平な条件では MAD の優位が大きく縮小すること、そして今後の方向としてエージェント間の異質性の活用を提案した（arXiv:2502.08788）。第三に、議論という相互作用そのものの失敗様式も特定されている。他者への追従（sycophancy）により、当初正答を持っていたエージェントが誤答へ引きずられる「正→誤の伝染」が観察されており（arXiv:2509.05396）、7〜8B 級の同一モデルの複製からなる同質な MAD は非効率かつ不安定であることが報告されている（arXiv:2605.00914）。

### 2.1.3 成功条件の知見と本研究への含意

批判研究と並行して、「どのような条件なら議論が機能するのか」を切り分ける研究も現れている。Choi ら (2025) は MAD の利得の大半が最終的な多数決（投票）で説明でき、議論中の信念更新はマルチンゲール的（期待値的に情報を増やさない）であることを示した（arXiv:2508.17536）。Kaesberg ら (2025) は集約方式を比較し、推論タスクでは合意形成よりも投票が優れ（+13.2%）、議論ラウンドの追加はむしろ逆効果であることを示した（arXiv:2502.19130）。さらに近年の分析は、初期回答の多様性と確信度に条件付けられた回答更新が MAD 成功の鍵であることを示している（arXiv:2601.19921）。

これらを総合すると、次の緊張関係が浮かび上がる。**議論の利得は「多様な初期回答の生成」と「正しい方向への選択的更新」から生じるが、同質で追従的な小型モデル集団はそのどちらも満たしにくい。** 7〜8B 級ですら同質 MAD の不安定性が報告されている以上、本研究が対象とする 4B 級では、素の（無調整の）debate は機能しない可能性が高い。本研究はこの緊張関係を回避すべき前提ではなく検証対象として引き受ける。すなわち、(i) プロトコル面では文献の成功条件に忠実に従い（独立初期回答、最大 2 ラウンド、多数決集約）、(ii) エージェント面では重み空間に焼き込まれたペルソナ多様性とチームレベル選択圧によって「議論が機能するエージェント集団」を明示的に育成できるかを問う。

## 2.2 モデルマージと LoRA 合成の理論

本研究の進化アルゴリズムは LoRA 重みの交叉（ブレンド）を核とするため、重み合成がいつ・なぜ妥当なのかという理論的根拠を必要とする。

重み平均によるモデル合成の原点は Model Soups（Wortsman ら (2022)、arXiv:2203.05482）である。同一の事前学習モデルから微調整された複数モデルの重みを単純平均するだけで精度と頑健性が向上することが示されたが、これは微調整解が損失地形上で線形に接続されている（linear mode connectivity）ことを前提とする。Ilharco ら (2022) の Task Arithmetic は、微調整前後の重み差分 ΔW を「タスクベクトル」とみなし、その加減算によって能力を合成・除去できることを示し、ΔW 空間での算術という枠組みを確立した（arXiv:2212.04089）。

一方で、素朴な線形和には劣化要因があることも明らかになっている。Yadav ら (2023) の TIES-Merging は、タスクベクトル間のパラメータ符号の衝突（干渉）が線形和の性能劣化の主因であることを特定し、符号衝突を解決してからマージする手法を提案した（arXiv:2306.01708）。Yu ら (2023) の DARE は、ΔW の大部分をランダムに drop して残りを rescale してもタスク性能が保たれる冗長性を利用し、干渉を減らしてマージする（arXiv:2311.03099）。

LoRA に固有の問題を扱ったのが Stoica ら (2024) の KnOTS である。同研究は、LoRA によって得られた ΔW はフルファインチューニングの場合よりもモデル間の整合（alignment）が低く、素朴な重み平均が顕著に劣化することを示し、複数の ΔW を SVD によって共通基底に整列させてからマージする手法を提案した（arXiv:2410.19735）。さらに、LoRA の低ランク因子 A・B を別々に補間する方式には**交差項問題**がある。すなわち、2 つのアダプタ (B₁A₁), (B₂A₂) の因子を係数 α で別々に補間すると、

((1−α)B₁ + αB₂)((1−α)A₁ + αA₂) = (1−α)²B₁A₁ + α²B₂A₂ + α(1−α)(B₁A₂ + B₂A₁)

となり、意図した凸結合 (1−α)B₁A₁ + αB₂A₂ ではなく、意味を持たない交差項 B₁A₂, B₂A₁ が混入する。加えて BA = (BR)(R⁻¹A) という再パラメータ化不変性のため、同一の ΔW を表す因子分解は無数に存在し、因子空間での補間は分解の取り方に依存してしまう。したがって、LoRA 集団の交叉演算子は因子空間ではなく ΔW 空間で定義し、必要に応じて SVD で低ランクに再分解するのが理論的に正当である。本研究の交叉演算子（ΔW 空間ブレンド + SVD 再分解）はこの知見に直接依拠しており、因子別補間（naive 方式）はアブレーションとして比較する。

なお、線形マージ以外の合成として、Prabhakar ら (2024) は複数 LoRA をランク方向に連結する CAT（ランクの合計をとる連結）が線形マージを上回り、数学タスクで +43% の改善を得ることを報告している（arXiv:2410.13025）。モデルマージ全般の体系的整理は Yang ら (2024) のサーベイに詳しい（arXiv:2408.07666）。

## 2.3 進化的アプローチによるモデル最適化

進化計算を LLM の重み・合成レシピの探索に適用する研究は、近年ひとつの系統を成しつつある。

先駆けとなったのは Akiba ら（Sakana AI）の進化的モデルマージである（arXiv:2403.13187）。CMA-ES を用いて既存フルモデル群のマージレシピ（層別のマージ係数や推論経路）を探索し、日本語 LLM である EvoLLM-JP 7B が一部ベンチマークで 70B 級モデルを超える性能を達成した。ただしこれは既存モデルを素材とする「レシピの 1 点探索」であり、モデル集団自体の世代交代は行われない。LoraHub（Huang ら (2023)、arXiv:2307.13269）も同様に、既存 LoRA 群の合成係数ベクトルを勾配フリー最適化で求める 1 点解の探索であり、かつ A・B 因子別の合成であるため前節の交差項問題を内包する。

集団ベースの探索へ進んだのが Model Swarms（Feng ら (2024)、arXiv:2410.11163）である。複数の LoRA エキスパートを粒子とみなし、粒子群最適化（PSO）によって重み空間を協調探索することで、200 例程度の少数データから最大 +21% の適応改善を得た。ただし PSO は速度ベクトルによる連続的な移動であり、交叉・突然変異・世代交代という遺伝的操作を持たず、効用関数も個体単位である。

遺伝的アルゴリズムを LoRA 重みに全面的に適用したのが GENOME/GENOME+（Zhang ら (2025)、arXiv:2503.01155）である。LoRA 重みを遺伝子とみなし、交叉・突然変異・選択・継承（succession）からなる世代交代を最大 40 個体の集団で実行する。本研究にとって最近接の先行研究であるが、その適応度は**個体単独のタスク性能のみ**である。同系統の EvoPref は 32 個体の LoRA を NSGA-II で 200 世代進化させ多様性アーカイブを併用するが、目的はアライメントであり評価は個体単位である（arXiv:2605.09777）。また PopuLoRA（Creus Castanyer ら (2026)、arXiv:2605.16727）は共有ベースモデル上の LoRA 集団を出題者対解答者の敵対的 self-play で共進化させ、同ランク制約下の交叉演算子を提案した。個体間の相互作用が適応度を決めるという点で本研究に近づいているが、その相互作用は敵対的であり、協調的な議論への寄与を測るものではない。実装基盤としては、進化的マージの汎用ライブラリ Mergenetic（Minut ら (2025)、arXiv:2505.11427）が公開されている。

主要研究の対比を表 2.1 に示す。

**表 2.1: 進化的モデル最適化の主要研究の対比**

| 研究 | 進化の対象 | 探索方式 | 集団と世代交代 | 適応度の評価単位 | 個体間相互作用 |
|---|---|---|---|---|---|
| 進化的モデルマージ (Akiba ら, arXiv:2403.13187) | マージレシピ（層別係数・推論経路） | CMA-ES | なし（レシピ 1 点解） | マージ後モデル単体の性能 | なし |
| LoraHub (Huang ら, arXiv:2307.13269) | LoRA 合成係数ベクトル | 勾配フリー最適化 | なし（1 点解） | 合成モデル単体の性能 | なし |
| Model Swarms (Feng ら, arXiv:2410.11163) | LoRA 重み（粒子） | PSO | 集団あり・世代交代なし | 個体単位の効用 | 探索情報の共有のみ |
| GENOME/GENOME+ (Zhang ら, arXiv:2503.01155) | LoRA 重み（遺伝子） | GA（交叉・突然変異・選択） | 集団あり・世代交代あり（最大 40 個体） | 個体単独のタスク性能 | なし |
| EvoPref (arXiv:2605.09777) | LoRA 重み | NSGA-II（200 世代） | 集団あり・世代交代あり（32 個体） | 個体単位（アライメント目的） | なし |
| PopuLoRA (Creus Castanyer ら, arXiv:2605.16727) | LoRA 重み | 共進化 GA | 集団あり・世代交代あり | 対戦の勝敗 | **敵対的** self-play |
| **本研究** | LoRA 重み（役割別サブ集団） | 協調的共進化 GA | 集団あり・世代交代あり | **チーム議論への厳密 Shapley 寄与** | **協調的** 議論 |

表 2.1 が示すとおり、系統の発展は「レシピの 1 点探索 → 集団による重み空間探索 → 世代交代型の遺伝的進化 → 相互作用を伴う共進化」と進んできたが、**協調的な相互作用（議論）のチームレベル評価を適応度とする世代交代型 LoRA 進化**は空白のまま残されている。

## 2.4 協調の定量化と適応度設計の理論

前節の空白を埋めるには、「チームへの貢献」を原理的に定義し、それを進化の選択圧として健全に機能させる理論が必要である。本節では本研究の適応度設計が依拠する 4 つの理論的支柱を整理する。

### 2.4.1 Shapley 値による貢献配分

協力ゲーム理論の Shapley 値は、連合（coalition）の価値関数 v が与えられたとき、各プレイヤーの貢献を全部分連合への限界貢献の加重平均として配分する。機械学習分野では Ghorbani & Zou (2019) が訓練データの価値評価（Data Shapley）に導入し、Shapley 値が効率性・対称性・null player・線形性の 4 公理を満たす一意の配分であることを配分原理としての根拠とした（arXiv:1904.02868）。Rozemberczki ら (2022) のサーベイが整理するように、Shapley 値の計算は一般に 2ⁿ 個の連合評価を要するため近似手法が発達してきたが、逆に言えば n が小さければ厳密計算が可能である。本研究は議論チームを 3 エージェントに設計することで、全 7 連合（空集合を除く）の議論を実測し**近似誤差のない厳密 Shapley 値**を得る。これは単純な leave-one-out（LOO）近似が持つ欠陥——たとえば同一能力の複製が 2 個体いる場合、どちらを除いても価値が変わらないため両者に貢献 0 を割り当ててしまう——を原理的に回避する。

マルチエージェント強化学習における credit assignment の系譜も同じ問題意識を共有する。Wolpert & Tumer (1999) の difference reward（Wonderful Life Utility）は「自分がいなかった場合」との反事実差分で個体の貢献を測る枠組みの源流であり（arXiv:cs/9908014）、Foerster ら (2018) の COMA は counterfactual baseline を用いた方策勾配としてこれを深層 MARL に実装した（arXiv:1705.08926）。Shapley 値はこれら反事実的貢献を全連合にわたって公理的に平均化したものと位置づけられる。

### 2.4.2 協調的共進化と集団ベース訓練

複数の部分要素が組み合わさって初めて全体の性能が決まる問題に対する進化計算の古典的解が、Potter & De Jong (1994) の協調的共進化（Cooperative Coevolution）である。問題を部分要素（本研究では議論の役割）ごとのサブ集団に分割し、各個体を他サブ集団の代表（協力者）と組み合わせて評価する。この「代表チーム文脈での個体評価」は、本研究において各候補 LoRA の Shapley 値を計算する際のチーム構成法（候補 1 体 + 他役割の代表 2 体）の骨格そのものである。また、Jaderberg ら (2017) の Population Based Training（PBT）は、訓練途中の集団に対する exploit（劣位個体の優位個体による置換）と explore（摂動）の交互適用が有効であることを示しており（arXiv:1711.09846）、本研究の世代ループ（役割内選抜と突然変異）の手続き的な根拠を与える。

### 2.4.3 多様性の扱い：加算ではなく分離・ペナルティ

進化計算において目的関数への単一目的化は欺瞞的（deceptive）な探索を招きうる。Lehman & Stanley (2011) の Novelty Search は、目的への直接最適化がかえって目的達成を妨げる場合があることを示し、行動の新規性そのものを探索圧とする発想を導入した。しかし「では多様性を適応度に加算すればよい」とは言えない。Mouret & Clune (2015) の MAP-Elites は、多様性を適応度に混ぜるのではなく行動記述子で張られたアーカイブとして分離保持する品質多様性（QD）の枠組みを確立し（arXiv:1504.04909）、多目的最適化の古典 NSGA-II（Deb ら (2002)）も、加重和方式が非凸 Pareto 前線を捉えられず重み係数の選択が恣意的になるという限界を明確にしている。アンサンブル学習の側からは、Krogh & Vedelsby (1995) の ambiguity 分解——アンサンブル誤差 E は構成員の平均誤差 Ē から構成員間の不一致 Ā を引いた E = Ē − Ā に分解される——が多様性の価値の数学的根拠を与える一方、Wood ら (2023) の統一理論は、多様性が bias・variance と並ぶ誤差の第 3 次元であり、損失関数に依存してその効き方が変わるトレードオフ管理の対象であることを示し、多様性の無条件な加算に対する最も強い反証となっている（arXiv:2301.03962）。

これらを踏まえ、本研究は多様性を適応度への加算項とせず、Goldberg & Richardson (1987) の fitness sharing——行動的に近接した個体同士で適応度を割り引く乗法ペナルティ——として組み込む。fitness sharing はニッチ形成による早期収束の防止という明確な役割を持ち、貢献度（Shapley 値）の順序構造を加算項のように恣意的な係数で歪めない。また Krogh & Vedelsby の分解は、進化の過程でチームの行動多様性（回答不一致プロファイル）がどう変化しチーム性能とどう関係するかを分析する枠組み（本研究の RQ3）として用いる。

### 2.4.4 評価方法論

適応度と最終評価の設計には評価方法論の知見を用いる。tinyBenchmarks（arXiv:2402.14992）は約 100 問のサブセットでフルベンチマークとの誤差が約 2% に収まることを示しており、進化ループ内の適応度セット（固定 100 問）の規模設定の根拠となる。最終評価の統計的検定は、SEM の併記・クラスタ標準誤差・複数回リサンプル・paired 分析・検出力分析を勧告する Miller (2024) の指針（arXiv:2411.00640）と、NLP における有意性検定の使い分けを整理した Dror ら (2018) に従う。

## 2.5 本研究の位置づけ

以上の 4 系統を俯瞰すると、既存研究は「協調プロトコルの改善（2.1 節）」「重み合成の演算子（2.2 節）」「重み集団の進化（2.3 節）」「貢献と多様性の理論（2.4 節）」をそれぞれ発展させてきたが、それらの交点——**議論のチームレベル適応度を選択圧とする LoRA 重み集団の世代交代型進化**——は未報告である。表 2.2 に、隣接する研究系統と本研究に欠けている要素を整理する。

**表 2.2: 隣接研究系統と本研究の位置づけ**

| 系統 | 代表研究 | 最適化の対象 | 最適化の信号 | 本研究との差分（欠けている要素） |
|---|---|---|---|---|
| LoRA 集団 + 進化演算 | GENOME (arXiv:2503.01155), EvoPref (arXiv:2605.09777) | LoRA 重み集団 | 個体タスク性能 | 適応度が個体性能のみで、協調寄与を測らない |
| LoRA 集団 + 相互作用共進化 | PopuLoRA (arXiv:2605.16727) | LoRA 重み集団 | 敵対的 self-play の勝敗 | 相互作用が敵対的であり、協調議論でない |
| 議論品質を最適化信号に | MAPoRL (arXiv:2502.18439) | モデル重み（勾配更新） | 議論の協調品質報酬 | RL であり、集団の世代交代型進化でない |
| チーム適応度で MAS を進化 | EvoMAS (arXiv:2602.06511), Meta-Team (arXiv:2605.29790) | プロンプト・エージェント構成 | チーム性能 | 進化対象が重みでなくプロンプト／構成 |
| LoRA 重み空間の協調探索 | Model Swarms (arXiv:2410.11163) | LoRA 重み集団 | 個体効用 | PSO であり交叉・突然変異・世代交代がない |
| 議論由来データで SFT | Multiagent Finetuning (arXiv:2501.05707) | モデル重み（勾配更新） | 議論トレースの模倣 | 勾配による模倣であり、選択圧でない |
| **本研究** | — | **役割別 LoRA 重み集団** | **議論チームへの厳密 Shapley 寄与 × fitness sharing** | — |

本研究の位置づけは次の 3 点に要約される。

第一に、**最適化信号の新規性**である。表 2.2 の左列 2 系統（GENOME・PopuLoRA）は重み集団の進化という「器」を持つが信号が個体的・敵対的であり、中列 2 系統（MAPoRL・EvoMAS 等）は協調・チームという「信号」を持つが対象が勾配更新やプロンプトである。本研究は、公理的に正当化された貢献配分である厳密 Shapley 値（2.4.1 項）を LoRA 重み集団の世代交代進化の適応度に導入することで、この器と信号を初めて接続する。GENOME 型の個体性能適応度は本研究のアブレーション（A1）として実験内に包含され、新規性の主張は直接比較によって検証可能な形になっている。

第二に、**理論的整合性を持つ演算子の選択**である。交叉は KnOTS の知見（2.2 節）に基づく ΔW 空間ブレンド + SVD 再分解とし、因子別補間の交差項問題を回避する。多様性は QD とアンサンブル統一理論の批判（2.4.3 項）を踏まえ、加算項ではなく fitness sharing の乗法ペナルティとして扱う。議論プロトコルは批判研究が特定した成功条件（2.1.3 項：独立初期回答・少数ラウンド・多数決集約）に準拠する。

第三に、**否定的知見も成果となる問題設定**である。4B 級の同質エージェントによる素の議論は機能しない可能性が高いという緊張関係（2.1.3 項）の下で、チームレベル選択圧が議論を機能させるならばそれは MAD を「エージェント集団の育成問題」として捉え直す証拠となり、機能しないならば適応度別・世代別の性能と行動多様性の軌跡から小型モデル議論の成立条件を統計的に特定できる。いずれの帰結でも、計算量を統制した Self-Consistency ベースライン（Smit ら (2023) の要請）と paired 検定（Miller (2024) の指針）に基づく厳密な比較の上で結論が導かれる点が、評価不備が指摘されてきた MAD 研究（Zhang ら (2025)）に対する方法論的な貢献となる。

なお、本領域は 2025〜2026 年にかけて発展が速く、PopuLoRA のような近接研究が継続的に出現している。執筆時点の位置づけは上記の通りであるが、最終稿では novelty の再検索を行い、本表を更新する。

---

## 参考文献（第2章）

### マルチエージェント議論
- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., Mordatch, I. Improving Factuality and Reasoning in Language Models through Multiagent Debate. ICML 2024 (arXiv:2305.14325), 2023.
- Liang, T. et al. Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate. EMNLP 2024 (arXiv:2305.19118), 2023.
- Subramaniam, V. et al. Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains. ICLR 2025 (arXiv:2501.05707), 2025.
- MACA: Preference Learning from Debate Traces for Small Models. arXiv:2509.15172, 2025.
- Park, C. et al. MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning. ACL 2025 (arXiv:2502.18439), 2025.
- Wang, J. et al. Mixture-of-Agents Enhances Large Language Model Capabilities. ICLR 2025 (arXiv:2406.04692), 2024.
- Li, W. et al. Rethinking Mixture-of-Agents: Is Mixing Different Large Language Models Beneficial? (Self-MoA). arXiv:2502.00674, 2025.
- Li, J. et al. More Agents Is All You Need. arXiv:2402.05120, 2024.
- Smit, A. P. et al. Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs. ICML 2024 (arXiv:2311.17371), 2023.
- Zhang, H. et al. If Multi-Agent Debate is the Answer, What is the Question? arXiv:2502.08788, 2025.
- Talk Isn't Always Cheap: Understanding Failure Modes in Multi-Agent Debate. arXiv:2509.05396, 2025.
- The Cost of Consensus: Inefficiency and Instability of Homogeneous Multi-Agent Debate. arXiv:2605.00914, 2026.
- Choi, S. et al. Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models? arXiv:2508.17536, 2025.
- Kaesberg, L. B. et al. Voting or Consensus? Decision-Making in Multi-Agent Debate. ACL 2025 Findings (arXiv:2502.19130), 2025.
- Demystifying Multi-Agent Debate: Conditions for Success. arXiv:2601.19921, 2026.

### モデルマージと LoRA 合成
- Wortsman, M. et al. Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy without Increasing Inference Time. ICML 2022 (arXiv:2203.05482), 2022.
- Ilharco, G. et al. Editing Models with Task Arithmetic. ICLR 2023 (arXiv:2212.04089), 2022.
- Yadav, P. et al. TIES-Merging: Resolving Interference When Merging Models. NeurIPS 2023 (arXiv:2306.01708), 2023.
- Yu, L. et al. Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE). ICML 2024 (arXiv:2311.03099), 2023.
- Stoica, G. et al. Model Merging with SVD to Tie the Knots (KnOTS). ICLR 2025 (arXiv:2410.19735), 2024.
- Prabhakar, A. et al. LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks. COLING 2025 (arXiv:2410.13025), 2024.
- Yang, E. et al. Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities. ACM Computing Surveys 2026 (arXiv:2408.07666), 2024.

### 進化的モデル最適化
- Akiba, T. et al. Evolutionary Optimization of Model Merging Recipes. Nature Machine Intelligence 2025 (arXiv:2403.13187), 2024.
- Huang, C. et al. LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition. COLM 2024 (arXiv:2307.13269), 2023.
- Feng, S. et al. Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence. ICML 2025 (arXiv:2410.11163), 2024.
- Zhang, W. et al. GENOME: Nature-Inspired Population-Based Evolution of Large Language Models. arXiv:2503.01155, 2025.
- Creus Castanyer, R. et al. PopuLoRA: Adversarial Co-Evolution of LoRA Populations. arXiv:2605.16727, 2026.
- EvoPref: Evolutionary Multi-Objective LoRA Optimization for Preference Alignment. arXiv:2605.09777, 2026.
- Minut, S. et al. Mergenetic: A Simple Evolutionary Model Merging Library. ACL 2025 System Demonstrations (arXiv:2505.11427), 2025.
- EvoMAS: Evolving Multi-Agent Systems with Team-Level Fitness. arXiv:2602.06511, 2026.
- Meta-Team: Meta-Optimization of Agent Team Configurations. arXiv:2605.29790, 2026.

### 協調の定量化・適応度設計・評価方法論
- Ghorbani, A., Zou, J. Data Shapley: Equitable Valuation of Data for Machine Learning. ICML 2019 (arXiv:1904.02868), 2019.
- Rozemberczki, B. et al. The Shapley Value in Machine Learning. IJCAI 2022, 2022.
- Wolpert, D. H., Tumer, K. An Introduction to Collective Intelligence. arXiv:cs/9908014, 1999.
- Foerster, J. et al. Counterfactual Multi-Agent Policy Gradients (COMA). AAAI 2018 (arXiv:1705.08926), 2018.
- Potter, M. A., De Jong, K. A. A Cooperative Coevolutionary Approach to Function Optimization. PPSN III, 1994.
- Jaderberg, M. et al. Population Based Training of Neural Networks. arXiv:1711.09846, 2017.
- Lehman, J., Stanley, K. O. Abandoning Objectives: Evolution through the Search for Novelty Alone. Evolutionary Computation 19(2), 2011.
- Mouret, J.-B., Clune, J. Illuminating Search Spaces by Mapping Elites. arXiv:1504.04909, 2015.
- Goldberg, D. E., Richardson, J. Genetic Algorithms with Sharing for Multimodal Function Optimization. ICGA 1987, 1987.
- Krogh, A., Vedelsby, J. Neural Network Ensembles, Cross Validation, and Active Learning. NeurIPS 1994, 1995.
- Wood, D. et al. A Unified Theory of Diversity in Ensemble Learning. JMLR 2023 (arXiv:2301.03962), 2023.
- Deb, K. et al. A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation 6(2), 2002.
- Polo, F. M. et al. tinyBenchmarks: Evaluating LLMs with Fewer Examples. ICML 2024 (arXiv:2402.14992), 2024.
- Miller, E. Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations. arXiv:2411.00640, 2024.
- Dror, R. et al. The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing. ACL 2018, 2018.
