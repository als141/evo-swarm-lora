# 初期ペルソナ設計の根拠づけ — 文献調査と、候補プール・選抜手続きの提案

- 作成: 2026-09-30（文献調査フォーク）
- 背景: 世代0の3ペルソナ（critic / pragmatist / explorer）は 2026-07 に根拠なく決めたものである。修士審査で「なぜこの3人か」と問われると答えられない。v4（ベースモデル Qwen3-4B-Instruct-2507 上のペルソナ3体の議論社会 → 議論からの社会学習で LoRA 化 → 厳密 Shapley で選抜する世代進化）の出発点を、理論と実証の両面から根拠づける。
- 方針: arXiv abs・出版社ページ・会議資料で実在と書誌を確認した。確認できなかった点は「未確認」と明記した。

---

## 0. 結論（要約）

1. **文献が一貫して示すこと**
   - 性格ラベルだけのペルソナは、客観課題の精度をほとんど変えない。効果の向きは不安定で、ほぼランダムである（Zheng+ EMNLP Findings 2024）。能力を削ることもある（Kim+ 2024、PRISM 2026）。
   - 一方、**議論に効くのは推論手続き（考え方の筋道）の多様性**である（DMAD, ICLR 2025／DiMo 2025／Wu+ 2025）。
   - 7月の再解析でも、日本語1文の性格ペルソナ間の不一致率は、同じエージェントを再サンプリングした場合と ±1pp で区別できなかった（`reanalysis.md` §2.2）。文献と整合する。
   - したがって、ペルソナは「性格」だけでなく、**役割に対応する具体的な推論手続き**として定義すべきである。
2. **候補プールは人間の問題解決チームの役割理論から網羅的に作る**
   - Belbin の9チーム役割（Belbin 1981/1993）をすべて候補にし、1役割も恣意的に落とさない。各役割は、確立したプロンプト手法に対応する推論手続きとして操作化する。例: Step-Back、Plan-and-Solve、Self-Verification、Analogical prompting。
   - 元の3人は Belbin の Monitor Evaluator（critic）、Implementer（pragmatist）、Plant（explorer）に対応し、**プールの部分集合として自然に含まれる**。
   - Belbin の実証的妥当性には留保がある（Furnham+ 1993、Aritzeta+ 2007）。このため理論は「候補を網羅的に作る枠組み」として使い、**3人の選択はデータで行う**。
3. **選抜は本研究のチーム貢献度（Shapley）と同じ考え方で行う**
   - 9役割を、アフィン平面 AG(2,3) の **12個の三つ組**（どの2役割も1回ずつ同じチームになる釣り合い型計画）で評価する。
   - 役割ごとの「チームへの平均寄与（主効果）」を推定し、独立した dev 半分で確認する。
   - 理論上の既定トリオ（元の3人）を上回ると確認できた場合だけ置き換える（事前登録の決定規則）。
   - 費用は A100 約2時間、**約$4〜5** の見込み。
4. **修論での主張の立て方**: 「ペルソナは人間のチーム役割理論から網羅的に候補化し、推論手続きとして操作化し、提案手法と同じチーム貢献度で選抜した」。仮に元の3人が残っても、それはデータで再確認された選択になり、恣意性の指摘に答えられる。

---

## 1. 文献調査

### 1.1 マルチエージェント議論で「どの多様性」が精度に効くか

| 文献 | 何を多様にしたか | 主な結果 | 本研究への含意 |
|---|---|---|---|
| **DMAD**: Liu, Cao, Li, He, Tan. *Breaking Mental Set to Improve Reasoning through Diverse Multi-Agent Debate*. **ICLR 2025**（OpenReview t6QHYUOQL7, [proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3de667dab3b3d812583abc0a786139a0-Abstract-Conference.html), [code](https://github.com/MraDonkey/DMAD)） | **推論手法**。LLM では CoT / Step-Back / Program-of-Thought を各エージェントに割り当てる | 詳細は下の囲みを参照 | 性格ではなく**推論手続きを変える**のが要点。「ペルソナを変えても推論法が同じなら mental set から抜けられない」という問題設定そのものが、本研究の7月の失敗と対応する |
| **DiMo**: He & Feng. *Unleashing Diverse Thinking Modes in LLMs through Multi-Agent Collaboration*. arXiv [2510.16645](https://arxiv.org/abs/2510.16645)（2025-10） | 4体がそれぞれ異なる推論パラダイムを担う | 単一モデルと通常の議論のベースラインを上回り、**数学で最大の改善**（6ベンチ） | 推論様式の多様化が数学で効く（本研究の MATH と対応） |
| Wu, Li, Li. *Can LLM Agents Really Debate? A Controlled Study of MAD in Logical Reasoning*. arXiv [2511.07784](https://arxiv.org/abs/2511.07784)（2025-11） | チーム規模・構成・確信度の開示・順序・深さ・難易度を統制 | "intrinsic reasoning strength and group diversity are the dominant drivers"。順序や確信度の開示の効果は限定的。"majority pressure suppresses independent correction" | 構成（誰を入れるか）が最重要変数。同調が正解の独立な訂正を抑える（7月の再解析の「少数派の95%が同調」と整合） |
| Zhu, Zhang, Chi, Stafford, Collier, Vlachos. *Demystifying Multi-Agent Debate: The Role of Confidence and Diversity*. arXiv [2601.19921](https://arxiv.org/abs/2601.19921)（2026-01, 改訂 06） | 初期回答の多様性（diversity-aware initialisation）と確信度 | 初期の候補集合を多様にすると正解が含まれる確率が上がる。確信度で条件づけた更新で正解側へ流れる | 初期多様性は議論成功の前提条件 |
| **ReConcile**: Chen, Saha, Bansal. **ACL 2024**. arXiv [2309.13007](https://arxiv.org/abs/2309.13007) | **異なるモデル**（API／オープン／特化） | "the diversity originating from different models is critical"。MATH で +8% | 同一ベースでは得にくい多様性。本研究では LoRA の世代進化で後から作る |
| Hegazy. *Diversity of Thought Elicits Stronger Reasoning Capabilities in Multi-Agent Debate Frameworks*. JRAR 5(3), 2024. arXiv [2410.12853](https://arxiv.org/abs/2410.12853) | 異種の中型モデル | GSM-8K で異種3モデルが4ラウンド後に 91%、同種（Gemini-Pro×3）は 82%。ASDiv 94% | 同上 |
| **ChatEval**: Chan+. **ICLR 2024**. arXiv [2308.07201](https://arxiv.org/abs/2308.07201) | 評価者の役割プロンプト | 多様な役割プロンプトが本質的で、同じ役割記述にすると性能が落ちる（本文の記述。検索要約による。本文の該当表は未精査） | 評価課題（生成物の採点）での結果で、正誤のある推論課題ではない点に注意 |
| **MachineSoM**: Zhang, Xu, Zhang, Liu, Hooi, Deng. *Exploring Collaboration Mechanisms for LLM Agents: A Social Psychology View*. **ACL 2024**. arXiv [2310.02124](https://arxiv.org/abs/2310.02124) | 特性（easy-going / overconfident）×思考パターン（debate / reflection）で4つの「社会」を構成 | 協調戦略によっては既存最良を上回り、トークンも少ない。**同調・合意形成など人間の社会心理学的な振る舞いが現れる** | 「人間社会のようなエージェント社会」という本研究の動機の直接の先行例。特性より協調戦略の設計が効くという示唆（相対比較の明示は abstract では未確認） |
| Zhang, Kim, Xiang, Gao, Cao. *Dynamic Role Assignment for Multi-Agent Debate*. arXiv [2601.17152](https://arxiv.org/abs/2601.17152)（2026-01） | 役割への配置（Meta-Debate で候補を採点して配置） | 一様な配置比**最大+74.8%**、ランダム配置比**最大+29.7%** | **役割の割り当て方は結果を大きく変える**。役割は恣意的に決めてよい変数ではない |
| Pappu+. *Multi-Agent Teams Hold Experts Back*. **ICML 2026**. arXiv [2602.01011](https://arxiv.org/abs/2602.01011) | 自己組織化チーム | 最強の個人に対し、最大 41.1% 劣る。"integrative compromise"（専門家と非専門家の平均化）が原因 | 同調型の役割は危険。反論・検証を担う役割を候補に含めるべき根拠 |

**DMAD の主な結果**（ICLR 2025 のスライドで確認。[slides](https://iclr.cc/media/iclr-2025/Slides/28079.pdf)）
- 評価: LLM は MATH と GPQA、GPT-4o-mini と LLaMA-3-70B-Instruct。
- 集計図（GPT-4o-mini）: CoT 52.50 / Self-Refine 52.61 / MAD(All CoT) 55.47 / MAD(All SBP) 53.10 / **DMAD 56.54**。
- 集計図（LLaMA-3-70B）: CoT 40.45 / MAD(All CoT) 42.09 / **DMAD 43.41**。集計の定義（どのベンチの平均か）は本文未確認。
- 「全手法が同じ MAD」が全ラウンド誤答し続けた問題（mental set 問題）のうち、DMAD が解けた割合: All-CoT の70問中48問（68.6%）、All-SBP の87問中60問（69.0%）、All-PoT の67問中49問（73.1%）。
- 異なる推論手法の数を 1→2→3 に増やすと精度が上がる（GPT-4o-mini の MATH、図5）。

**まとめ**
- 精度に効くのは、①推論手続きの多様性（DMAD, DiMo）、②モデルそのものの多様性（ReConcile, Hegazy）、③誰を入れるか・どこに配置するか（Wu+、Dynamic Role Assignment）である。
- 性格だけの多様性は、主に「同調のしやすさ」を変える（次節）。
- 同一ベースモデルを使う本研究の世代0では②が使えない。このため**①を役割ごとに与え、③をデータで決める**のが最も筋が良い。

### 1.2 システムプロンプトのペルソナの効果と限界

| 文献 | 主な結果 | 本研究への含意 |
|---|---|---|
| Zheng, Pei, Logeswaran, Lee, Jurgens. *When "A Helpful Assistant" Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of LLMs*. **Findings of EMNLP 2024**. arXiv [2311.10054](https://arxiv.org/abs/2311.10054) | 162役割（対人関係6類型×専門8領域）、4モデル系列、事実問題2,410問。**ペルソナを足しても精度は上がらない**。"the effect of each persona can be largely random"。最良ペルソナの自動選択は "often performing no better than random selection" | ①性格ラベルだけで精度向上を期待しない。②**ペルソナの「選択」自体がノイズに弱い**ので、選抜には独立セットでの確認が必須 |
| Kim, Yang, Jung. *Persona is a Double-edged Sword: Mitigating the Negative Impact of Role-playing Prompts in Zero-shot Reasoning Tasks*. arXiv [2408.08631](https://arxiv.org/abs/2408.08631)（2024） | Llama3 では12データセット中7つで、役割演技プロンプトが推論を劣化させた。ペルソナ回答と中立回答の両方を作り評価器で選ぶ（Jekyll & Hyde）と改善した | **中立（ペルソナなし）対照**を必ず置き、能力税（capability tax）を測る |
| Hu, Rostami, Thomason. *Expert Personas Improve LLM Alignment but Damage Accuracy* (PRISM). arXiv [2603.18507](https://arxiv.org/abs/2603.18507)（2026-03） | 専門家ペルソナは整合（好み・安全）を改善するが、識別的課題の精度を損なう。ゲート付き LoRA への自己蒸留で精度を保つ（MMLU の具体値は前回の文献調査による。abstract では未確認） | ペルソナの能力税は既知の現象。v4 の自己生成データによる LoRA 化と整合する |
| Baines+. *Persona Cartography: Charting Language Model Personality Traits in Weight Space*. arXiv [2607.07916](https://arxiv.org/abs/2607.07916)（2026-07） | OCEAN 特性ごとの LoRA（4B〜32B の6モデル）。ほぼ加法的に合成でき、中程度の強さなら能力を保つ。**協調性（agreeableness）軸が sycophancy を、神経症傾向軸が frustration を動かす** | 協調性の高い役割は議論で同調しやすいという予測になる。候補プールに「同調しにくい役割」と「協調的な役割」の両方を入れ、データで確かめる価値がある |
| Keluskar, Bhattacharjee, Liu. *When Does Personality Composition Matter for Multi-Agent LLM Teams?* **COLM 2026**. arXiv [2606.27443](https://arxiv.org/abs/2606.27443) | 性格の効果は課題の構造に依存する。低協調性は言語を大きく変えるが、**構造化された課題（コーディング）の成果にはほとんど効かず**、非構造的な課題（研究協働・交渉）では劣化させる | 本研究の課題（多肢選択・数学）は構造化されている。性格より推論手続きを操作化すべき根拠 |
| Duan+. *The Power of Personality: A Human Simulation Perspective to Investigate LLM Agents*. arXiv [2502.20859](https://arxiv.org/abs/2502.20859)（2025） | Big Five の付与で閉じた課題の推論精度が有意に変わる。多エージェントでは性格の組合せが集合知を左右する | 組合せ（構成）を選ぶ価値の傍証 |
| Wang+. *Unleashing the Emergent Cognitive Synergy in LLMs* (Solo Performance Prompting). **NAACL 2024**. arXiv [2307.05300](https://arxiv.org/abs/2307.05300) | 複数ペルソナの自己協働で効果が出るのは GPT-4 のみで、GPT-3.5 と Llama2-13b では出ない | **4B 級では複雑なペルソナ指示は守られにくい**。プロンプトは1〜2文の手続き的な記述にとどめ、手続きの遵守を簡易に計測する（§5.4） |

### 1.3 人間の問題解決チームの役割理論

- **Belbin のチーム役割**
  - 書誌: Belbin, R. M. (1981). *Management Teams: Why They Succeed or Fail*. Heinemann。9役割目の Specialist は Belbin (1993) *Team Roles at Work* で追加された。
  - 9役割は3群に分かれる。
    - 思考系: Plant（独創的な発想）、Monitor Evaluator（冷静な批判的評価）、Specialist（専門知識）
    - 行動系: Shaper（推進・挑戦）、Implementer（実務的な遂行）、Completer Finisher（誤りの点検・仕上げ）
    - 対人系: Co-ordinator（目標の明確化・統合）、Teamworker（協調・調停）、Resource Investigator（外部の着想・機会の探索）
  - Belbin の中心主張は「役割がバランスよくそろったチームは成果が高い」。
  - 実証的妥当性は次のとおり、**留保付き**である。
    - Furnham, Steele & Pendleton (1993), *J. Occup. Organ. Psychol.* 66:245–257: 自己評価尺度（BTRSPI）の α 係数は高くなく、因子構造も明確でない（Belbin の反論、Furnham らの再反論が同誌にある）。
    - Senior (1997), *J. Occup. Organ. Psychol.* 70(3):241–258: 11チームの調査で「役割バランスと成果の関連」に一部支持。
    - Aritzeta, Swailes & Senior (2007), *J. Manage. Stud.* 44(1):96–118（DOI 10.1111/j.1467-6486.2007.00666.x）: 43の実証研究をもとに構成概念妥当性を総括。
  - **扱い方**: Belbin を「正しい法則」として採用するのではない。「人間の問題解決チームの役割を網羅する、広く使われた分類」として、**候補を恣意的でなく網羅的に作るため**に使う。
- **Six Thinking Hats**
  - 書誌: de Bono (1985), *Six Thinking Hats*, Little, Brown.
  - 情報（白）、感情・直観（赤）、慎重・批判（黒）、楽観・利点（黄）、創造（緑）、進行管理（青）の6つの思考モード。
  - LLM への適用例: PTFA（arXiv [2503.12499](https://arxiv.org/abs/2503.12499)、合意形成の司会エージェント）。2026年の MAD 戦略サーベイ（arXiv [2607.26212](https://arxiv.org/abs/2607.26212)）も、Six Hats の役割演技を役割付与の一類型として挙げている。
  - 本提案の Belbin 由来プールが Six Hats の各モードを覆うこと（§4.2）を、補助的な根拠にする。
- **Hong & Page (2004)**: *Groups of diverse problem solvers can outperform groups of high-ability problem solvers*. PNAS 101(46):16385–16389（DOI 10.1073/pnas.0403723101）。
  - 多様な集団から無作為に選んだチームが、個人成績上位者のチームを上回ることがある。候補プールが大きくなると、上位者どうしが似てしまうためである。
  - Thompson (2014, *Notices AMS* 61:1024–1030) による数学的批判と、Kuehn (2017, *Critical Review* 29(1)) による反論がある。
  - **本研究への含意**: 「単独成績の上位3人」を選ぶ（A1 型）のではなく、**チームとしての寄与で選ぶ**ことの古典的な動機になる。修論では、批判があることも併記する。
- **LLM への Belbin の適用**
  - IMACS（Chen+, arXiv [2607.25446](https://arxiv.org/abs/2607.25446), 2026-07）が Belbin 型の役割を宣言的なチーム構成に取り入れている。ただし abstract には、役割構成と精度の関係を示す実験結果がない（未確認）。
  - **Belbin の9役割を推論手続きに操作化し、チーム貢献度で選抜する研究は見当たらなかった**（2026-09-30 の検索範囲）。本研究の世代0の設計は、それ自体が小さな新規性になりうる。

### 1.4 ペルソナ・役割の自動生成・選抜・進化

| 手法 | 何を探索／選抜するか | 評価のしかた | 本研究との関係 |
|---|---|---|---|
| **DyLAN**: Liu, Zhang, Li, Liu, Yang. **COLM 2024**. arXiv [2310.02170](https://arxiv.org/abs/2310.02170) | 候補エージェント（役割プロンプト）からのチーム選抜 | 予備試行での **Agent Importance Score**（貢献の逆伝播、無監督）で選抜する。MMLU の一部科目で最大25%改善 | 「候補プールから貢献度でチームを選ぶ」直接の先行例。本研究はその貢献度を**厳密 Shapley＋正解付き dev**で測る |
| Agents that Matter（arXiv 2605.27621）／MoCA（arXiv 2605.24048） | 低貢献エージェントのモデル差し替え（LOO）／相補性による貪欲選択 | LOO・相補性 | 前回の文献調査（`literature_update.md` b-1）で確認済み。選抜基準の比較対象 |
| **EvoAgent**: Yuan+. **NAACL 2025**. arXiv [2406.14228](https://arxiv.org/abs/2406.14228) | エージェント設定（ペルソナ等）の変異・交叉・選択で多エージェントを自動生成 | タスク性能 | プロンプト空間の進化。本研究は世代0をプロンプトで選び、世代1以降は**重み（LoRA）空間**で進化する二段構え |
| **MASS**: Zhou+. **ICLR 2026**. arXiv [2502.02533](https://arxiv.org/abs/2502.02533) | 局所プロンプト → トポロジー → 全体プロンプトの段階的探索 | 検証セットの性能 | 「プロンプト（役割）とトポロジーの両方が決定的」。役割プロンプトを最適化対象として扱う正当化になる |
| AgentVerse（arXiv [2308.10848](https://arxiv.org/abs/2308.10848)、ICLR 2024 採録とされる。abs では未確認）／AutoAgents（**IJCAI 2024**, arXiv [2309.17288](https://arxiv.org/abs/2309.17288)） | 課題に応じた専門家の募集・生成 | タスク性能 | LLM に役割を生成させる流儀。再現性と恣意性の点で、理論由来の固定プールより説明しにくい |
| GPTSwarm（Zhuge+, **ICML 2024**, arXiv [2402.16823](https://arxiv.org/abs/2402.16823)）／ADAS（Hu, Lu, Clune, arXiv [2408.08435](https://arxiv.org/abs/2408.08435)。ICLR 2025 採録とされるが未確認） | エージェントの計算グラフ・設計そのもの | 検証性能 | 構成探索の一般化。予算上、本研究では扱わない |
| PromptBreeder（Fernando+, arXiv [2309.16797](https://arxiv.org/abs/2309.16797)。ICML 2024 とされるが abs では未確認）／**EvoPrompt**（Guo+, **ICLR 2024**, arXiv [2309.08532](https://arxiv.org/abs/2309.08532)） | プロンプトの進化 | 検証性能 | 世代0のペルソナ文の微調整に使えるが、過適合と恣意性の懸念がある。本提案は文面を固定し、組合せだけを選ぶ |

**評価のしかたについての教訓**
- ペルソナの自動選択は、独立セットで確認しないとランダム選択並みになりうる（Zheng+ 2024）。
- DyLAN や MASS も検証セットで選んでいる。
- よって、**選抜用と確認用に dev を分けること**、**事前に決定規則を固定すること**が必須である。

### 1.5 役割を操作化するための推論手続き（確立したプロンプト手法）

| 手続き | 文献（確認済み） | 対応させる Belbin 役割 |
|---|---|---|
| 原理・定義に一段抽象化してから解く | Step-Back Prompting: Zheng+ **ICLR 2024**, arXiv [2310.06117](https://arxiv.org/abs/2310.06117)（MMLU 物理 +7%、化学 +11%） | Specialist |
| 計画を立て、小問に分けて順に解く | Plan-and-Solve: Wang+ **ACL 2023**, arXiv [2305.04091](https://arxiv.org/abs/2305.04091)／Least-to-Most: Zhou+ **ICLR 2023**, arXiv [2205.10625](https://arxiv.org/abs/2205.10625) | Co-ordinator |
| 解いた後に逆向きに検証する | Self-Verification: Weng+ **Findings of EMNLP 2023**, arXiv [2212.09561](https://arxiv.org/abs/2212.09561) | Completer Finisher |
| 類似の問題・事例を自分で想起して適用する | Analogical prompting: Yasunaga+ **ICLR 2024**, arXiv [2310.01714](https://arxiv.org/abs/2310.01714) | Resource Investigator |
| 計算をプログラム的な手順に切り出す | Program of Thoughts: Chen+ **TMLR 2023**, arXiv [2211.12588](https://arxiv.org/abs/2211.12588)（DMAD でも使用） | （コード実行環境がないため不採用。必要なら Implementer の変種） |
| 複数の解釈・解法を比べる | DMAD の多様化の考え方、Liang+（EMNLP 2024, 既引用）の divergent thinking | Plant |
| 各選択肢を条件に照らして消去する | 一般的な消去法（MCQ） | Monitor Evaluator |
| 仮説を早く立て、自分で反証を試みる | 反証主義的な検証。低協調性（Persona Cartography）の操作化 | Shaper |

---

## 2. 本研究のデータとの整合（追加費用なし）

- 7月の c7 ペルソナ（日本語の性格1文）間の回答不一致は、同一エージェントの再サンプリングと ±1pp で同じだった。チームの oracle@3 は、ベースの3サンプルの oracle@3 を下回った（`reanalysis.md` §2.2, §4.2）。
  - これは Zheng+ 2024（ペルソナ効果はほぼランダム）と Keluskar+ 2026（構造化課題では性格が効かない）に合致する。
  - つまり**7月のペルソナは多様性の源になっていなかった**。v4 の世代0でペルソナを推論手続きとして定義し直す根拠を、自分たちのデータで示せる。
- 議論では、少数派の正解の95%が多数派に同調していた。これは Wu+ 2025 の "majority pressure suppresses independent correction"、MachineSoM の同調、Pappu+ 2026 の integrative compromise と同じ現象である。同調しにくい役割（Shaper、Monitor Evaluator）を候補に入れる動機になる。

---

## 3. 提案 (a): 候補プール

### 3.1 作り方（修論に書く手順）

1. **理論**: Belbin の9チーム役割を**すべて**候補にする。取捨選択をしないことで、恣意性の指摘を避ける。
2. **操作化**: 各役割が問題解決チームで果たす機能を、**確立したプロンプト手法に対応する推論手続き**として1〜2文で記述する（§1.5）。性格語（温厚、大胆など）は最小限にする（§1.2 の知見）。
3. **対照**: **ペルソナなし（plain）**を1体加える。能力税と、ペルソナが再サンプリング以上の多様性を生むかを測るためである。3体の plain チーム（乱数だけ異なる）も対照にする。
4. **形式**: 英語。4B 級でも守れるよう短く手続き的に書く（SPP の知見）。システムプロンプトは「ペルソナ文＋回答書式の指示」で、これは v4 の既存形式と同じである。

### 3.2 候補（10体 = Belbin 9役割 + plain）

番号は §4 の実験計画 AG(2,3) の点番号（id = 3x + y）である。x = 群（0 = 思考系、1 = 行動系、2 = 対人系）、y = 群内番号。

| id | Belbin 役割（和名） | 群 | 英語ペルソナ案（system prompt に入れる文） | 対応する手法 | 元の3人との対応 |
|---|---|---|---|---|---|
| 0 | Plant（創造者） | 思考 | "You are a creative idea generator. Before settling on an answer, consider at least two different interpretations of the question or two different ways to solve it, then choose the best-supported one." | 解法・解釈の多様化（DMAD / divergent thinking） | explorer |
| 1 | Monitor Evaluator（監視評価者） | 思考 | "You are a critical evaluator. Test each answer option or intermediate claim against the conditions of the problem one by one, and eliminate those that fail before committing to an answer." | 消去法・批判的評価 | critic |
| 2 | Specialist（専門家） | 思考 | "You are a domain specialist. First state the key principle, definition, or formula that governs this problem, then apply it step by step." | Step-Back | — |
| 3 | Shaper（形づくる人） | 行動 | "You are a decisive challenger. Commit early to the most likely answer, then deliberately try to refute it by attacking its weakest assumption, and change it only if the refutation succeeds." | 仮説→反証（低協調性の操作化） | — |
| 4 | Implementer（実行者） | 行動 | "You are a disciplined implementer. Apply the most standard, direct method systematically, keep every step concrete, and sanity-check the result with a quick estimate." | 標準解法＋概算検算 | pragmatist |
| 5 | Completer Finisher（完成者） | 行動 | "You are a meticulous finisher. Solve the problem, then verify your answer backwards against every condition (substitute it back or re-derive it another way) and fix any error you find." | Self-Verification | — |
| 6 | Co-ordinator（調整者） | 対人 | "You are a coordinator. Break the problem into a short plan of sub-questions, answer them in order, and combine the partial results into the final answer." | Plan-and-Solve / Least-to-Most | — |
| 7 | Teamworker（チームワーカー） | 対人 | "You are a cooperative team player. Explain your reasoning plainly so that others can follow and check it, and resolve the most likely misunderstanding of the question before answering." | 明示的説明・誤解の解消（高協調性。同調しやすいと予測される。データで検証する） | — |
| 8 | Resource Investigator（資源探索者） | 対人 | "You are a resourceful investigator. Recall a similar problem, example, or well-established fact you are confident about, and adapt its solution to this problem." | Analogical prompting | — |
| P | plain（対照） | — | （ペルソナ文なし。回答書式の指示のみ） | — | — |

- 元の3人（critic / pragmatist / explorer）は、**Monitor Evaluator / Implementer / Plant** に対応する。これを「理論上の既定トリオ（default trio）」と呼ぶ。Belbin の分類では思考系2＋行動系1であり、3群のバランスが取れた構成ではない点に注意する（§4 で検証できる）。
- v4 の現行英語プロンプト（`src/evo4/prompts.py` の PERSONAS）は、この3役割とほぼ同じ内容である。**既定トリオには現行文をそのまま使ってよい**。表の案と差し替える場合は、事前登録前に1つに決める。

### 3.3 Six Thinking Hats による補助的な網羅性の確認

| 帽子 | 思考モード | プール内の対応役割 |
|---|---|---|
| 白 | 事実・情報 | Specialist（原理の明示）、Resource Investigator（既知の事実の想起） |
| 赤 | 直観 | Shaper（直観で仮説を立ててから反証） |
| 黒 | 慎重・批判 | Monitor Evaluator、Completer Finisher |
| 黄 | 利点・肯定 | Teamworker（相手の筋道を活かす） |
| 緑 | 創造 | Plant |
| 青 | 進行管理 | Co-ordinator |

→ Belbin の9役割は Six Hats の6モードをすべて覆う。2つの独立した理論から見て、候補プールに偏りがないことを示せる。

---

## 4. 提案 (b): 選抜手続き（予算 約$5）

### 4.1 設計の考え方

- **本研究のチーム貢献度と同じ原理で選ぶ**: 単独精度の上位3人（A1 型）ではなく、「チームに入れたときの精度への寄与」で選ぶ。
- 三つ組を全列挙すると C(9,3)=84 通りで予算を超える。**釣り合い型不完備ブロック計画**を使う。アフィン平面 AG(2,3)（Steiner 三重系 S(2,3,9)）の12本の直線を三つ組として評価する。
  - 各役割はちょうど4チームに出る。
  - どの2役割もちょうど1回だけ同じチームになる。
  - これにより役割の主効果（平均寄与）を偏りなく推定できる。
- 点の番号付けで「群」を x に揃えると、12チームは次の2種類に分かれる。
  - 群内で同質な3チーム（x が一定の直線）
  - **3群から1人ずつのバランス型9チーム**（残りの3平行類の直線）
  - つまり、Belbin の中心主張「役割バランスが成果を上げる」を同じ予算内で検証できる。

**12チーム（id は §3.2）**

| 平行類 | 三つ組 | 型 |
|---|---|---|
| x 一定 | {0,1,2} = {PL, ME, SP} / {3,4,5} = {SH, IMP, CF} / {6,7,8} = {CO, TW, RI} | 群内同質（3チーム） |
| y 一定 | {0,3,6} = {PL, SH, CO} / {1,4,7} = {ME, IMP, TW} / {2,5,8} = {SP, CF, RI} | バランス |
| y = x + c | {0,4,8} = {PL, IMP, RI} / {1,5,6} = {ME, CF, CO} / {2,3,7} = {SP, SH, TW} | バランス |
| y = 2x + c | {0,5,7} = {PL, CF, TW} / {1,3,8} = {ME, SH, RI} / {2,4,6} = {SP, IMP, CO} | バランス |

これに**既定トリオ {PL, ME, IMP}**（計画外。元の3人）と、**plain×3 チーム**を加えて、計14チームを評価する。

### 4.2 手順（事前登録する内容）

**dev の分割**: dev 400問を、ベンチ別に層化して固定 seed で2つに分ける。
- 選抜用 A: MMLU-Pro 75 / SuperGPQA 75 / MATH 50 = 200問
- 確認用 B: 同数の200問

**段階1（単独スクリーニング、A）**
- 10候補と plain の追加乱数系列2本が、A で round0 を生成する（v4 プロトコル）。計 12×200 = 2,400 生成。
- 測るもの:
  - 単独 macro 精度と、plain との対応あり差（問題クラスタのブートストラップ 90%CI）
  - 回答分布の多様性: plain の再サンプリングを基準にした不一致率の超過分
  - 手続きの遵守率（§4.4）
- **能力税フィルタ**（事前登録）: plain より macro で 2pt 以上低く、かつ 90%CI の上限が 0 未満の候補は、最終選択から除外する。ただし段階2の計画は崩さず評価する（釣り合いを保つため）。

**段階2（チーム評価、A）**
- 14チームについて、v4 プロトコルの round1 を生成する（round0 は段階1の保存分を共有）。計 14×3×200 = 8,400 生成。
- チームの macro 精度 `acc(T)` を計算する。
- **役割の主効果**: 計画の12チームに加法モデル `acc(T) = μ + Σ_{i∈T} α_i + ε` を最小二乗で当てはめ、`α_i` を得る。
  - 釣り合い型計画なので、`α_i` は「役割 i を含む4チームの平均 − 全体平均」に比例する推定量になる。
  - これは、全員が3人チームに入る人口ゲームにおける**平均限界寄与（Banzhaf 的な寄与）の回帰版**である。本研究の Shapley 適応度と同じ「チームへの寄与」の考え方に立つ。
- **Belbin のバランス仮説の検定**（副次・探索的）: バランス型9チームと同質型3チームの平均 macro 精度の差を、並べ替え検定で調べる。

**段階3（確認と決定、B）**
- 最終候補を B で評価する。
  - F1: 段階2で観測 macro が最大のチーム（除外役割を含まないもの）
  - F2: 主効果 `α_i` の上位3役割からなるチーム（F1 と異なる場合）
  - F3: 単独精度の上位3役割からなるチーム（A1 型の対照。F1・F2 と異なる場合）
  - F0: 既定トリオ {PL, ME, IMP}
  - C: plain×3 チーム
- 各チームの round0 と round1 を B で生成する。最大 5チーム×（3 round0 + 3 round1）×200 ≒ 6,000 生成。ただし重複する役割の round0 は共有する。
- 採用チームについて、B で**厳密 Shapley**（7連合: 対の round1 を追加で 3×2×200 = 1,200 生成）を計算し、世代1以降の選抜と同じ尺度で各役割の寄与を報告する。
- **決定規則**（事前登録）: B の macro 精度が最大の最終候補 F* を選ぶ。ただし F* が既定トリオ F0 を**+1.5pt 以上**上回り、かつ対応ありの差の **90%CI の下限が 0 を超える**ときだけ F* を採用する。それ以外は F0（元の3人）を採用する。
  - この規則により、どちらに転んでも「データで確認した選択」になる。
  - 検出力が足りない場合は理論上の既定に戻る、保守的な規則である。
- 世代0の社会は採用したトリオとし、その3つのペルソナ文を**全系統（S / N / A1）・全世代で固定**する。LoRA はそのペルソナの下で社会学習する。

### 4.3 費用見積（A100 Spot の実効 $2.2/h、出力 ~2,500 tok/s 前提）

| 段階 | 生成数 | 出力トークン概算 | 時間 | 費用 |
|---|---|---|---|---|
| 1 単独（A） | 2,400（round0、平均~1,500 tok） | 3.6M | ~0.4h | ~$0.9 |
| 2 チーム（A） | 8,400（round1、平均~600 tok、入力~4K） | 5.0M ＋ prefill | ~0.8h | ~$1.8 |
| 3 確認（B）＋ Shapley | ~7,200 | ~6M | ~0.8h | ~$1.8 |
| **計** | ~18,000 | ~15M | **~2h** | **~$4.5** |

パイロットで実測したスループットで再計算する。並列を上げて 3,000 tok/s 以上出れば $4 を下回る。

### 4.4 ノイズで選ばないための注意

1. **選抜（A）と確認（B）を分ける**。最終判断は B の値だけで行う。A の最大値は勝者の呪いで上振れしている（7月の進化ログでは、選抜時の最大値が次世代の再測定で平均 −3.2pt 下がった）。
2. **対応あり比較**: すべての候補とチームを同じ問題で評価し、問題 ID クラスタの差で比べる。vLLM の seed は共通乱数として働かないので、seed による分散削減は期待しない。
3. **主効果の推定は4チームの平均**なので、単一チームの観測値より分散が小さい。単独精度・主効果・チーム精度の3つを必ず併記し、どれで選んだかを明示する。
4. **macro 精度**（ベンチ等重み）で選ぶ。問題数の多いベンチに引きずられないためである。
5. **手続きの遵守を測る**（4B 級では指示が守られにくい）。出力に役割の手続きの痕跡があるかを、正規表現で簡易計測して報告する。痕跡の例:
   - CF: 回答後の "verify" や "substitute" などの検算節
   - SP: 冒頭の "principle" や "formula" の明示
   - PL: 複数の解法・解釈の列挙
   - ME: 選択肢ごとの判定
   遵守率が plain と変わらない役割は「操作化に失敗した」と記録する。
6. **効果が小さい可能性を事前に認める**: 文献上、ペルソナ効果は小さく不安定である（Zheng+ 2024）。段階3で既定トリオが残った場合も「ペルソナの組合せの差は dev で ±1.5pt 未満」と数値つきで報告できる。
7. **世代0の選択は test を一切見ない**。test は v4 の主要評価専用とする。

### 4.5 世代1以降との関係（修論の筋）

- 世代0: **プロンプト空間**での役割選抜。候補は Belbin の9役割、基準はチーム寄与。
- 世代1以降: 同じチーム寄与（厳密 Shapley）で、**重み空間（LoRA）**の子を選抜する。
- どちらの段階でも「チームとして役に立つか」で選ぶので、手法の一貫性を示せる。
- 段階3の F3（単独上位3人）と F1/F2（チーム寄与）の比較は、世代0における RQ3（チーム貢献 vs 単独成績）の予備的な証拠にもなる（探索的と明記する）。

---

## 5. 修論での書き方と、審査で想定される質問

### 5.1 手法章の文例（第3章「世代0のペルソナ」の節）

> 世代0の社会を構成するペルソナは、人間の問題解決チームの役割理論である Belbin のチーム役割（9役割）\cite{belbin1981,belbin1993} から網羅的に候補化した。各役割は、その役割がチームで果たす機能を、確立したプロンプト手法に対応する推論手続き（Step-Back \cite{zheng2024stepback}、Plan-and-Solve \cite{wang2023plan}、Self-Verification \cite{weng2023verification}、類推 \cite{yasunaga2024analogical} 等）として操作化し、英語1〜2文のシステムプロンプトとした。性格語のみのペルソナは客観課題の精度をほとんど変えず、効果の向きも不安定であること \cite{zheng2024persona,kim2024persona}、一方で議論では推論手続きの多様性が精度に寄与すること \cite{liu2025dmad,wu2025debate} による。候補の中から、本研究のチーム貢献度の考え方に基づき、アフィン平面 AG(2,3) による釣り合い型の12チーム評価で各役割の平均寄与を推定し、独立した dev 部分集合で確認した上で3役割を選んだ（事前登録した決定規則は付録）。

### 5.2 想定質問と回答

| 質問 | 回答の骨子 |
|---|---|
| なぜ Belbin か。妥当性に疑問があるのでは | 人間のチーム役割を網羅する代表的な分類として、**候補を恣意的に作らないため**に使った。妥当性の留保（Furnham+ 1993、Aritzeta+ 2007）は認識しており、**選択そのものはデータで行った**。Six Hats でも網羅性を確認した |
| ペルソナで精度が本当に変わるのか | 文献上、性格ラベルだけの効果は小さく不安定である。そこで推論手続きとして操作化し、**plain 対照で能力税と多様性を測った**。7月のデータでも性格1文は再サンプリングと区別できなかった |
| 3つの選び方はノイズでは | 選抜と確認を分け、決定規則を事前登録し、確認で明確に勝たなければ理論上の既定に戻る。選抜時の値と確認時の値の両方を報告する |
| 4B で役割が守られるのか | 手続きの遵守率を出力から計測して報告した（§4.4-5）。守られない役割はそう明記した |
| なぜ3体か | 厳密 Shapley が全7連合の実測で近似なしに計算できる規模（本研究の中核）であり、Du+ 2023 の標準構成でもある。人数を増やす効果は今後の課題とする |
| プロンプト進化（EvoPrompt 等）で最適化しないのか | 文面を最適化すると dev への過適合と恣意性が増える。本研究は文面を理論由来で固定し、**組合せ**だけをチーム寄与で選んだ。文面の最適化は、世代1以降の LoRA 空間の進化が担う |

---

## 6. 未確認・要フォロー

- DMAD の集計図の定義（どのベンチの平均か）と、個別ベンチ（MATH、GPQA）の数値: 本文 PDF が取得上限を超えたため未確認。スライドの集計図と mental set の表だけを引用した。
- ChatEval の「同じ役割記述で性能が落ちる」の該当表・数値は未精査（検索要約による）。
- PRISM の MMLU の具体値（前回の文献調査による。abstract では未確認）。
- AgentVerse（ICLR 2024）・ADAS（ICLR 2025）・PromptBreeder（ICML 2024）の採録は arXiv abs に記載がなく未確認（一般にそう引用されている）。
- MachineSoM における「特性より協調戦略が効く」という相対比較は、abstract には明示がない（本文未確認）。
- IMACS（2607.25446）の Belbin の具体的な操作化と、精度への効果は abstract では不明。

## 7. refer.bib 追加候補（キー案）

- 議論と多様性: `liu2025dmad`（ICLR 2025）、`he2025dimo`（arXiv 2510.16645）、`wu2025debate`（arXiv 2511.07784）、`dynamicrole2026`（arXiv 2601.17152）、`pappu2026experts`（ICML 2026）、`chen2024reconcile`（ACL 2024）、`hegazy2024diversity`（JRAR 2024）、`chan2024chateval`（ICLR 2024）、`zhang2024machinesom`（ACL 2024）
- ペルソナの効果: `zheng2024persona`（Findings of EMNLP 2024）、`kim2024persona`（arXiv 2408.08631）、`hu2026prism`（arXiv 2603.18507）、`baines2026cartography`（arXiv 2607.07916）、`keluskar2026personality`（COLM 2026）、`duan2025personality`（arXiv 2502.20859）、`wang2024spp`（NAACL 2024）
- 人間のチーム役割理論: `belbin1981`、`belbin1993`、`furnham1993belbin`、`senior1997`、`aritzeta2007belbin`、`debono1985`、`hong2004diverse`、`thompson2014`、`kuehn2017`
- 自動生成・選抜: `liu2024dylan`（COLM 2024）、`yuan2025evoagent`（NAACL 2025）、`zhou2026mass`（ICLR 2026）、`chen2024autoagents`（IJCAI 2024）、`guo2024evoprompt`（ICLR 2024）
- 推論手続き: `zheng2024stepback`（ICLR 2024）、`wang2023plan`（ACL 2023）、`zhou2023least`（ICLR 2023）、`weng2023verification`（Findings of EMNLP 2023）、`yasunaga2024analogical`（ICLR 2024）、`chen2023pot`（TMLR 2023）
