# 文献調査アップデート（2026-07-05 → 2026-09-29）— 進化v4・チーム選抜の再設計に向けて

**調査日**: 2026-09-29 / **担当**: 文献調査フォーク / **前回**: `docs/literature_notes_v3.md`（2026-07-05）
**方法**: WebSearch で候補を洗い出し、**arXiv abs ページ（一部は HTML 本文）を WebFetch で開いて**タイトル・著者・日付・採録状況・主要数値を確認した。確認できなかった項目は「未確認」と明記している。数値は原典の abstract/本文からの転記で、再計算はしていない。
**前提（本研究側の既知事実）**:
- 旧進化の初期集団は「SFT 個体＋2%ノイズの複製」だけで、遺伝的多様性がゼロだった。
- 適応度はサンプリングノイズに支配されていた（同一個体の solo 精度が±5pt 揺れ、ほぼ同一重みでも予測不一致率が0.28〜0.45）。
- fitness sharing は K=2 では無効だった。
- 最終チーム c7 は SC@9 に −3.5pt（MMLU-Pro −2.4 / MATH −0.8 ns / SuperGPQA −5.5）で負けた。

---

## 0. エグゼクティブサマリ（設計判断に直結する結論）

1. **新規性の最大の脅威は、v3 調査が見落としていた Heterogeneous Swarms（Feng+, NeurIPS 2025, arXiv:2502.04510）**。この研究はマルチLLMシステム内での各モデルの個別貢献（JFK-score）を指標に、**モデル重みを群知能（PSO）で最適化**している。「チーム貢献で重み空間の集団最適化を誘導する」という大枠は、すでに NeurIPS に先行例がある。**修論 Sec1 の「著者の知る限り報告されていない」「初の定式化」は、そのままでは書けない**。残る差分は次の組合せに絞って主張し直す必要がある。
   - 厳密 Shapley（周辺貢献）
   - 議論（debate）を特性関数にすること
   - 役割別サブ集団の協調的共進化＋ΔW 交叉の世代交代 GA
   - 生成回数マッチ SC との統制比較
2. **「チーム貢献を学習信号にする」研究は 2026 年に RL 側で急増した**。SHARP(2602.08335)、C3(2603.06859)、CCPO(2603.21563)、Marginal-Contribution PG(2604.22785)、COSAC(2604.17693)、MADA-RL(2607.18006) などがある。ただしいずれも**勾配ベース**で、LOO/反実仮想による近似が主流である。**勾配なしの集団選抜で、厳密 Shapley を使う点は依然として空白**。
3. **「チーム構成・メンバー選抜を貢献度で決める」研究も層が厚くなった**。対象はプロンプト／モデル選択のレベルで、DyLAN(COLM 2024)、Agents that Matter(2605.27621: LOO で低貢献エージェントのモデルを差し替えて最大+17%)、AgentSlimming、HiveMind(AAAI 2026, DAG-Shapley)、MoCA(2605.24048)、AC/DC(ICLR 2026: カバレッジでチーム選抜) などがある。**静的なチーム選抜研究を行うなら、これらが直接の先行研究かつベースライン**になる。
4. **同一モデル Qwen3-4B-Instruct-2507 について、集約系手法は多肢選択の SuperGPQA で多数決に勝てていない**。RSA 原著の表では、多数決 48.2 > RSA(T=10) 47.39 > RSA(T=1) 45.91 > ベース 41.85 で、著者自身が「多肢選択形式では多数決が特に強い」と明記している。これは我々の「SuperGPQA で SC@9 に −5.5pt」と整合する。**MCQ 知識系で SC を超える主張は文献上も最も難しい**。勝負どころは数学（MATH-500）である。
5. **2026年5〜9月の報告は「小型モデルでは同一コストの反復サンプリングが強い」で一貫**している。
   - 1.5B〜7B で Self-Refine/Reflexion はいずれも同トークンの反復サンプリングに勝てない（2607.28576）。
   - 多数決は disjunctive 課題の潜在利得（oracle）をほぼ回収できない（2609.31563）。
   - 最初の投票が全員一致で誤っていたケースが、議論が効いたケースの66%を占める（2605.09618）。
6. **議論ラウンドは「候補に正解が2つ以上あるときだけ効き、全員誤りのときは逆効果」**（2608.18379: **Qwen3-4B**、AIME/HMMT。新規に解き直す場合と比べ、正解候補が2つ以上なら+0.290、全員誤りなら−0.123）。議論（候補を見せる更新）をゲートする設計に直接使える。
7. **適応度ノイズへの処方は文献で確立済み**。
   - 同一問題での予測ノイズ（paired prediction noise）は、問題標本のノイズ（data noise）を通常上回る（Sida Wang 2512.21326）。反復平均または貪欲デコードで下げるべき。
   - ES at Scale（ICML 2026）は適応度評価に**貪欲デコード**を使っている。
   - 共通乱数（同一シード）は正の相関があるときに分散を下げる（2512.24145）。
   - 選抜後の報告には勝者の呪い補正が要る（SIREN 2605.05973、隠れた選抜の感度 2609.28177）。
   - 逐次停止で評価コストを57〜97%削れる（optstop 2608.14425）。
8. **ノイズのある特性関数には Shapley より Banzhaf 値が頑健**（Data Banzhaf, AISTATS 2023: 準値の中で最大の safety margin）。3体・7連合の実測から**追加コストゼロで両方計算できる**ため、v4 ではアブレーションとして必須級。
9. **「多様性指標」は能力と絡み合っていて、チーム利得を予測しない**（2607.20768: MMLU-Pro で30 LLM の3体部分集合を調べると、多数決が最強メンバーを上回るのは**9.98%のみ**。同時正解の代理指標と「1−平均精度」の相関は ρ=+0.991）。回答不一致率で測った旧 sharing は、能力とサンプリングノイズの混合を測っていた可能性が高い。**チーム成果を直接測る適応度（Shapley/周辺貢献）の正当化材料**になる。
10. **初期集団の多様性と交叉の成否**について。
    - 独立シードの LoRA を素朴に内挿すると崩壊する（CoTo, ICML 2025: λ=0.5 で通常の LoRA は39%、CoTo は79%）。
    - 共通の種アダプタから学習すると線形モード連結になる（Seeded LoRA）。
    - PopuLoRA の変異幅はテンソル標準偏差の0.1〜0.15倍で、旧実装の0.02の5〜7倍。SVD 系変異・成分マスク・外挿交叉（係数>1）も使い、「保持テスト」で有害な演算子を除いている。
11. **ペルソナ付与は知識系の精度を削る（再確認）**。
    - 専門家ペルソナで MMLU が一貫して低下する（PRISM 2603.18507）。
    - 文脈（他者の出力など）が推論を最大74%短縮し、自己検証を減らす（Reasoning Shift 2604.01161）。これは議論ラウンドでの CoT 圧縮の説明候補になる。
    - 推論を保持する LoRA として NB-LoRA（2609.25618）が出た。
12. **評価環境の系統差**（我々の+6pt 発見）を裏付ける報告が増えた。
    - vLLM の attention backend や prefix caching の設定が精度にも影響する（2607.09172）。
    - GPU アーキテクチャ間で出力が非決定的になる（2609.25624）。
    - 修論の方法論的貢献として関連研究に位置づけられる。

---

## (a) 新規性評価 — 本研究の核「チーム貢献 Shapley 適応度 × LoRA 集団進化」に対する最近接研究

### a-1. 比較表

| 研究 | 最適化対象 | 貢献・適応度の測り方 | 相互作用プロトコル | 本研究との差分 | 脅威度 |
|---|---|---|---|---|---|
| **Heterogeneous Swarms**（Feng, Wang, Goyal ほか。arXiv:2502.04510, **NeurIPS 2025**） | DAG の役割（隣接行列）＋**モデル重み**。PSO で最適化 | **JFK-score**: 最良 DAG の各位置へモデルをランダムに割り当てるのを M 回繰り返し、効用を登場頻度で重み付けして平均する（周辺貢献ではない） | DAG 上のトポロジカルなメッセージ伝達（議論ではない） | 次の5点で差がある。(i) 厳密 Shapley（周辺貢献）か、平均効用型の JFK か。(ii) 議論型の特性関数か、DAG か。(iii) 役割固定の協調的共進化＋ΔW 交叉 GA か、PSO か。(iv) 共有 4B ベース上の3人格 LoRA か、Gemma-7B の専門家10体か。(v) SC@k との統制比較・統計設計の有無 | **高**（「チーム貢献で重み集団を最適化」の大枠は先行） |
| **AC/DC**（Dai, Meinardus, Regan, Tian, Tang。arXiv:2604.14969, **ICLR 2026**） | モデル集団（タスクベクトル内挿＋特異値ノイズ変異）とタスク集団の共進化 | 支配的新規性探索（DNS）とスキルベクトル。**チームはカバレッジ最大化で事後選抜** | 複数エージェントの best-of-N（judge 選択） | 貢献の周辺性（Shapley）を使わず、選抜は事後的。議論ではない。報告される利得は N=3 で専門家比+1.34% | **中** |
| Model Swarms（Feng+, ICML 2025、既引用） | LoRA 専門家の重み（PSO） | 個体効用 | なし（単体） | チーム適応度なし | 中（既引用） |
| GENOME/GENOME+（Zhang+, **ACL 2026 main**、既引用） | LLM 重み（交叉・変異・選択・継承・アンサンブル） | 個体の少数ショット精度 | アンサンブル（投票） | チーム適応度なし。初期集団は専門家重みのランダム線形結合で多様化 | 中（既引用） |
| PopuLoRA（Creus Castanyer+, 2605.16727、既引用） | LoRA 集団（教師/生徒）。勾配学習と演算子の混成 | TrueSkill の下側信頼限界（対戦的 self-play） | 教師が出題し生徒が解く（競争的） | 協調チームではない。勾配学習を併用 | 中 |
| RL 系の credit 割当: SHARP（2602.08335）/ C3（2603.06859）/ CCPO（2603.21563）/ 周辺貢献 PG（2604.22785）/ COSAC（2604.17693）/ MADA-RL（2607.18006） | ポリシー重み（GRPO 等。MADA-RL は LoRA） | LOO・マスキング・反実仮想の継続・リッジ分解など。SHARP は名称こそ Shapley だが、実装は R(τ)−R(τ∖m) のマスキング差 | planner–worker、Think–Solve、generator–critic | 勾配ベースで、集団選抜ではない。厳密 Shapley ではない | 中（「チーム貢献を学習信号に」という着想は一般化した） |
| エージェント選抜・帰属: DyLAN（COLM 2024）/ Agents that Matter（2605.27621）/ AgentSlimming（2605.08813）/ HiveMind（AAAI 2026）/ SelfOrg（ICLR 2026）/ MoCA（2605.24048） | チームのメンバー選択、プロンプト、モデル差し替え | peer 評価の逆伝播（DyLAN）、LOO（Agents that Matter, AgentSlimming）、DAG-Shapley（HiveMind）、Shapley 近似（SelfOrg）、相補性の貪欲選択（MoCA） | 議論・DAG・提案者＋要約者 | 重みを進化させない。**静的チーム選抜研究のベースラインとして必須** | 中 |
| CoPES（Wang+, 2608.02391, AAAI 2027 投稿） | パラメータ部分空間を協調的共進化する ES（Qwen3.5-4B、原文表記） | 課題報酬 | 単一エージェント | 「協調的共進化」の意味が異なる（部分空間であり役割・エージェントではない）。用語の区別が必要 | 低〜中 |
| 周辺貢献適応度による SNN アンサンブル共進化（Rodriquez & Ghawaly, 2606.13985, ACM ICONS 2026） | SNN 集団 | **グループ性能への周辺貢献**（固定連合サイズの差分評価） | アンサンブル | 領域違い（LLM ではない）。ただし着想は同型 | 低（ただし「周辺貢献適応度は古典的な考え方」と示す材料） |
| EvE（Yu & Yang, 2605.09018） | エージェントの指針・スキル（テキスト） | ソルバ集団への周辺利得に基づく Elo | 競走（racing） | 重みではない | 低 |
| SAT（Pappu+, 2609.22682） | 協調戦略（役割・フェーズ・情報流）を学習 | 課題精度 | 異種フロンティアモデルのチーム | 重みではない。**考察章の材料（demonstrability）**として重要 | 低（新規性）／高（考察） |

### a-2. 結論（新規性の言い直し案）

- **依然として未報告（2026-09-29 の探索範囲）**: 次の組を同時に満たす研究は見つからなかった。
  - 「3体の全7連合を実測する**厳密 Shapley**」を、
  - 「**議論（debate）**の多数決精度を特性関数」として定義し、
  - 「**役割別サブ集団の協調的共進化**（代表チーム文脈）」の下で、
  - 「**LoRA の ΔW 空間交叉と変異による世代交代 GA**」の適応度に用い、
  - 「**生成回数マッチ SC@k と、問題クラスタを考慮した統計**」で評価する。
- **既に先行があり、「初」とは言えない要素**:
  - チーム内の個別貢献で重み空間の集団最適化を誘導すること → Heterogeneous Swarms（NeurIPS 2025）
  - 貢献（LOO/反実仮想/Shapley 近似）を学習・選抜の信号にすること → 2026年の RL・MAS 文献多数
  - 周辺貢献を共進化の適応度にすること → 古典的な差分評価（Agogino & Tumer, Evol. Comput. 2008；Wolpert & Tumer の collective intelligence）と SNN 共進化（ICONS 2026）
- **修論への反映（必須）**:
  1. Sec1 の課題(a)「…議論チームへの貢献を選択圧として…重みを進化させる研究は報告されていない」を修正する。**Heterogeneous Swarms・AC/DC・RL 系 credit 割当を明示的に引用**したうえで、差分（厳密性・議論・GA/共進化・統制評価）に限定して主張する。
  2. 貢献1の「初の定式化」は「3体構成での**厳密** Shapley を**議論**の特性関数で定義し、LoRA 集団 GA の適応度とした定式化」に絞る。あわせて差分評価・周辺貢献適応度の古典系譜の一例として位置づける。
  3. Sec2 の表2.2（位置づけ）に Heterogeneous Swarms、AC/DC、SHARP/C3、Agents that Matter、MoCA の行を追加する。
  4. 本研究の実証的な売りは「手法の新規性」単独ではなく、次の3点に置く。**勝てなかった条件の統制的な同定**、**選抜ノイズ（勝者の呪い）の定量化**、**評価環境の系統差の発見**。これらは 2026 年の評価統計系の文献（2512.21326、2605.05973、2605.30315、2609.28177）とも接続できる。

---

## (b) テーマ別の新文献リスト

凡例: 日付は arXiv v1 の投稿日（改訂があれば併記）。「引用章」は修論のどの章に効くか。★は2026年7〜9月の新規。

### b-1. チーム貢献・credit 割当・エージェント帰属

| arXiv / 書誌 | 著者 | 日付 / 採録 | 要点 | 本研究との関係 | 引用章 |
|---|---|---|---|---|---|
| 2502.04510 Heterogeneous Swarms | Feng, Wang, Goyal ほか | 2025-02-06（改訂 2025-10-22）/ **NeurIPS 2025** | 役割（DAG）と重みを PSO で交互に最適化。JFK-score は最良 DAG へのランダム割当を M 回行い、効用を頻度重み付き平均したもの。15ベースラインに対し12タスク平均+18.5%（MMLU-Pro・GSM8K 等を含む）。専門家は Gemma-7B を Tulu-v2 の10領域で微調整した10体（HTML 本文で確認。LoRA かフル重みかは本文で明示なし）。**計算量マッチの SC 比較はなく**、投票系は「Prediction Merge（多数決）」の1ベースラインのみ | **最近接の先行研究**。貢献指標で重み集団を最適化する大枠が一致。一方で**生成回数マッチ SC との統制比較がない**点は我々の差分になる | 1, 2.3, 2.5 |
| 2602.08335 SHARP | Yanming Li ほか | 2026-02-09（v2 2026-06-02）/ arXiv | planner–worker 型 MAS を GRPO で学習。報酬は大域・「Shapley 周辺 credit」・ツール過程の3種。credit の実装は R(τ)−R(τ∖m) のマスキング差（LOO 型）。Qwen3-8B / LLaMA-3.1-8B。単一エージェント比+23.66%、MATPO 比+14.05% | RL 側で「Shapley」を名乗る最近接例。厳密計算ではない | 2.4 |
| 2603.06859 C3 "The Trace Is the State: Exact Credit Assignment for LLM Agent Teams" | Yanjun Chen ほか | 2026-03-06（v3 2026-09-26）/ arXiv | 1メッセージを差し替えて終端報酬まで継続する反実仮想 credit。16継続の参照解との順位相関0.69。MAGRPO を上回り、MAPPO 比でトークン−37%。※他論文の引用中に「Exact Is Easier…」という旧題らしき表記があるが**未確認** | 「厳密な credit」を名乗る RL 手法。我々の「厳密 Shapley」との語の衝突に注意 | 2.4 |
| 2603.21563 CCPO / SEPO | Zhongyi Li ほか | 2026-03-23（改訂 2026-06-11）/ arXiv | エージェント除去の反実仮想で周辺貢献を報酬化。2エージェント Think–Solve。MATH500 で改善 | 同上 | 2.4 |
| 2604.22785 Marginal-Contribution PG | Ben-Gal, Tong | 2026-04-03（v2 2026-09-08）/ arXiv | 共有報酬は各エージェントの私的効用を最適化してしまうことを証明。周辺貢献による勾配補正の信号を導出。routed GSM8K を追加生成コストなしで改善 | 「共有報酬ではだめで周辺貢献が要る」ことの理論的裏付け | 2.4, 3.3 |
| 2604.17693 COSAC | Deshmukh ほか | 2026-04-20（v2 2026-05-09）/ arXiv | 逐次協調チームの critic-free 反実仮想 credit（リッジ回帰で加法分解）。Qwen3-0.6B×4で ARC | 周辺の関連研究 | 2.4 |
| ★2607.18006 MADA-RL | Pulici ほか | 2026-07-20 / TMLR 査読中 | 小型モデル（DeepSeek-R1-Distill-Qwen-1.5B）を LoRA で generator/critic に特化させる。critic の利得を generator アンサンブル精度に対する反実仮想ベースラインで定義。39.9→41.9%（+2.0pt、p<0.001） | **LoRA×小型×議論×反実仮想 credit** の RL 版。勾配学習の対照として重要 | 2.1, 2.4, 6 |
| 2310.02170 DyLAN | Zijun Liu ほか | 2023-10-03（v2 2024-11-15）/ **COLM 2024** | Agent Importance Score（後続ノードの peer 評価を逆伝播で集計）で候補からチームを選抜（MMLU では7→4体、科目により最大+25%）。Shapley との関係は論文中で言及なし | **チームメンバー選抜の古典的先行**（プロンプト級） | 2.4, 2.5 |
| 2605.27621 Agents that Matter | Mingyu Lu, Yushan Huang, Chris Lin, Su-In Lee | 2026-05-26 / arXiv | エージェント帰属を協力ゲームとして定式化。**LOO は組合せ的手法と同等にボトルネックを特定でき、はるかに安価**。v1 の abstract は「低貢献エージェントの基盤モデルを差し替えて最大+17%・コスト−35%（3ベンチ）」。現行版の題は "…How Subtle Choices Shape Agent Attribution…" に変わっている | 貢献に基づく**モデル差し替え**＝我々の「選抜」に最も近い応用。LOO と Shapley の比較軸を提供 | 2.4, 4（ベースライン） |
| 2605.08813 AgentSlimming | Yulang Chen ほか | 2026-05-09 / arXiv | LOO を Shapley の代理とし、低重要度ノードから貪欲に剪定・量子化。トークン最大−78.9% | 同上（効率化目的） | 2.4 |
| 2512.06432 HiveMind | Yihan Xia ほか | 2025-12-06 / **AAAI 2026** | DAG-Shapley（非成立連合を公理的に剪定）で LLM 呼び出し80%以上削減。Shapley 貢献でプロンプトを自動改善（株取引 MAS） | Shapley を「最適化の信号」に使う最近接（プロンプト級） | 2.4 |
| 2510.00685 SelfOrg | Tastan, Horváth, Nandakumar | 2025-10-01（改訂 2026-03-09）/ **ICLR 2026** | 応答の Shapley 近似評価から通信 DAG を毎ラウンド構成。弱いモデルほど効く | 議論内の貢献評価の応用例 | 2.1, 2.4 |
| 2607.18255 Semantic Cooperative Games | Pengyi Jiang, Xiaoguang Zhu, Quanyan Zhu | abs には「Submitted 2026-05-14」と表示されるが ID（2607）と整合しない。**要再確認** / arXiv | 意味的 Shapley（SSV）で、連合の再実行なしに帰属を計算。モンテカルロ Shapley 比で計算コスト−93.3%。帰属・解釈が目的で、最適化は行わない | 連合を再実行しない Shapley の可能性（我々の 7連合実測との対比） | 2.4 |
| 2603.29871 ShapE-GRPO | Rui Ai ほか | 2026-03-31 / arXiv | 集合レベル報酬を Shapley 公理を保ったまま多項式時間で候補別に分解 | Shapley 分解の効率計算例 | 2.4 |
| 2604.09459 Credit assignment サーベイ | Chenchen Zhang | 2026-04-10 / arXiv | LLM の RL における credit 割当の体系的レビュー（69本） | RL 側を1本でまとめて引用するのに使える | 2.4 |
| ★2606.13985 周辺貢献適応度による SNN アンサンブル共進化 | Rodriquez, Ghawaly | 2026-06-12 / **ACM ICONS 2026** | グループ性能への周辺貢献を適応度とする共進化。単体進化や事後アンサンブルより有意に良く、制御課題で質的な差が出た | **周辺貢献適応度の共進化**が他領域で有効だった例（LLM への適用が我々の差分） | 2.4, 5 |
| 古典: Agogino & Tumer, "Efficient Evaluation Functions for Evolving Coordination" | Agogino, Tumer | **Evolutionary Computation 16(2):257–288, 2008** | 大域評価と整合し、かつ自成分に敏感な成分評価関数（差分評価）の設計 | Shapley/LOO 適応度の**理論的な系譜** | 2.4, 3.3 |
| 古典: Wiegand, Liles, De Jong（GECCO 2001, pp.1235–1242） | — | 2001 | 協調的共進化における協力者の選び方（最良・ランダム・複数）の実証比較 | 代表チーム文脈の評価設計の根拠 | 3.4 |
| 古典: Panait, Luke, Wiegand, "Biasing Coevolutionary Search for Optimal Multiagent Behaviors" | — | **IEEE TEC 10(6):629–645, 2006** | 協調的共進化の病理（相対的過汎化）と、最適協力者推定による楽観バイアス | 代表文脈評価の弱点と対策 | 3.4, 5 |

### b-2. 進化・集団ベースの LLM/LoRA 最適化（演算子・初期集団・ES）

| arXiv / 書誌 | 著者 | 日付 / 採録 | 要点 | 本研究との関係 | 引用章 |
|---|---|---|---|---|---|
| 2605.16727 PopuLoRA（演算子の詳細） | Creus Castanyer ほか | 2026-05-16 / arXiv（v1 のみ） | 共有凍結ベース（Qwen2.5-Coder-7B）上の rank-32/α=64 LoRA 集団。**変異**は6種類ある。M1: SVD 特異値への対数正規ノイズ σ=0.1 と Cayley 回転。M2: 33%のスロットに ε=0.1×std のガウスノイズ。M3: 特異成分の30%をゼロ化。M4: 全テンソルに ε=0.15×std。M5: NEFTune 型。M6: ランク摂動。**交叉**は9種類（DARE、層単位、SVD 部分空間、**外挿（係数>1）**、線形、TIES、DELLA、SLERP、Fisher）。17演算子を試し、**保持テスト**（子が10〜20更新で親の水準に戻るか）に落ちたものは除外。10ステップごとに TrueSkill の LCB 下位25%を置換。HumanEval+ 78.7 vs 76.8、LCB 27.0 vs 17.3 | **変異幅**（0.1〜0.15×std。旧実装は0.02）、SVD 系変異、外挿交叉、保持テストの直接の参考 | 2.3, 3.4 |
| 2604.14969 AC/DC | Dai, Meinardus, Regan, Tian, Tang | 2026-04-16 / **ICLR 2026** | タスクベクトル内挿による交叉と、**特異値ノイズによる変異**。LLM アーカイブ（DNS）と合成タスクの共進化。カバレッジ（誰か1体が正解する割合）で N 体チームを選抜。N=3 で専門家比+1.34%、対照比+0.99% | 集団を作りチームを選ぶ流れの最近接。**oracle 型カバレッジ vs Shapley** の対比軸 | 2.3, 2.5 |
| 2410.14735 CycleQD | Kuroki, Nakamura, Akiba, Tang | 2024-10-16（v4 2025-02-17）/ **ICLR 2025** | QD（品質と行動記述を循環的に入れ替え）に、マージ交叉と **SVD 変異**を組み合わせる。Llama3-8B でエージェントスキルを獲得 | QD×重みマージの代表例（MAP-Elites 系の具体化） | 2.3, 2.4 |
| 2503.01155 GENOME/GENOME+ | Yiqun Zhang ほか | **ACL 2026 main**（Proc. 64th ACL Vol.1 pp.44187–44205, DOI 10.18653/v1/2026.acl-long.2044） | 既引用。**採録情報を更新**。初期集団は専門家重みのランダム線形結合で作る | bib 更新（(d)参照） | 2.3 |
| 2604.17753 ENMP | Anda Cao ほか | 2026-04-20 / **ACL 2026 main** | LoRA マージ時に害をなす「負のモジュール」を進化探索で除外する（既存マージ法へのプラグイン） | ΔW 交叉における層・モジュール単位の選択的継承の根拠 | 2.2, 3.4 |
| ★2609.18560 "The evolution of sex for AI" | Giorgio F. Gilestro | 2026-09-16 / arXiv | 集団遺伝学の枠組み。**重みの単純平均は逆効果で、親は各自の強い成分を選択的に寄与させるべき**（Fisher–Muller 効果）。モデル崩壊を遺伝的浮動とみなす。相反する規約を学んだ系統間ではマージが失敗する（生殖的隔離） | 内挿交叉の限界と、選択的・モジュール単位の組換えの理論的な裏付け | 5 |
| 2506.05713 CoTo | Zhuang ほか | 2025-06-06 / **ICML 2025** | 同一課題を**異なるシードで独立学習した LoRA** をパラメータ空間で線形内挿すると、λ=0.5 で通常の LoRA は39%、CoTo は79%（常識推論、LLaMA）。確率的な活性化スケジュールで線形モード連結を改善 | **独立シード LoRA の交叉は素朴には壊れる**。初期集団設計の必須知識 | 2.2, 3.4 |
| Seeded LoRA | Salamanca, Üstün, Detlefsen, Dettmers | **ICML 2024 Workshop** | 小データで作った共通の種アダプタから初期化すると、以降の独立学習が同じ最適化部分空間に入り、マージ可能になる | **v4 初期集団は共通の種（run002 アダプタ）から分岐させる**案の根拠 | 3.4 |
| 2509.24372 ES at Scale | Qiu, Gan, Hayes ほか | 2025-09-29（v3 2026-07-14）/ **ICML 2026** | 全パラメータの ES を10億規模で実行。**N=30、σ=0.001、α=5e-4**、z スコア正規化。**適応度評価は貪欲デコード**。Countdown で Qwen2.5-7B が ES 66.8 vs GRPO 57.5 | **勾配なし・小集団・決定的評価**で LLM を最適化できる実例 | 2.3, 3.4 |
| ★2607.19408 ES の集団規模スケーリング | Sung Cho, Gyubin Han | 2026-07-08 / ICML 2026 HiLD WS | 二値報酬 ES で N=2 が失敗するのは z スコア正規化のせいだった（外すと N=2 でも改善。Qwen2.5 0.5B〜7B、GSM8K・TREC）。優位性ゼロとなる確率を閉形式で導出 | 小集団（K=3〜4）でも選抜が成立しうる根拠と、正規化の注意 | 3.4 |
| ★2608.27351 ES vs GRPO の推論カバレッジ | Yunpeng Ba ほか | 2026-08-27 / arXiv | ES は GRPO よりエントロピー崩壊が少なく pass@k が広い。**大きいモデルほど小さい集団で足りる**。更新は疎で大きく、忘却を避ける | 集団探索の多様性利点 | 2.3 |
| ★2608.05541 Hyper-ES | Yu Gu ほか | 2026-08-06 / arXiv | 少数の安価な勾配学習で「降下方向」を得て、それが張る空間で**層単位の DARE-TIES 合成係数を CMA-ES で最適化**。GRPO-LoRA 比+1%、勾配更新−10% | **「多様な LoRA を安価に作り、合成係数を進化させる」**という v4 の現実的な設計の直接の先行例 | 3.4 |
| 2507.04453 ESSA | Korotyshova ほか | 2025-07-06（v4 2026-09-09）/ arXiv | LoRA 因子を SVD し、**特異値だけ**を ES で最適化。MATH500 の目標精度到達が LoRA-GRPO より最大7.8倍速い（128 GPU） | 変異空間を特異値に絞る設計 | 3.4 |
| 2511.16652 EGGROLL | Sarkar ほか（20名） | 2025-11-20 / arXiv | ES の摂動をランク r 行列として構造化し、学習速度を100倍に | 低ランク摂動の効率化 | 2.3 |
| ★2608.02391 CoPES | Zhiyuan Wang ほか | 2026-08-03 / AAAI 2027 投稿 | パラメータ部分空間を協調的共進化する ES（Qwen3.5-4B、原文表記、ツール使用エージェント）。同じ GPU 時間で GRPO の利得の92%を回収（標準 ES は67%）、メモリは1/8未満 | 「協調的共進化×LLM 重み」の同名別概念。用語の区別が必要 | 2.3 |
| ★2609.18779 CERA-MoA | Jiaxuan Jiang ほか | 2026-09-16 / arXiv | **4B ベース上で4エージェントに個別の LoRA**（総計約4.4B）。ルータとエージェントを RL で共進化させる | 共有4B＋個別 LoRA の MAS という**アーキテクチャが同型**。重みの進化・Shapley はない | 2.1, 2.3 |
| ★2605.09018 EvE | Zongmin Yu, Liu Yang | 2026-05-09（v4 2026-08-17）/ arXiv | 指針とスキルの集団を進化させ、ソルバ集団への周辺利得で Elo を更新（racing） | 周辺利得＋racing の組合せ例 | 2.3 |

### b-3. ノイズ下の選抜・評価統計（racing / IRT / 共通乱数 / 勝者の呪い）

| arXiv / 書誌 | 著者 | 日付 / 採録 | 要点 | 本研究との関係 | 引用章 |
|---|---|---|---|---|---|
| 2512.21326 "Measuring all the noises of LLM Evals" | Sida Wang | 2025-12-24（v2 2026-03-29）/ arXiv | 総分散の法則で、予測ノイズ・データノイズ・総ノイズを分離。全ペアの paired 法で**同一問題での予測ノイズがデータノイズを通常上回る**ことを示した。反復予測の平均で検出力が大きく上がる | **旧進化の適応度がサンプリングノイズ支配だったことの一般的な裏付け**。v4 の評価設計（反復・貪欲・共通乱数）の根拠 | 4.4, 5.3 |
| 2512.24145 "When Does Pairing Seeds Reduce Variance?" | Udit Sharma | 2025-12-30（改訂 2026-01-31）/ arXiv | 共通乱数（シード対応）は、結果がシード単位で正に相関するときに分散を削減する | v4 で「シードを**エージェント同一性×問題**に紐づける」根拠 | 3.4, 4.4 |
| 2605.30315 Resolution Diagnostics | Anany Kotawala | 2026-05-28 / ICML 2026 WS (Hypothesis Testing) | 解像度比 q=N/N*。MMLU-Pro 上位10の隣接ペア9組のうち4組が未解決。一般的な計算機（Cohen's h）は接近比較で N* を約2倍誤る | 適応度セットの必要問題数を見積もる道具 | 4.4 |
| ★2605.05973 SIREN | Yang Xu ほか | 2026-05-07 / arXiv | 適応的チューニング後の「勝者スコア」は楽観的になる。探索後の候補リストを凍結し、分割ごとに選抜と評価を分離し、問題単位の乗数ブートストラップで CI を付ける。MMLU-Pro でのチューニング実験を含む | **選抜後の報告を選抜考慮型にする方法**（旧 dev+12pt → test±0 の現象そのもの） | 4.4, 5.3 |
| ★2609.28177 隠れた選抜への感度 | Chen Yang, Jun Chen | 2026-09-23 / arXiv | Open LLM Leaderboard の隣接順位の主張394件のうち391件が、選抜を考慮する前から統計的な支持を欠く。候補間の相関に依存する感度曲線を提案 | 「何個試したか」の開示と、感度分析の根拠 | 4.4 |
| 2601.13885 連続スコアの適応評価 | Balkır ほか | 2026-01-20（v2 2026-09-14）/ **EMNLP 2026 main** | IRT 適応テストを連続スコアへ拡張し、不確実性つき順位付けと適応停止を行う。較正後は**2%の問題**で済む | racing/逐次選抜の実装指針 | 3.4 |
| ★2608.14425 optstop | Toby D. Pilditch | 2026-08-14 / arXiv | 階層ベイズで「不確実な所だけ追加評価する」逐次停止。計画試行を57〜97%削減し、結論は全評価と同等 | 適応度評価の予算配分（successive halving の代替） | 3.4 |
| 2402.14992 tinyBenchmarks / 2502.10436 MERGE³ | Polo ほか / — | ICML 2024 / ICML 2025（既出） | IRT で厳選した100問で誤差約2%。適応度計算を50倍削減 | 既出（v3 設計で採用済み） | 3.4 |
| 2205.15466 Data Banzhaf | Jiachen T. Wang, Ruoxi Jia | 2022-05-30（改訂 2023-12-18）/ **AISTATS 2023 (Oral)** | 学習の確率性で効用がノイズを持つと、Shapley や LOO の順位は不安定になる。**Banzhaf 値は準値の中で safety margin（順位の頑健性）が最大** | **ノイズのある特性関数下では Shapley より Banzhaf**。3体では追加コストゼロで比較できる | 3.3, 4, 5 |
| 古典: Jin & Branke, "Evolutionary Optimization in Uncertain Environments—A Survey" | Yaochu Jin, Jürgen Branke | **IEEE TEC 9(3):303–317, 2005** | ノイズ適応度下の進化（再サンプリング・再評価・母集団規模） | 雑音進化の標準的な引用 | 2.4, 5.3 |
| 古典: Rakshit, Konar, Das, "Noisy evolutionary optimization algorithms – A comprehensive survey" | — | **Swarm and Evolutionary Computation 33:18–45, 2017** | 雑音 EA の手法分類（明示的・暗黙的な平均化、選抜の修正など） | 同上 | 2.4 |
| 古典: Caruana ほか, "Ensemble Selection from Libraries of Models" | Caruana, Niculescu-Mizil, Crew, Ksikes | **ICML 2004** | ライブラリからの前向き段階選択。hillclimb 集合への過適合と対策（置換あり選択・bagging） | **静的チーム選抜研究の古典ベースライン** | 2.4, 4 |

### b-4. MAD vs SC・集約・sycophancy・異質性（2026年6〜9月を中心に）

| arXiv / 書誌 | 著者 | 日付 / 採録 | 要点 | 本研究との関係 | 引用章 |
|---|---|---|---|---|---|
| 2509.26626 RSA（数値の再確認） | Venkatraman, Jain, Mittal, Shah, Obando-Ceron, Bengio, Bartoldson, Kailkhura, Lajoie, Berseth, Malkin, Jain | 2025-09-30（v2 2026-02-24）/ arXiv | **Qwen3-4B-Instruct-2507、SuperGPQA、予算マッチ（N×T 生成）の Pass@1**: ベース41.85 / 棄却サンプリング46.18 / 自己改良43.5 / **多数決48.2** / RSA(T=1, K=4)45.91 / RSA(T=10, N=16, K=4)47.39（SuperGPQA のみ1シード）。著者は「多肢選択では多数決が特に有効」と明記。K=1→2 の利得が最大。集約を意識した RL で集約時の性能が上がる | **同一モデル・同一ベンチで、集約系は多数決に負ける**という原著の証拠。我々の SuperGPQA の負けと整合（我々の測定値もベース0.431〜0.436 / SC@9 0.486 で近い水準） | 2.1, 5.4 |
| ★2608.18379 Candidate-free control | Guiv Farmanfarmaian | 2026-08-18 / COLM 2026 WS (Efficient Reasoning) | **Qwen3-4B**、AIME-2025 と HMMT-2025。同じ予算で新規に解き直す対照と比べ、正解候補が2つ以上なら**+0.290**、**全員誤りなら−0.123**、正解1つなら結論不定 | **議論ラウンド（候補を条件にした更新）が効く・効かない条件**。ゲート設計の根拠 | 3.2, 5.2 |
| ★2609.31563 Disjunctive vs Compensatory | Fortuna, Bertalanič | 2026-09-25 / arXiv | Steiner の課題類型で分析（13モデル、最大30体）。disjunctive 課題ではチーム拡大で oracle が5〜20pt 増えるが、**単純多数（plurality）投票は単体比0.5pt 以内**しか回収しない。多ラウンド改訂は大きく効き、**相手1体でも29体でもほぼ同じ利得**。compensatory 課題（フェルミ推定）では項目共通のバイアスが二乗誤差の約87%を占める | 我々の**集約損失**（oracle−チーム）と同型の一般的な知見。「相手1体で十分」は3体設計の正当化にも使える | 2.1, 5.1, 5.2 |
| ★2609.22682 SAT（Self-Organizing Agent Teams） | Pappu, Suzgun, Kwon, Bianchi, El, Kochenderfer, Cao, Zou | 2026-09-19 / arXiv | 少数の問題（数学15、大学院25）から協調戦略を学習。チーム66.7% vs 最良メンバー48.8% / 計算量マッチ58.7% / 完全ルータ59.0%。**demonstrability**（10モデルの審査団が正しいチーム証明を選ぶ率）と利得の相関は ρ=0.90（p=0.005）。同質チーム56.0% vs 異質66.7% | 格下げした「検証可能性仮説」を**ベンチ単位の demonstrability として操作化した先行例**。考察を立て直すのに使える | 5.1 |
| ★2605.09618 Statistical Scouting（matched ceiling） | Julia Hu ほか | 2026-05-10 / arXiv | Llama3.1-8B / Ministral-3-8B、960トークン上限。問題ごとの最適プロトコル oracle は+14pt だが、事前に選べる制御器は投票エントロピー閾値のみ（+1.3/+1.7pt）。**議論が効いたケースの66%は初期投票が全員一致で誤り** | 「全員一致なら議論を省く」設計は利得の大半を捨てる、という警告 | 3.2, 5.2 |
| ★2607.28576 "Sample More, Reflect Less" | Iliya Mirzaei | 2026-07-30 / arXiv | 1.5B/3B/7B、数学2種（各150問）、36比較。**同一トークンの反復サンプリングより確実に良い手法は1つもない**。自己点検18比較はすべて負 | SC 系が小型モデルで強いことの最新の再確認 | 2.1, 5.4 |
| ★2608.11403 "When Self-Consistency Backfires" | Utkarsh Bahuguna | 2026-08-11（v2 08-15）/ COLM 2026 WS | GPQA-Diamond で、多数決は**問題ごとの正答率を Qwen2.5-7B では56.6%、Llama-3-8B では65.7%の問題で下げる**。トークンエントロピーや多数一致率のゲートは機能しない | SuperGPQA の hard 帯で議論が+17pt（旧分析）だったことと、問題単位では SC が逆効果になりうることの対応 | 5.1 |
| ★2608.18795 Wrong-consensus の分解 | Lizhuo Zhang ほか | 2026-08-19（改訂 08-31）/ arXiv | 一致を「機械的成分（問題ごとの回答選好）」と残差に分解。**多肢選択（GPQA-D）では一致の81〜93%が機械的成分**、AIME では59〜78% | MCQ で SC・議論が誤った合意に陥りやすい構造の説明 | 5.1 |
| 2606.29270 Minority Sentinel | Chuan He ほか | 2026-06-28 / SIGIR 2026 AgentSearch WS | 議論ログの特徴量から LightGBM で「多数決を覆すべきか」を判定。意見が割れたケースの約25%で少数派が正解、理論的な回収余地は10pt、**反転の精度は81.2%で、6データセット×20シードすべてで純利得が正**。LLM-as-judge は純利得が負 | **我々の大量の議論ログで学習できる安価な集約器**。集約損失への実装しやすい処方 | 3.2, 6 |
| 2603.06801 AceMAD | Yuhan Liu ほか | 2026-03-06 / arXiv | 他者の信念を予測させる peer-prediction で、正解保持者と誤った多数派の非対称性を検出し、マルチンゲールを破る | 「マルチンゲールの呪い」への介入の一形態 | 2.1 |
| ★2609.33974 CPP | Ruosong Ye ほか | 2026-09-27 / arXiv | 通信辺の条件付き漸進剪定。**「初めて同一コストで consistency 系に完全勝利」と主張**。ただしモデルは異種フロンティア（Grok-4.1 / GPT-4.1-nano / Mistral-small）で、テストは各100問。MMLU-Pro では CPP 66.33 vs Consistency-Fuse 66.17 と実質同等 | 「SC に勝った」主張の実態（異種フロンティア・小テスト）。**単一4Bベースでは再現性が疑わしい**ことの対照材料 | 2.1, 5.4 |
| ★2609.03619 R²-MAD | Xuanfa Jin ほか | 2026-09-03 / **EMNLP 2026 Findings** | 経験メモリ（議論状態に応じた検索）で事前知識を較正し、信頼度重みで peer の影響を調整。「既存の MAD より一貫して改善」（SC 比は abstract に記載なし） | 共有された誤解（shared misconception）への対策例 | 2.1 |
| ★2609.08016 議論は何を変えるか | Chen Qian | 2026-09-07（v2 09-22）/ arXiv | 議論は表明する立場を大きく変えるが（口調で一致率が50.4pt 変わる）、**回答の質は改善しない**（GlobalOpinionQA 50問、299ペア） | 議論の効果は見かけの一致に偏る、という警告 | 5.2 |
| ★2608.02827 偏った合意の創発 | Maya Okawa | 2026-08-03 / **ICML 2026** | 同調性が臨界点を超えると集団バイアスへ**相転移**する。温度サンプリングのノイズが増幅の主因で、**異質性が抑制する** | sycophancy・合意崩壊の理論（温度0.7の議論設計への示唆） | 2.1, 5.2 |
| 2606.19826 敵対的 peer 下の異種議論 | Nilayam ほか | 2026-06-18 / arXiv (cs.CR) | Llama-3.1-70B・MATH-hard で、有害な改訂率が同質パネルの89%から、誠実な異種 peer を入れると35%に下がる。敵対者がいると90%に戻る | 異質性は防御にも脆弱性にもなる | 2.1 |
| 2605.00914 Cost of Consensus（書誌更新） | Bertalanič, Fortuna | 2026-04-29 / ACM Conference on AI and Agentic Systems（arXiv Comments による） | 7〜8B の10体×3ラウンド。同調率は最大85.5%、oracle gap は最大32.3pt、議論は自己訂正の2.1〜3.4倍のトークン | 既引用（**venue を追記**） | 2.1 |
| 2605.29116 Trace-level synthesis | Fadnavis ほか | 2026-05-27 / arXiv | 集約の単位は回答ではなく推論トレースであるべき。**単一モデル＋入力摂動のトレース多様性が、異種モデルの集合を上回る** | Self-MoA 系。「人格の多様性」より「トレース合成」の方が効く可能性 | 2.1, 5.1 |
| 2603.20324 Selection bottleneck | Maryanskyy ほか | 2026-03-20（v2 2026-07-21）/ **Applied Sciences 16(10):4914 (2026)** | 集約（選択器）の質に交差閾値があり、多様性が効くか害になるかを分ける。judge による選択＋多様なチームは勝率0.810、同質の Self-MoA は0.512。合成型の集約は42課題中0課題でしか選ばれない | 「生成の多様性より選択器の質が効く」 | 5.2 |
| ★2607.26212 MAD 戦略サーベイ | Motger ほか | 2026-07-28 / ACM CSUR 査読中 | 分野は「静的な全結合・逐語共有・短期記憶・投票で解決」という狭い型に**慣習で収束**している | 関連研究章のまとめ引用 | 2.1 |
| 2601.22297 SDRL | Chenxi Liu ほか | 2026-01-29（v2 2026-05-17）/ arXiv | 自己議論の文脈で、初期応答と議論を条件にした応答を同時に RL 最適化する | 議論向けの事後学習 | 6 |
| 2604.24881 Latent Agents | Yi, Mueller, Lee | 2026-04-27 / **ACL 2026 main** | 議論を単一モデルに内在化。明示的な議論と同等以上の性能を、トークン最大93%減で達成。エージェント別の活性化部分空間が現れる | 議論の蒸留 | 6 |
| ★2607.13643 CANON | Gkountouras, Jukić, Titov | 2026-07-15 / arXiv | 多数派に到達した解を条件にした教師で、ラベルなし自己蒸留を行う。**pass@1 で最大+12pt**、ラベルなし RL より+6pt で計算量は1/7 | 「ラマルク型」変異（議論・合意トレースでの短い自己蒸留）の候補 | 3.4, 6 |
| 2510.01499 OW / ISP（高次情報による集約） | — | arXiv / ICML 2026（二次情報: papernotes。**要確認**） | 1次（精度）と2次（相関）の情報を使う集約（Optimal Weight / Inverse Surprising Popularity）。多数決に理論的に優る | 異質エージェントの重み付き集約の理論 | 3.2 |
| 2602.05395 ベイズ最適停止による一貫回答推定 | Jingkai Huang ほか | 2026-02-05 / ICML 2026（二次情報。**要確認**） | 上位 L=3 の回答カウントだけを追う停止則で、呼び出しを最大50%削減 | SC 側の効率化（公平比較の補助） | 4 |
| 2509.06870 AggLM（venue 確認） | Wenting Zhao ほか | 2025-09-08 / arXiv（採録情報なし） | 集約器を RLVR で学習し、少数派の正解を回収する | 既出 | 3.2 |
| 2507.17797 GenSelect（venue 確認） | Toshniwal ほか | 2025-07-23 / 2nd AI for MATH WS @ ICML 2025 | 既出（G2 で上積みなし） | 既出 | 3.2 |
| 2508.15260 DeepConf（venue 確認） | Yichao Fu ほか | 2025-08-21 / arXiv | 既出（v1 のみ、採録情報なし） | 既出 | 3.2 |

### b-5. ペルソナ SFT の能力劣化・保持／議論トレースでの事後学習

| arXiv / 書誌 | 著者 | 日付 / 採録 | 要点 | 本研究との関係 | 引用章 |
|---|---|---|---|---|---|
| 2603.18507 PRISM "Expert Personas Improve Alignment but Damage Accuracy" | Zizhao Hu, Rostami, Thomason | 2026-03-19 / arXiv | 専門家ペルソナで MMLU が一貫して低下する（ベース71.6% → 66.3〜68.0%、ペルソナの長さによる）。事前学習依存の課題が壊れ、整合依存の課題は改善する。ゲート付き LoRA 自己蒸留で精度を保持（Qwen2.5-7B で MMLU 71.7%を維持） | **ペルソナ注入の能力税**の直接証拠（我々の実験1の毀損と同じ向き） | 2, 5.2 |
| 2604.01161 Reasoning Shift | Rodionov, Garipov, Yakushev | 2026-04-01（改訂 2026-09-28）/ COLM 2026 WS | 同じ問題でも文脈（長い無関係文脈・多ターン・部分課題化）があると、**推論トレースが最大74%短くなり**自己検証が減る | 議論ラウンドで他者の出力を入れたときの CoT 圧縮を説明する候補。我々のログで round0 と round1 の長さを比べれば検証できる | 5.2 |
| ★2609.25618 NB-LoRA | Wenzhi Fang ほか | 2026-09-22 / arXiv | 推論時の活性化の近似零空間に LoRA 更新を制限する。適応性能は通常の LoRA 並みで、推論精度は学習前の水準に近いまま | リプレイ以外の能力保持手段（OPLoRA の後継筋） | 5.2, 6 |
| ★2607.07916 Persona Cartography | Baines ほか | 2026-07-08 / arXiv | OCEAN 特性ごとの LoRA（4B〜32B、6モデル）。**ほぼ加法的に合成でき**、強さに応じて特性が単調に動き、中程度の強さなら能力ベンチを維持する。**協調性（agreeableness）軸が sycophancy に効く** | ペルソナを「追従しにくさ」で設計する根拠 | 3, 6 |
| 2411.15382 "On the Impact of Fine-Tuning on CoT Reasoning" | — | NAACL 2025（検索結果による。**未精査**） | ファインチューニングが CoT に与える影響 | 補助 | 2 |
| ★2607.18006 MADA-RL | （b-1参照） | | LoRA で小型の議論役割を特化させる RL | 事後学習の対照 | 6 |

### b-6. 評価インフラの再現性（我々の+6pt の系統差の位置づけ）

| arXiv | 著者 | 日付 / 採録 | 要点 | 関係 | 引用章 |
|---|---|---|---|---|---|
| ★2607.09172 "Attention to Detail" | Zine ほか | 2026-07-10（v2 07-17）/ 投稿中 | vLLM の attention kernel と prefix caching が性能・エネルギーに主に効き、**精度にも影響しうる**（5モデル×5課題、9,000実行）。abstract に精度差の具体値はない | 評価イメージ差による系統差の傍証 | 4.6.4, 5 |
| ★2609.25624 GPU 間の非決定性 | Cooper ほか | 2026-09-22 / arXiv | 同一モデル・同一プロンプトでも GPU アーキテクチャで出力が変わる（浮動小数点の非結合性とカーネル選択）。FP32 化した GEMM で Ampere/Ada/Hopper 間をビット一致させる | 同上（**GPU 種の固定**を推奨する根拠） | 4.6.4 |

---

## (c) 再設計（進化v4・チーム選抜）に直接使える知見

### c-1. 初期集団の多様性 — 「差のある候補」を作る

1. **共通の種から分岐させる**（Seeded LoRA、PopuLoRA）。各役割の初期集団は、run002 のリプレイ再学習アダプタを種にして、次の軸で K=4〜6 体に分岐させる。
   - データの部分集合（bootstrap）
   - シード（データ順・dropout）
   - ペルソナの記述や推論戦略の変種（導出型・検証型・具体例型）
   - 学習率・エポック

   同じ最適化部分空間に保てば ΔW 交叉が意味を持つ。**独立シードから素朴に内挿すると崩壊する**（CoTo: λ=0.5 で39%）。
2. **多様性は「行動差 − ノイズ床」で検収する**。同一アダプタを別シードで再サンプルしたときの不一致率（ノイズ床）を先に測る。個体間の不一致がそれを有意に上回ることを、集団の合格基準にする（旧集団の0.28〜0.45は、ノイズ床とほぼ同じだった可能性が高い）。多様性指標は能力と交絡するので（2607.20768）、**能力を統制した対の共失敗（co-failure）**で見る。
3. **Hyper-ES 型の2段構え**も有力。安価な SFT で「降下方向」（多様な LoRA）を作り、それらの**合成係数（層単位・DARE-TIES）を進化させる**。探索空間が低次元になり、ノイズ下でも選抜が成立しやすい。

### c-2. 演算子 — 探索の不全への処方

1. **変異幅を大きくし、保持テストで上限を決める**。PopuLoRA は ε=0.1〜0.15×std（旧は0.02）。ただし PopuLoRA は変異後に勾配学習で回復させるのに対し、我々は勾配なしなので、**「行動差がノイズ床を超え、かつ solo 精度の低下が小さい」範囲を事前の小規模スイープで決める**（保持テストの勾配なし版）。
2. **SVD 系変異**を入れる。特異値への対数正規ノイズ（PopuLoRA M1 σ=0.1、AC/DC、CycleQD）と特異成分のマスク（M3 30%）。ランク構造を保ったまま「方向」を動かせる。
3. **外挿交叉**を入れる（係数>1。PopuLoRA X4 / タスク算術）。凸包の外へ出られる（旧演算子は α∈[0.3,0.7] の内挿のみ）。
4. **モジュール単位の選択的継承**を入れる（ENMP の負モジュール除外、Gilestro「平均は逆効果、強い成分を選択的に」）。層・モジュールごとにどちらの親を継ぐかを遺伝子にする（PopuLoRA X2）。
5. 実装面では、ランダム化 SVD の再分解は KnOTS / Core Space 系の「共有基底で合成する」設計が安全で、既存の `delta_blend_lora` を拡張できる。

### c-3. ノイズ制御 — 選抜の不全への処方

1. **適応度評価は決定的にする**。ES at Scale に倣い**貪欲デコード**（temperature 0）で評価するか、少なくとも**共通乱数**を使う。共通乱数ではシードを「問題 ID×エージェント同一性×ラウンド」の関数にし、チーム内の位置に依存させない（現実装は位置 i に依存。共通乱数にならない）。これで round0 の出力をチーム間で再利用でき、コスト削減も大きい。
2. **反復平均**（同じ問題を複数回サンプル）は、予測ノイズがデータノイズを上回るなら問題数を増やすより効く（Sida Wang）。ただし貪欲評価なら不要。
3. **racing / 逐次停止**を使う。successive halving（既存設計）に加えて、optstop（57〜97%削減）や適応 IRT（2%の問題）の考え方で、差が大きい候補は早く落とし、接戦にだけ問題を追加する。解像度比 q=N/N* で「判定できる差」を事前に明示する（2605.30315）。
4. **エリートを毎世代再評価する**（古典: Jin & Branke 2005）。ただし再評価値は共通乱数下で差分としてのみ使い、単独の最大値選抜はしない。**同点（差<判定可能差）は持ち越す**。
5. **Shapley と Banzhaf を併算**する（Data Banzhaf）。7連合の実測があれば Banzhaf（他2体の4部分集合に等重み）は追加コストゼロで出る。ノイズ下の順位安定性（分割半分での順位相関）を両者で比べ、アブレーションとして報告する。
6. **選抜考慮型の報告**（SIREN）。「dev で選んだ個体の test 値」は、繰り返し分割（dev/test の入替え）で分布として報告し、最終候補リストを凍結してから評価する。何個の候補を試したかを開示する（2609.28177）。

### c-4. チームレベル適応度の設計

1. **チーム成果を直接測る適応度の正当化**（2607.20768）。3体部分集合の多数決が最強メンバーを上回るのは約10%にすぎず、多様性指標は能力の代理になってしまう。チーム成果（連合の実測）に基づく Shapley/Banzhaf は、この問題を原理的に回避する。修論で強調すべき論点になる。
2. **協力者の選び方**（Wiegand 2001、Panait 2006）。代表1体の文脈だけで評価すると、相対的過汎化（協調的共進化の病理）を招きうる。v4 では(a)(b)のいずれかを入れる。
   - (a) 各候補を「代表」と「ランダム協力者」の2文脈で評価して平均する
   - (b) 楽観的バイアス（最良協力者との結果）を併用する
3. **K≥3 でないと sharing は無意味**（既知）。そもそも sharing の距離（回答不一致率）はノイズと能力が交絡するので、v4 では sharing を外し、多様性は初期集団の設計（c-1）と Shapley の周辺性に任せるのが文献上も妥当。

### c-5. 静的なチーム選抜研究（進化の前段・または代替）のデザイン

目的は「チームレベルの信号（Shapley）が、個体精度だけでは得られない選抜情報を持つか」を、進化のダイナミクスから切り離して検証すること。

- **候補プール**: 各役割 K=4〜6 の多様な LoRA（c-1）。
- **選抜則（すべて同じ dev データ・同じ共通乱数で計算）**:
  1. 役割別 solo 上位（個体適応度＝アブレーション A1）
  2. 代表文脈の厳密 Shapley
  3. Banzhaf
  4. LOO（Agents that Matter / AgentSlimming 型）
  5. 前向き貪欲の相補性選択（MoCA / Caruana 型。MoCA は400問のラベル付き集合、k=5 で AIME のトップ精度選択0.377→貪欲0.654。**ただし MMLU-Pro では0.746→0.738/0.755 とほぼ差なし**）
  6. カバレッジ最大化（AC/DC 型）
  7. dev での全探索最良
  8. ランダム
- **評価**: 選抜したチームを held-out の test で評価する（繰り返し分割・問題クラスタのブートストラップ）。事前の予想は次のとおり（文献から）。
  - 多肢選択（MMLU-Pro）では、選抜則による差が小さい（MoCA の結果）。
  - 数学では、相補性・チーム信号が効く（MoCA の AIME、SAT の demonstrability）。
- **この設計の利点**: round0 のキャッシュと共通乱数により、全チーム（K³）の dev 景観を安価に作れる。進化ループの結果を解釈する基準線（どこまで伸びうるか）も同時に得られる。

### c-6. SC に勝つための集約・議論の改良 — 文献に基づく期待値の較正

- **MCQ 知識系（MMLU-Pro・SuperGPQA）で SC@9 を超えることは、文献上も最難関**。
  - RSA 原著でも SuperGPQA では多数決が最強。
  - MCQ では誤った合意の81〜93%が機械的選好（2608.18795）。
  - 単純多数は oracle をほぼ回収しない（2609.31563）。
  - → 「全ベンチで SC 超え」を目標にするのは不適切。**主張は数学（MATH-500）に絞るか、生成回数ではなくトークン数マッチの比較を併記する**。
- **議論ラウンドはゲートする**。全員誤りの候補で条件付けると悪化する（2608.18379、Qwen3-4B）一方、「全員一致で誤り」の問題こそ議論が効く（2605.09618 の66%）。ゲートには次の2案がある。
  - 候補を見せる前に、各エージェントに1回「新規解き直し」をさせ、候補と一致したものだけを更新材料にする
  - 「候補を見せる議論」と「新規に解き直す」を半々にして両方を投票に入れる（予算は同じ）
- **学習する安価な集約器**。Minority Sentinel（LightGBM、反転精度81.2%、純利得が常に正）のように、**既存の大量の議論ログ（`results/gcs/*/llm_calls`、transcripts）から、多数決を覆すべき状況を学習**させる。追加の GPU コストはほぼゼロで、集約損失（旧 SGPQA で13pt）を直接狙える。LLM-as-judge 型が負だった点は、我々の G2（GenSelect が無効）と整合する。
- **トレース単位の合成**（2605.29116）と RSA（K=2 で最大利得）は数学系で有望。ただし MCQ には効かない。
- **事後学習による議論への適応**（ラマルク型の変異）。CANON（合意を条件にした自己蒸留で最大+12pt）、SDRL、MADA-RL（1.5B・LoRA で+2.0pt）。v4 の「変異」の一部を、短い自己蒸留ステップ（Hyper-ES の降下方向と同型）に置き換える選択肢がある。
- **公平性**: SC 側にも同じ集約器（CISC / DeepConf / 学習集約器）を与えた条件を併記する（v3 設計で既定済み）。

### c-7. ペルソナ設計

- ペルソナを「性格」ではなく**追従しにくさ・推論戦略の軸**で設計する（Persona Cartography: 協調性軸が sycophancy を左右する）。
- 能力保持はリプレイ（既存）に加えて、NB-LoRA / OPLoRA 型の部分空間制約を選択肢にする。
- 議論ラウンドの CoT 圧縮（Reasoning Shift）は、既存ログの round0 と round1 の長さ比較で**追加費用なしに検証**できる。

### c-8. 評価インフラ

- イメージ digest・vLLM のフラグ（attention backend・prefix caching・chunked prefill）・GPU 種・並列度を、全実行の JSON に記録する。**主要ベースラインは同一イメージ・同一 GPU 種で再測定**する（既存の教訓を文献で補強）。

---

## (d) refer.bib 更新提案

### d-1. 既存エントリの修正

| key | 現状 | 修正提案 | 根拠 |
|---|---|---|---|
| `zhang2025genome` | @misc, arXiv:2503.01155 | **@inproceedings, ACL 2026 main**（Proc. 64th ACL Vol.1: Long Papers, pp.44187–44205, DOI 10.18653/v1/2026.acl-long.2044） | ACL Anthology で確認 |
| `cost2026consensus` | venue なし | note に「ACM Conference on AI and Agentic Systems」（arXiv Comments による）を追記 | arXiv abs |
| `li2024more` | @misc, arXiv:2402.05120 | **@article, Transactions on Machine Learning Research (TMLR), 2024** | arXiv abs（v2 2024-10-11） |
| `maca2025preference` | @misc | そのまま（arXiv v3 2026-01-29）。OpenReview では ICLR 2026 に「Internalizing Self-Consistency in Language Models: Multi-Agent Consensus Alignment」の題で投稿された形跡がある（desk reject とする二次情報あり。**未確認**、引用はしない） | arXiv abs |
| `evopref2026evolutionary` | — | v2（2026-06-18）で採録情報なし。abstract に **LoRA 集団を NSGA-II で進化**と明記されている（本文の紹介と整合を取る） | arXiv abs |
| `evomas2026evolving` | ICML 2026 採録 | 変更なし（確認済み。v2 2026-05-27） | arXiv abs |
| `demystifying2026mad` | — | 変更なし（改訂 2026-06-03、採録情報なし） | arXiv abs |
| `creuscastanyer2026populora`, `metateam2026meta` | — | 変更なし（いずれも v1、採録情報なし） | arXiv abs |
| `talk2025cheap` | ICML 2025 MAS WS | 変更なし（改訂 2025-10-13） | arXiv abs |
| `judge2025bottleneck`, `li2025rethinking` | — | 変更なし（採録情報なし） | arXiv abs |
| Sec1 課題(a) の記述 | 「…重みを世代交代的に進化させる研究は…報告されていない」 | Heterogeneous Swarms（NeurIPS 2025）を引用し、差分を限定した記述に修正（(a) a-2 参照） | 本調査 |

### d-2. 追加 bibtex 案（優先度順。★は必須）

```bibtex
% ★最近接の先行研究
@inproceedings{feng2025heterogeneous,
  author    = {Feng, Shangbin and Wang, Zifeng and Goyal, Palash and others},
  title     = {Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-{LLM} Systems},
  booktitle = {Advances in Neural Information Processing Systems 38 (NeurIPS)},
  year      = {2025},
  note      = {arXiv:2502.04510}
}

% ★集団進化＋チーム選抜
@inproceedings{dai2026acdc,
  author    = {Dai, Andrew and Meinardus, Boris and Regan, Ciaran and Tian, Yingtao and Tang, Yujin},
  title     = {Discovering Novel {LLM} Experts via Task-Capability Coevolution},
  booktitle = {Proceedings of the 14th International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {arXiv:2604.14969}
}

@inproceedings{kuroki2025cycleqd,
  author    = {Kuroki, So and Nakamura, Taishi and Akiba, Takuya and Tang, Yujin},
  title     = {Agent Skill Acquisition for Large Language Models via {CycleQD}},
  booktitle = {Proceedings of the 13th International Conference on Learning Representations (ICLR)},
  year      = {2025},
  note      = {arXiv:2410.14735}
}

% ★エージェント選抜・帰属（静的チーム選抜のベースライン）
@inproceedings{liu2024dylan,
  author    = {Liu, Zijun and Zhang, Yanzhe and Li, Peng and Liu, Yang and Yang, Diyi},
  title     = {A Dynamic {LLM}-Powered Agent Network for Task-Oriented Agent Collaboration},
  booktitle = {Proceedings of the 1st Conference on Language Modeling (COLM)},
  year      = {2024},
  note      = {arXiv:2310.02170 (DyLAN)}
}

@misc{lu2026agents,
  author       = {Lu, Mingyu and Huang, Yushan and Lin, Chris and Lee, Su-In},
  title        = {Agents that Matter: Optimizing Multi-Agent {LLMs} via Removal-Based Attribution},
  howpublished = {arXiv:2605.27621},
  year         = {2026},
  note         = {v1 の題。現行版の題は "Agents that Matter: How Subtle Choices Shape Agent Attribution in Multi-Agent Systems"}
}

@misc{zhang2026moca,
  author       = {Zhang, Yichi and Lu, Kevin and Zhang, Yuang and Gao, Jie and Xia, Lirong and Yu, Fang-Yi},
  title        = {Mixture of Complementary Agents for Robust {LLM} Ensemble},
  howpublished = {arXiv:2605.24048},
  year         = {2026}
}

@inproceedings{caruana2004ensemble,
  author    = {Caruana, Rich and Niculescu-Mizil, Alexandru and Crew, Geoff and Ksikes, Alex},
  title     = {Ensemble Selection from Libraries of Models},
  booktitle = {Proceedings of the 21st International Conference on Machine Learning (ICML)},
  year      = {2004}
}

% ★RL 側の credit 割当（代表を数本）
@misc{li2026sharp,
  author       = {Li, Yanming and others},
  title        = {Who Deserves the Reward? {SHARP}: Shapley Credit-based Optimization for Multi-Agent System},
  howpublished = {arXiv:2602.08335},
  year         = {2026}
}

@misc{chen2026trace,
  author       = {Chen, Yanjun and Sun, Yirong and Wang, Hanlin and Wang, Jinghan and Zhang, Xinming and Shen, Xiaoyu and Li, Wenjie and Zhang, Wei},
  title        = {The Trace Is the State: Exact Credit Assignment for {LLM} Agent Teams},
  howpublished = {arXiv:2603.06859},
  year         = {2026}
}

@misc{pulici2026madarl,
  author       = {Pulici and Chu and Kharlamov and Ding and Tresp and Ma},
  title        = {{MADA-RL}: Multi-Agent Debate-Aware Reinforcement Learning for Parameter-Efficient Reasoning in Compact Models},
  howpublished = {arXiv:2607.18006},
  year         = {2026},
  note         = {著者のファーストネームは要確認}
}

% ★古典（差分評価・協調的共進化・雑音進化）
@article{agogino2008efficient,
  author  = {Agogino, Adrian K. and Tumer, Kagan},
  title   = {Efficient Evaluation Functions for Evolving Coordination},
  journal = {Evolutionary Computation},
  volume  = {16},
  number  = {2},
  pages   = {257--288},
  year    = {2008}
}

@inproceedings{wiegand2001empirical,
  author    = {Wiegand, R. Paul and Liles, William C. and De Jong, Kenneth A.},
  title     = {An Empirical Analysis of Collaboration Methods in Cooperative Coevolutionary Algorithms},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference (GECCO)},
  pages     = {1235--1242},
  year      = {2001}
}

@article{panait2006biasing,
  author  = {Panait, Liviu and Luke, Sean and Wiegand, R. Paul},
  title   = {Biasing Coevolutionary Search for Optimal Multiagent Behaviors},
  journal = {IEEE Transactions on Evolutionary Computation},
  volume  = {10},
  number  = {6},
  pages   = {629--645},
  year    = {2006}
}

@article{jin2005evolutionary,
  author  = {Jin, Yaochu and Branke, J{\"u}rgen},
  title   = {Evolutionary Optimization in Uncertain Environments---A Survey},
  journal = {IEEE Transactions on Evolutionary Computation},
  volume  = {9},
  number  = {3},
  pages   = {303--317},
  year    = {2005}
}

@article{rakshit2017noisy,
  author  = {Rakshit, Pratyusha and Konar, Amit and Das, Swagatam},
  title   = {Noisy Evolutionary Optimization Algorithms -- A Comprehensive Survey},
  journal = {Swarm and Evolutionary Computation},
  volume  = {33},
  pages   = {18--45},
  year    = {2017}
}

% ★ノイズ下の値付け・評価統計
@inproceedings{wang2023databanzhaf,
  author    = {Wang, Jiachen T. and Jia, Ruoxi},
  title     = {Data {Banzhaf}: A Robust Data Valuation Framework for Machine Learning},
  booktitle = {Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year      = {2023},
  note      = {arXiv:2205.15466}
}

@misc{wang2025noises,
  author       = {Wang, Sida},
  title        = {Measuring All the Noises of {LLM} Evals},
  howpublished = {arXiv:2512.21326},
  year         = {2025}
}

@misc{xu2026siren,
  author       = {Xu, Yang and Zhang, Jiefu and Sun, Haixiang and Zhou, Zihan and Cao, Tianyu and Aggarwal, Vaneet},
  title        = {Towards Reliable {LLM} Evaluation: Correcting the Winner's Curse in Adaptive Benchmarking},
  howpublished = {arXiv:2605.05973},
  year         = {2026}
}

@misc{kim2026diversity,
  author       = {Kim, Donghwan},
  title        = {Are Diversity Metrics Measuring Diversity? {A} Capability-Controlled Audit of Majority-Vote Gain in {LLM} Ensembles},
  howpublished = {arXiv:2607.20768},
  year         = {2026}
}

% ★MAD/集約の最新（考察で使用）
@misc{venkatraman2025rsa,
  author       = {Venkatraman, Siddarth and Jain, Vineet and Mittal, Sarthak and Shah, Vedant and Obando-Ceron, Johan and Bengio, Yoshua and Bartoldson, Brian R. and Kailkhura, Bhavya and Lajoie, Guillaume and Berseth, Glen and Malkin, Nikolay and Jain, Moksh},
  title        = {Recursive Self-Aggregation Unlocks Deep Thinking in Large Language Models},
  howpublished = {arXiv:2509.26626},
  year         = {2025}
}

@misc{farmanfarmaian2026candidatefree,
  author       = {Farmanfarmaian, Guiv},
  title        = {Selection, Recombination, or a Fresh Solve? {A} Candidate-Free Control for Single-Pass Test-Time Aggregation},
  howpublished = {arXiv:2608.18379},
  year         = {2026},
  note         = {COLM 2026 Workshop on Efficient Reasoning}
}

@misc{fortuna2026disjunctive,
  author       = {Fortuna, Carolina and Bertalani{\v{c}}, Bla{\v{z}}},
  title        = {Multi-agent Scaling Across Disjunctive and Compensatory Tasks},
  howpublished = {arXiv:2609.31563},
  year         = {2026}
}

@misc{pappu2026sat,
  author       = {Pappu, Aneesh and Suzgun, Mirac and Kwon, Yongchan and Bianchi, Federico and El, Batu and Kochenderfer, Mykel J. and Cao, Hancheng and Zou, James},
  title        = {Self-Organizing Agent Teams Learn to Reason Together},
  howpublished = {arXiv:2609.22682},
  year         = {2026}
}

@misc{hu2026scouting,
  author       = {Hu, Julia and Shen, Alfred and Lakshmipathi, Kumar},
  title        = {Statistical Scouting Finds Debate-Safe but Not Debate-Useful Cases: A Matched-Ceiling Study of Open-Weight {LLM} Reasoning Protocols},
  howpublished = {arXiv:2605.09618},
  year         = {2026}
}

@misc{mirzaei2026samplemore,
  author       = {Mirzaei, Iliya},
  title        = {Sample More, Reflect Less: Self-Refine and Reflexion Lose to Repeated Sampling at Equal Token Cost, from 1.5{B} to 7{B}},
  howpublished = {arXiv:2607.28576},
  year         = {2026}
}

@misc{he2026minority,
  author       = {He, Chuan and Chen, Zebin and Yang, Zhengyi and Qiao, Shaobo and Ju, Mingchen and Liu, Jiate and Wen, Dong and Liu, Guanfeng},
  title        = {Minority Sentinel: When to Overturn Majority Voting in Multi-Agent {LLM} Debates},
  howpublished = {arXiv:2606.29270},
  year         = {2026},
  note         = {AgentSearch Workshop @ SIGIR 2026}
}

@inproceedings{okawa2026biased,
  author    = {Okawa, Maya},
  title     = {Emergence of Biased Consensus in Multi-Agent {LLM} Debates},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {arXiv:2608.02827}
}

@misc{motger2026madsurvey,
  author       = {Motger, Quim and Oriol, Marc and Marco, Jordi and Franch, Xavier},
  title        = {Multi-Agent Debate Strategies: Survey, Taxonomy, and Challenges},
  howpublished = {arXiv:2607.26212},
  year         = {2026}
}

% 演算子・初期集団・ES
@inproceedings{qiu2026es,
  author    = {Qiu, Xin and Gan, Yulu and Hayes, Conor F. and Liang, Qiyao and Xu, Yinggan and Dailey, Roberto and Meyerson, Elliot and Hodjat, Babak and Miikkulainen, Risto},
  title     = {Evolution Strategies at Scale: {LLM} Fine-Tuning Beyond Reinforcement Learning},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {arXiv:2509.24372}
}

@misc{gu2026hyperes,
  author       = {Gu, Yu and Zheng, Zhi and Ba, Yunpeng and Tong, Xialiang and Yuan, Mingxuan and Wang, Zhenkun},
  title        = {{Hyper-ES}: Effective Evolution Strategies for {LLM} Reasoning via Descent Direction Merging},
  howpublished = {arXiv:2608.05541},
  year         = {2026}
}

@inproceedings{zhuang2025coto,
  author    = {Zhuang, Zhan and others},
  title     = {Come Together, But Not Right Now: A Progressive Strategy to Boost Low-Rank Adaptation},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  note      = {arXiv:2506.05713}
}

@misc{salamanca2024seeded,
  author       = {Salamanca, Alejandro and {\"U}st{\"u}n, Ahmet and Detlefsen, Nicki Skafte and Dettmers, Tim},
  title        = {Seeded {LoRA}: Collaborative Fine-Tuning Through Seed Initialization of Adapters},
  howpublished = {ICML 2024 Workshop},
  year         = {2024},
  note         = {著者名表記・ワークショップ名は要確認}
}

@inproceedings{cao2026enmp,
  author    = {Cao, Anda and others},
  title     = {Evolutionary Negative Module Pruning for Better {LoRA} Merging},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026},
  note      = {arXiv:2604.17753}
}

@misc{wang2026copes,
  author       = {Wang, Zhiyuan and Liu, Shengcai and Wu, Jiahao and Lu, Ning and Ouyang, Hui and Zhang, Shaofeng and Lv, Haoze and Tang, Ke},
  title        = {Cooperative Coevolution for Resource-Constrained Agentic {LLM} Post-Training},
  howpublished = {arXiv:2608.02391},
  year         = {2026}
}

@misc{rodriquez2026snn,
  author       = {Rodriquez, Catherine and Ghawaly, James, Jr.},
  title        = {Co-Evolved Spiking Neural Network Ensembles via Marginal Contribution Fitness},
  howpublished = {arXiv:2606.13985},
  year         = {2026},
  note         = {ACM ICONS 2026}
}

% ペルソナ・能力保持
@misc{hu2026prism,
  author       = {Hu, Zizhao and Rostami, Mohammad and Thomason, Jesse},
  title        = {Expert Personas Improve {LLM} Alignment but Damage Accuracy: Bootstrapping Intent-Based Persona Routing with {PRISM}},
  howpublished = {arXiv:2603.18507},
  year         = {2026}
}

@misc{rodionov2026reasoningshift,
  author       = {Rodionov, Gleb and Garipov, Roman and Yakushev, George},
  title        = {Reasoning Shift: How Context Silently Shortens {LLM} Reasoning},
  howpublished = {arXiv:2604.01161},
  year         = {2026}
}

@misc{baines2026cartography,
  author       = {Baines, Luke and others},
  title        = {Persona Cartography: Charting Language Model Personality Traits in Weight Space},
  howpublished = {arXiv:2607.07916},
  year         = {2026}
}
```

---

## 付録: 未確認・要フォローのリスト

- **Heterogeneous Swarms の専門家が LoRA かフル重みか**: HTML 本文では「Gemma-7B を Tulu-v2 の10領域で微調整」とだけあり、PSO の更新対象（フル重みか LoRA か）は本文で明示されていない。**SC との計算量マッチ比較はない**（投票系は Prediction Merge のみ）ことは HTML 本文で確認済み。更新対象は PDF の付録で要確認（差分の主張に直結）。
- **2607.18255**: abs の Submitted 表示（2026-05-14）と arXiv ID（2607）が不整合。再確認が必要。
- **2603.06859 の旧題**（"Exact Is Easier…"）: 他論文の引用中の表記のみで、未確認。
- **MACA の OpenReview の扱い**（desk reject）: 二次情報のみ。
- **2510.01499（OW/ISP）と 2602.05395（ベイズ停止）の ICML 2026 採録**: papernotes の二次情報のみ。
- **MADA-RL・Seeded LoRA・CoTo・ENMP の著者名のフル表記**: bib 化する前に原典で確認する。
- **RSA の数学系ベンチでの T=1（1ステップ集約）と多数決の比較値**: 今回は SuperGPQA のみ抽出した。数学で「1ステップ集約（=我々の議論1ラウンド）が予算マッチの多数決に勝つか」は本文 Table を要確認。
- **PRISM の MMLU 数値**: 抽出結果に「68.0（長いペルソナ）」と「最小ペルソナ68.0 vs 長い66.3」の食い違いがある。原典の表で確認する。
