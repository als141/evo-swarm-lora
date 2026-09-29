# 文化的進化・社会学習・自己学習の限界・均質化の文献 — 第6章（考察）と第2章のために

**調査日**: 2026-09-30
**目的**: 第6章 6.2節「人間の社会との対応」、6.1節「何が精度を上げたか」、6.5節「限界」と、第2章 2.3節「議論と自己生成データからの学習」を補う。4つのテーマで15件に絞った。
**方法**: WebSearch で候補を探し、1件ずつ一次情報で書誌を確認した。確認先は arXiv の abs ページ（必要なら HTML 版の本文）、学会の proceedings ページ（NeurIPS / ICLR / ACL Anthology）、Crossref API、Europe PMC API である。数値は確認したページに書かれていたものだけを載せ、再計算はしていない。確認先の URL は末尾の「検証記録」にまとめた。
**検証の状態**: 15件すべて実在と書誌を確認した（UNVERIFIED なし）。次の2点だけは確認の経路が間接的である。
- vallinder2025cultural の頁（2771–2773）は Semantic Scholar の書誌（DBLP 由来）で確認した。IFAAMAS の PDF は取得が途中で止まり、本文は arXiv HTML 版で確認した。
- zhao2025sirius の NeurIPS 2025 採録は neurips.cc のポスターページで確認した。proceedings の頁番号は未確認である。

**査読前の論文が2件ある**（xu2026phantom、roe2026idempotent。いずれも2026年の arXiv）。本研究と同じ Qwen3 系・LoRA の設定で直接関係するので採用した。引用するときは「査読前」と分かるように書く。

---

## 0. 要点

1. **「選抜・伝達・変異」を世代で回す LLM の社会には直接の先例がある**。vallinder2025cultural では、最終資源の上位50%が生き残り、新しい個体が生き残りの戦略を受け継ぐ。ただし伝達はプロンプトの文章で行い、重みは変わらない。重みの学習で世代を重ねる例は、強化学習のエージェントにある（cook2024generational）。
2. **古典理論は「社会学習がいつ役立つか」を具体的に予測する**。
   - 社会学習は無差別であってはならず、いつ・誰から模倣するかの戦略が要る（laland2004social）。
   - 模倣が得になるのは、手本が自分の最良の行動を示し、情報が意図せずふるいにかけられるからである（rendell2010copy）。
   - 社会学習だけでは集団の適応度は上がらない（Rogers のパラドックス）。個体学習と組み合わせた「批判的社会学習」なら上がる（enquist2007critical）。
   - 本研究の子b は「自分が誤り、他者が正解した問題」だけを他者から学び、残りを自分の正解例で補う。これは条件付きの模倣と個体学習を組み合わせたものに当たる。
3. **自己生成の正解で学ぶ手法には、飽和と副作用の報告が多い**。
   - すでに強いモデルほど RFT の上積みは小さい（既存キー yuan2023scaling）。
   - 反復 SFT では pass@1 が上がっても、出力の多様性が一貫して下がった（wu2025reversal）。
   - RLVR で得た推論はベースモデルの範囲を出ない。教師からの蒸留だけが範囲を広げた（yue2025rlvr）。
   - Qwen3-8B の LoRA 自己学習では、平均精度が上がった同じ実行で、基準時に解けていた問題が多数壊れた（xu2026phantom、査読前）。
   - 3体が同じベースモデルを共有する本研究では、他者から学ぶ子b も「ベースモデルの範囲内での鋭敏化」にとどまると予想される。
4. **議論のデータから学ぶと精度は上がりうる**。役割ごとに成功した軌跡で学ぶ方式（zhao2025sirius）でも、議論を単一モデルに蒸留する方式（srivastava2025dte）でも上がった。ただし 3B 未満のモデルでは、2回目の反復で精度が落ちた（srivastava2025dte）。
5. **互いの出力で学ぶと多様性が失われる危険がある**。
   - 生成データの無差別な学習は、元の分布の裾を消す（shumailov2024collapse）。
   - 報酬による選別つきの反復再学習は、期待報酬を最大化する。同時に報酬モデルの偏りも増幅する（ferbach2024curated）。本研究では採点器が報酬に当たる。
   - 自分の出力での反復 SFT では、ペルソナの性質はほとんど増幅されず、減衰か維持にとどまった。対象には Qwen3-4B-Instruct が含まれる（roe2026idempotent、査読前）。
   - 「世代を重ねるとペルソナの個性が育つ」とは楽観できない。育たない場合の説明材料がそろっている。

---

## 1. テーマ別の一覧

「関係」列の記号は次のとおり。**支**: 本研究の設計・主張を支持する。**対**: 本研究との違いを示す対比に使う。**注**: 解釈や限界の注意点になる。

### テーマ1: LLM・学習エージェントの集団における文化的進化と社会学習

| キー | 文献 | 掲載・ID | 確認した内容 | 関係 |
|---|---|---|---|---|
| vallinder2025cultural | Vallinder, Hughes. Cultural Evolution of Cooperation among LLM Agents | AAMAS 2025（extended abstract）, pp. 2771–2773 / arXiv:2412.10270 | 12体が寄付ゲーム（間接互恵）を行い、最終資源の上位50%が次世代に残る。新しい6体は「前世代の上位50%の助言」を受け取り、それを改変して自分の戦略を作る。これを10世代繰り返す。協力の進化はモデルで大きく異なった（Claude 3.5 Sonnet > Gemini 1.5 Flash > GPT-4o）。同じモデルでも乱数 seed で結果が変わり、初期条件に敏感だった。 | 支・対・注 |
| cook2024generational | Cook, Lu, Hughes, Leibo, Foerster. Artificial Generational Intelligence: Cultural Accumulation in Reinforcement Learning | NeurIPS 2024 / arXiv:2406.00392 | 社会学習と独立の学習を釣り合わせた学習設定で、文化的蓄積が生じた。蓄積するエージェントは、同じ累積経験を1世代で学んだエージェントを上回った。世代には2つの定義がある。文脈内学習による「エピソード世代」と、重みの学習による「学習時世代」である。著者らは前者を知識の蓄積、後者を技能の蓄積に対応づけている。 | 支・対 |
| ren2024iterated | Ren, Guo, Qiu, Wang, Sutherland. Bias Amplification in Language Model Evolution: An Iterated Learning Perspective | NeurIPS 2024 / arXiv:2404.04286 | 人間の文化進化の研究で使われるベイズ的反復学習で、LLM の多段の自己改善と多エージェントの相互作用を説明した。反復学習は模倣・相互作用・伝達の3段階からなる。事前分布の小さな偏りは世代を経て増幅される。この増幅は、相互作用の段階で生成物を「選別」または「順位付け」することで制御できる。 | 支・注 |

### テーマ2: 古典理論（社会学習の戦略、Rogers のパラドックス、学習と進化）

| キー | 文献 | 掲載・ID | 確認した内容 | 関係 |
|---|---|---|---|---|
| laland2004social | Laland. Social learning strategies | Learning & Behavior 32(1):4–14, 2004 / DOI:10.3758/BF03196002 | 理論モデルからは、社会学習は無差別であってはならない。いつ模倣し、誰から学ぶかを決める戦略が要る。候補として「不確かなときに模倣」「多数派を模倣」「自分より良ければ模倣」などを挙げ、動物とヒトの証拠を整理した。 | 支（対応づけの語彙） |
| rendell2010copy | Rendell ほか計10名. Why Copy Others? Insights from the Social Learning Strategies Tournament | Science 328(5975):208–213, 2010 / DOI:10.1126/science.1184719 | 社会学習と個体学習の使い分けを競う計算機トーナメントを行った。社会学習に強く頼る戦略が際立って成功した。個体学習の情報が社会学習より高価でない場合でも、この傾向は変わらなかった。手本は自分の最も利得の高い行動を示すので、模倣者のために情報が意図せずふるいにかけられる。優勝戦略はほぼ社会学習だけに頼った。 | 支・注 |
| enquist2007critical | Enquist, Eriksson, Ghirlanda. Critical Social Learning: A Solution to Rogers's Paradox of Nonadaptive Culture | American Anthropologist 109(4):727–734, 2007 / DOI:10.1525/aa.2007.109.4.727 | Rogers (1988) のモデルでは、社会学習は集団の適応度を上げない。批判的社会学習は、まず社会学習を試し、得た行動が不十分なら個体学習に切り替える戦略である。これは純粋な社会学習より常に優れ、多くの場合は純粋な個体学習より適応度が高い。次の場合を除き ESS になる。(i) 文化の伝達が非常に不正確、(ii) 環境の変動が大きい、(iii) 社会学習が個体学習よりずっと高価。 | 支（設計の根拠）・理論的予測 |
| hinton1987learning | Hinton, Nowlan. How Learning Can Guide Evolution | Complex Systems 1(3):495–502, 1987 | 獲得形質が遺伝しなくても、学習は進化の探索空間の形を変え、共適応した対立遺伝子の組への道筋を作る（ボールドウィン効果）。学習する個体は、学習しない個体よりずっと速く進化した。 | 対（本手法はラマルク型） |

### テーマ3: 自己生成の正解での学習の効果と限界、議論のデータからの学習

| キー | 文献 | 掲載・ID | 確認した内容 | 関係 |
|---|---|---|---|---|
| wu2025reversal | Wu, Li, Liu. Progress or Regress? Self-Improvement Reversal in Post-training | ICLR 2025 / arXiv:2407.05013 | 対象は LLaMA-2-7B、Mistral-7B、LLaMA3-8B（全パラメータ学習）。比べた方法は3つ。正解ラベルで判定した自己生成の正解による反復 SFT、反復 DPO、両者の組合せである。どの方法でも pass@1 は上がった。一方、出力の多様性（構文・意味・論理）はどの方法でも反復とともに一貫して下がった。反復 SFT と SFT-DPO は分布外の汎化を大きく損なった（反復 DPO は改善した）。3つとも、易しい問題に偏る方向に難易度群の差を広げた。 | 注 |
| yue2025rlvr | Yue, Chen, Lu, Zhao, Wang, Yue, Song, Huang. Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? | NeurIPS 2025（Oral）/ arXiv:2504.13837 | RLVR で学習したモデルは、小さい k（k=1 など）の pass@k ではベースモデルを上回る。大きい k ではベースモデルが上回る。推論能力はベースモデルに由来し、その範囲に縛られる。6つの RLVR アルゴリズムの結果はほぼ同じだった。一方、教師からの蒸留は新しい推論の型を持ち込み、能力を実際に広げた。 | 注（解釈の枠組み） |
| xu2026phantom | Xu, Yan, Chen, Kechadi. Phantom Gains: Auditing Self-Improvement Against a Measured Null | arXiv:2608.20290（2026-08、査読前） | Qwen3-8B に rank-32 の LoRA で自己学習を3ラウンド行った（各256問。STaR、多数決 SFT、方策勾配の3種）。学習しない凍結モデルを同じ手順に通し、帰無を実測した。貪欲デコード1回で作った問題ごとの台帳は、学習していないモデルにも能力の変化を作り出した（主に推論のバッチ処理による）。難易度帯の1,163問のうち、基準時に解けていた問題を STaR は106問、多数決 SFT は88問壊した（設計を合わせた床は8問）。同じ実行で STaR の平均精度は5.6pt上がった。新たに pass@1 で解けた問題は、すべて基準時の pass@k で到達可能だった。基準時にほとんど到達しない問題を改善したのは、外部の教師（gpt-oss-120b）からの蒸留だけだった。 | 注（測定と解釈） |
| zhao2025sirius | Zhao, Yuksekgonul, Wu, Zou. SiriuS: Self-improving Multi-agent Systems via Bootstrapped Reasoning | NeurIPS 2025 / arXiv:2502.04780 | 多エージェント系で成功に至った推論の軌跡を「経験ライブラリ」に集め、各エージェントを自分の役割の成功軌跡で別々に SFT する。失敗した軌跡は修正してライブラリに加える。推論と生物医学の QA で2.86%〜21.88%向上した。骨格モデルは gpt-3.5-turbo と gpt-4o-mini（OpenAI の fine-tuning API）。序論では、多エージェント系の最適化の難しさとして、成功や失敗をどのエージェントに帰属させるか（貢献の割り当て）が曖昧な点を挙げている。 | 支・対 |
| srivastava2025dte | Srivastava, Bi, Lu, Wang. DEBATE, TRAIN, EVOLVE: Self-Evolution of Language Model Reasoning | EMNLP 2025, pp. 32764–32810 / arXiv:2505.15734 | 多エージェント議論の推論と合意の答えを使い、正解ラベルなしで単一モデルを GRPO で学習する。6モデル（Qwen2.5 の 1.5B〜14B、Llama の 3B と 8B）で、GSM-PLUS の精度が平均8.92%上がり、他の課題でも平均5.8%上がった。Qwen2.5-1.5B の上昇が最大だった（GSM-Plus で+13.92pt）。3B 未満のモデルは2回目の反復で精度が落ちた。サンプリング温度を0.7から0.3に下げると、失った性能の最大76%を回復した。 | 支・対・注 |

### テーマ4: 互いの出力で学ぶことによる均質化と多様性の喪失

| キー | 文献 | 掲載・ID | 確認した内容 | 関係 |
|---|---|---|---|---|
| shumailov2024collapse | Shumailov, Shumaylov, Zhao, Papernot, Anderson, Gal. AI models collapse when trained on recursively generated data | Nature 631(8022):755–759, 2024 / DOI:10.1038/s41586-024-07566-y | モデルが生成した内容を無差別に学習に使うと、元の分布の裾が消える不可逆の欠陥が生じる（モデル崩壊）。LLM、VAE、GMM で示した。同じ著者らの arXiv:2305.17493（The Curse of Recursion）が同じ現象を扱う。 | 注 |
| ferbach2024curated | Ferbach, Bertrand, Bose, Gidel. Self-Consuming Generative Models with Curated Data Provably Optimize Human Preferences | NeurIPS 2024 / arXiv:2407.09499 | 生成データを報酬モデルで選別して反復再学習すると、期待報酬が最大化されることを証明した。各段階で実データを一定の割合で混ぜると、再学習の反復は安定する。実験では、この手続きが報酬モデルの偏りを増幅した。 | 支・注 |
| roe2026idempotent | Roe, Sanderson, Nguyen, Huang, Nief, Shrivastava, Tan, Holtzman. Iterative Finetuning is Mostly Idempotent | arXiv:2605.01130（2026-05、査読前） | 性格や信念を種データで与えたモデルを、前の世代の出力で次々に微調整した。instruct モデルへの SFT では、性質はほぼ減衰か維持にとどまった。対象は Qwen3-4B-Instruct と Llama-3.3-70B-Instruct で、LoRA rank 16 を使った。増幅はまれで、起きると一貫性が落ちた。種データが十分なら性質は20周以上保たれたが、増幅はしなかった。DPO を継続して行うと増幅が起きた。毎周期に初期モデルから学習し直すと、増幅は消えた。 | 注・対 |

---

## 2. 本研究の設計・結果との関係

本研究の要素を次のように略記する。
- 子a: 自己学習。自分が正解した発話で学ぶ。
- 子b: 社会学習。自分が誤り他者が正解した問題で他者の発話を学び、上限1,000例に満たない分を子a の例で補う。
- 子c: 兄弟交叉（ΔW の平均）。
- 子はいずれも親の LoRA から継続して学習する。
- 系統は S（厳密 Shapley で選抜）、N（選抜なし・常に子b）、A1（単独精度で選抜）、RFT（ペルソナも議論も使わない単一モデルの自己学習）の4つ。これらを SC@k と比べる。

### テーマ1

- **vallinder2025cultural**
  - 支: 「選抜・伝達・変異」を世代で回す LLM の社会で、集団の性質（協力）が世代とともに改善しうることを示した。本研究の構図（チーム貢献による選抜と、他者の成功からの学習）に最も近い先例である（6.2節、2.3節）。
  - 対: 伝達は戦略の文章（プロンプト）で、重みは変わらない。選抜の基準は個体の利得で、チームへの貢献ではない。課題は社会的ジレンマで、正解のある推論課題ではない。
  - 注: 結果がモデルと seed に強く依存し、初期条件に敏感だった。本研究は各系統を1回しか実行していないので、6.5節「系統の反復がない」の根拠として引ける。
- **cook2024generational**
  - 支: 文化的蓄積の条件は、独立の学習と社会学習の釣り合いだった。子a と子b を併存させ、どちらを残すかを選抜で決める本研究の設計の根拠になる。この研究の「同じ累積経験を1世代で学ぶ」対照は、本研究の RFT（同じ問題で単一モデルが自己学習）と同じ役割を持つ。
  - 対: 強化学習のエージェントと単純な環境での結果で、LLM ではない。重みの学習で蓄積が起きる「学習時世代」（著者らは技能の蓄積に対応づけている）は、本研究の LoRA の世代に対応する。
- **ren2024iterated**
  - 支: 「正解に至った発話だけを学ぶ」は、反復学習の相互作用の段階での選別に当たり、増幅の向きを正答に向ける。
  - 注: 選別を通っても、ベースモデルの事前分布の偏り（3体に共通の解き方）は世代を経て増幅されうる。3体が同じベースモデルを共有する本研究では、ペルソナが互いに似ていく方向の圧力になる。ペルソナ間の不一致率の世代推移を測る根拠になる。

### テーマ2

- **laland2004social**
  - 支: 本研究の各要素を社会学習の戦略の語彙で位置づけられる（6.2節）。
    - 子b は「自分より良ければ模倣」に当たる。自分が失敗した問題に限るので「不確かなときに模倣」の要素も持つ。
    - 議論の round1 での同調は「多数派を模倣」に当たる。
    - パイロットでは、少数派が正解でも多数派に合わせた率が81〜95%だった。これは多数派の模倣が誤りを広げる例として説明できる（既存キー okawa2026biased と併用）。
- **rendell2010copy**
  - 支: 模倣が得になるのは、手本が良い行動を示し、情報がふるいにかけられるからである。子b は正解ラベルで確かめた他者の発話だけを学ぶので、このふるいを明示的に入れている。
  - 注: トーナメントの手本は、それぞれ独立に環境を探索した個体である。本研究の手本は同じベースモデルを共有し、持つ情報が大きく重なる。模倣で得られる新しい情報は少ない可能性がある（yue2025rlvr と併せて論じる）。
- **enquist2007critical**
  - 支（設計）: 子b は他者の例に自分の正解例を加えて作る。純粋な社会学習ではなく、個体学習と組み合わせた社会学習である。この理論は、組み合わせが純粋な社会学習より適応的であると予測する。子b の構成（第3章、第4章の設計変更の記録）の理論的な裏付けとして引ける。
  - 予測: 本研究の設定は、問題の分布が世代で変わらず（環境の変動が小さい）、社会学習の費用も個体学習と同程度である。したがって ESS の条件のうち残る懸念は「伝達の忠実さ」である。約1,000例の LoRA 学習で他者の解き方がどこまで忠実に移るかが、社会学習の効果を左右すると予想できる。
  - 注: この理論は選抜（チーム貢献による代表の選択）を扱わない。S と N の比較そのものの予測には使えない。
- **hinton1987learning**
  - 対: ボールドウィン効果では、学習の成果は遺伝しないが、学習が進化の探索を導く。本研究の子は親の LoRA を受け継いで学習を続けるラマルク型である。既存キー whitley1994lamarckian と併せて、6.2節で「本研究はラマルク型を選んだ」と位置づけられる。
  - 限界: ボールドウィン型の対照（毎世代ベースモデルから学び直し、選抜の結果だけを受け継ぐ）は実施していない。6.5節か第7章の今後の課題に書ける。roe2026idempotent で「継続学習か毎回の初期化か」が結果を分けたことも、この対照の意義を補強する。

### テーマ3

- **既存キー yuan2023scaling（再利用）**
  - 注: arXiv の要旨に「RFT はより性能の低い LLM ほど改善が大きい」とある。Qwen3-4B-Instruct-2507 は事後学習を十分に受けた強いモデルで、自己学習の上積みが小さいことの説明に使える。
  - 支: 同じ要旨に「複数のモデルの棄却サンプルを合わせると、LLaMA-7B の GSM8K が49.3%になり、SFT の35.9%を大きく上回った」とある。他者の正解から学ぶ子b の先例として使える。
- **wu2025reversal**
  - 注: 子a・子b と RFT は、正解ラベルで確かめた生成の正解による反復 SFT で、Wu らの反復 SFT と同じ型である。pass@1 が上がっても、多様性と未見の分布での汎化が落ちうる。本研究は test を新しい問題で評価し、ペルソナ間の不一致率の推移も測るので、この懸念を検証できる設計になっている。
  - 対: Wu らは 7〜8B の旧世代モデルを全パラメータで学習した。本研究は LoRA なので、変化の幅は小さいと予想される。
- **yue2025rlvr**
  - 注（解釈の枠組み）: 正解で選別した学習は、ベースモデルがすでに確率を持つ推論へ確率を寄せる鋭敏化にとどまりやすい。3体が同じベースモデルを共有するので、子b は外部の教師からの蒸留とは違い、新しい推論の型を持ち込みにくい。
  - 支（評価設計）: 進化的社会学習の利得は、同じモデルの多数決（SC@k）の曲線と比べて評価するのが妥当である。本研究が結果を SC@k 曲線上の位置で示すことの根拠になる（6.3節）。
- **xu2026phantom（査読前）**
  - 注（測定）: 本研究の「議論の正誤遷移」や「世代間で得た問題・失った問題」の分析は、雑音のある2つの推定の差である。学習しないモデルで同じ比較をした帰無を並べる必要がある。本研究でも vLLM の生成は同じ seed で再現しない（研究ログ）。Xu らは推論のバッチ処理だけで変化が作られることを示した。
  - 注（結果）: 平均精度が上がっても、基準時に解けていた問題が壊れている可能性がある。得た問題と失った問題を両方報告するとよい。
  - 設定が Qwen3・LoRA・自己学習と近いので説得力は高いが、査読前である。本文では補助的な引用にとどめる。
- **zhao2025sirius**
  - 支: 役割ごとに成功した軌跡で各エージェントを別々に SFT すると、多エージェント系の精度が上がった。本研究の子a（役割ごとの LoRA を自分の正解発話で学習）に近い（2.3節）。
  - 対: 世代交代と選抜を持たない。貢献の割り当てが曖昧だと序論で述べている。本研究の厳密 Shapley による選抜は、貢献の割り当てを連合の実測で行う点でこの課題に応える。骨格は API のモデルで、小さい公開モデルではない。
- **srivastava2025dte**
  - 支: 議論の記録から学習すると、1.5B〜14B の公開モデルでも精度が上がる（2.3節）。
  - 対: ペルソナを持たない単一モデルへの蒸留で、正解ラベルではなく合意を報酬にした GRPO である。本研究は正解で選別した SFT で、役割ごとに別々の LoRA を保つ。
  - 注: 3B 未満のモデルでは、2回目の反復で精度が落ちた。4B は境界より上だが、世代を重ねる学習では世代ごとの劣化を監視する必要がある。

### テーマ4

- **shumailov2024collapse**
  - 注: 生成データでの反復学習は分布の裾を消す。本研究には4つの緩和策がある。
    - 正解ラベルで選別する。
    - 世代ごとに新しい学習用の問題を使う（Q0〜Q5 は重複なし）。
    - ベースモデルは凍結し、LoRA だけを学習する。
    - 世代数は3である。
  - それでも、まれな解き方（ペルソナ固有の推論様式）が消える方向の圧力は残る。
- **ferbach2024curated**
  - 支: 正解だけを残す選別は報酬による選別であり、反復再学習は期待報酬（正答率）を最大化する向きに働く。この理論的な裏付けになる。
  - 注: 同じ理論から、選別に使う報酬の偏りが増幅される。本研究では採点器（答えの抽出と同値判定）が報酬に当たる。7月の研究で見つかった抽出の誤り（単語 ANSWER の 'A' を拾う）のような偏りがあれば、世代を重ねて増幅される。採点器を監査して固定したこと（第4章）の意義を、この理論で説明できる。多様性の保持は、この理論では保証されない。
- **roe2026idempotent（査読前）**
  - 注: 本研究と同じ Qwen3-4B-Instruct 系で、自分の出力での反復 SFT は、ペルソナの性質をほとんど増幅しなかった。本研究のパイロットでも、世代0のペルソナの行動差はサンプリングの揺らぎとほぼ同じだった。SFT だけの社会学習で個性（行動差）が世代とともに育つとは期待しにくい。育たなかった場合の説明として使える。
  - 対: Roe らは自分の出力だけで学ぶ。本研究の子b は他者の出力も学ぶので、性質はむしろ平均化（均質化）する方向に働くと予想される。継続学習と毎回の初期化で結果が変わった点は、本研究のラマルク型（親から継続）の設計に関係する。

---

## 3. 既存の refer.bib との関係（重複なし、再利用するキー）

- 今回の15件は、いずれも refer.bib に未登録である（キー・題名・arXiv ID で照合した）。
- 次の既存キーは、そのまま再利用する。
  - `boyd1985culture`: 二重継承（遺伝と文化）の基本文献。6.2節の骨格に使う。
  - `whitley1994lamarckian`: ラマルク型とボールドウィン効果の比較。hinton1987learning と併せて引く。
  - `zelikman2022star`, `yuan2023scaling`: 自己学習（STaR、RFT）。yuan2023scaling の要旨の2点（弱いモデルほど RFT の改善が大きい、複数モデルの棄却サンプルの結合が効く）を今回 arXiv で確認した。
  - `subramaniam2025multiagent`: モデルを別々のデータで学習させ、推論の多様性を保った。均質化への対策の先例として roe2026idempotent・ren2024iterated と併せて論じられる。
  - `maca2025preference`, `gkountouras2026canon`, `liu2026sdrl`, `pulici2026madarl`: 議論からの学習（2.3節）。zhao2025sirius・srivastava2025dte はこの段落に追加できる。
  - `okawa2026biased`: 同調が臨界を超えると集団の偏りに相転移する（literature_update.md に記載）。laland2004social の「多数派を模倣」と併せて引ける。

## 4. 考察章での使い方の案

- **6.2節（人間の社会との対応）の骨子**
  1. 二重の継承。学習した重みを子が受け継ぐ（ラマルク型。whitley1994lamarckian、hinton1987learning と対比）。チームへの貢献で代表を選ぶ（ダーウィン型）。両者を合わせると文化的進化の二重継承の構図になる（boyd1985culture）。
  2. 社会学習の戦略。子b は「自分より良ければ模倣」、議論での同調は「多数派を模倣」に当たる（laland2004social）。正解に限った模倣は、ふるいの効いた模倣である（rendell2010copy）。自分の例を混ぜる子b の構成は、批判的社会学習に近い（enquist2007critical）。
  3. LLM の社会での先例。プロンプトで伝達する例（vallinder2025cultural）、重みで世代を重ねる例（cook2024generational、RL）、反復学習による偏りの増幅（ren2024iterated）がある。
  4. 人間の社会との違い。本研究の3体は同じベースモデルを共有する。人間の集団のように独立に得た情報を持ち寄れないので、模倣で得る新しい情報が少ない（rendell2010copy の前提との違い、yue2025rlvr）。互いの出力で学ぶことは多様性を減らしうる（shumailov2024collapse、wu2025reversal、ren2024iterated）。ペルソナの性質は SFT では増幅されにくい（roe2026idempotent）。
- **6.1節（何が精度を上げたか）**: yuan2023scaling、yue2025rlvr、xu2026phantom、srivastava2025dte、zhao2025sirius を使い、結果の大小を「同じベースモデルの範囲内での鋭敏化」という枠組みで解釈する。
- **6.5節（限界）**: vallinder2025cultural（seed と初期条件への依存）、xu2026phantom（遷移の分析に実測の帰無が要る）、ferbach2024curated（採点器の偏りの増幅）、hinton1987learning（ボールドウィン型の対照がない）を使う。
- **第2章 2.3節**: zhao2025sirius と srivastava2025dte を、Multiagent Finetuning と MACA の後に1文ずつ追加できる。ラマルク的進化の段落には、hinton1987learning、cook2024generational、vallinder2025cultural を追加できる。

## 5. 本編から外した候補（上限15件に収めるため）

いずれも書誌は確認したが、BibTeX は付けていない。必要なら追加できる。
- Perez ほか. When LLMs Play the Telephone Game: Cultural Attractors as Conceptual Tools to Evaluate LLMs in Multi-turn Settings. ICLR 2025, arXiv:2407.04503。伝達の連鎖で文章の性質が「文化的アトラクタ」に引き寄せられる。ren2024iterated と論点が重なるので外した。確認先は arXiv abs と ICLR 2025 proceedings の索引。
- Song ほか. Mind the Gap: Examining the Self-Improvement Capabilities of Large Language Models. ICLR 2025, arXiv:2412.02674。自己検証による自己改善を生成と検証の差で定式化した。本研究は正解ラベルで選別するので関係が薄く、外した。確認先は arXiv abs と ICLR 2025 proceedings の索引。
- Guo ほか. The Curious Decline of Linguistic Diversity: Training Language Models on Synthetic Text. NAACL 2024 Findings, arXiv:2311.09807。前の世代の生成で再帰的に学習すると、語彙・構文・意味の多様性が一貫して下がる。wu2025reversal と論点が重なるので外した。掲載先は arXiv の注記で確認した。
- Chen ほか. MAGDi: Structured Distillation of Multi-Agent Interaction Graphs Improves Reasoning in Smaller Language Models. ICML 2024, arXiv:2402.01620。複数の LLM の議論を小さいモデルへ蒸留する。教師が異種の大きいモデルで、本研究の同じベースの仲間からの学習とは設定が遠いので外した。掲載先は arXiv の注記で確認した。

---

## 6. BibTeX（検証済みの15件）

refer.bib の書式（note に DOI と arXiv ID、note は英語のみ）に合わせた。refer.bib には追加していない。

```bibtex
% -------------------- 文化的進化・社会学習（2026-09-30 追加候補） --------------------

@inproceedings{vallinder2025cultural,
  author    = {Vallinder, Aron and Hughes, Edward},
  title     = {Cultural Evolution of Cooperation among {LLM} Agents},
  booktitle = {Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  pages     = {2771--2773},
  year      = {2025},
  note      = {Extended abstract. arXiv:2412.10270}
}

@inproceedings{cook2024generational,
  author    = {Cook, Jonathan and Lu, Chris and Hughes, Edward and Leibo, Joel Z. and Foerster, Jakob},
  title     = {Artificial Generational Intelligence: Cultural Accumulation in Reinforcement Learning},
  booktitle = {Advances in Neural Information Processing Systems 37 (NeurIPS)},
  year      = {2024},
  note      = {DOI: 10.52202/079017-1907, arXiv:2406.00392}
}

@inproceedings{ren2024iterated,
  author    = {Ren, Yi and Guo, Shangmin and Qiu, Linlu and Wang, Bailin and Sutherland, Danica J.},
  title     = {Bias Amplification in Language Model Evolution: An Iterated Learning Perspective},
  booktitle = {Advances in Neural Information Processing Systems 37 (NeurIPS)},
  year      = {2024},
  note      = {DOI: 10.52202/079017-1220, arXiv:2404.04286}
}

@article{laland2004social,
  author  = {Laland, Kevin N.},
  title   = {Social Learning Strategies},
  journal = {Learning \& Behavior},
  volume  = {32},
  number  = {1},
  pages   = {4--14},
  year    = {2004},
  note    = {DOI: 10.3758/BF03196002}
}

@article{rendell2010copy,
  author  = {Rendell, L. and Boyd, R. and Cownden, D. and Enquist, M. and Eriksson, K. and Feldman, M. W. and Fogarty, L. and Ghirlanda, S. and Lillicrap, T. and Laland, K. N.},
  title   = {Why Copy Others? {Insights} from the Social Learning Strategies Tournament},
  journal = {Science},
  volume  = {328},
  number  = {5975},
  pages   = {208--213},
  year    = {2010},
  note    = {DOI: 10.1126/science.1184719}
}

@article{enquist2007critical,
  author  = {Enquist, Magnus and Eriksson, Kimmo and Ghirlanda, Stefano},
  title   = {Critical Social Learning: A Solution to {Rogers's} Paradox of Nonadaptive Culture},
  journal = {American Anthropologist},
  volume  = {109},
  number  = {4},
  pages   = {727--734},
  year    = {2007},
  note    = {DOI: 10.1525/aa.2007.109.4.727}
}

@article{hinton1987learning,
  author  = {Hinton, Geoffrey E. and Nowlan, Steven J.},
  title   = {How Learning Can Guide Evolution},
  journal = {Complex Systems},
  volume  = {1},
  number  = {3},
  pages   = {495--502},
  year    = {1987}
}

@inproceedings{wu2025reversal,
  author    = {Wu, Ting and Li, Xuefeng and Liu, Pengfei},
  title     = {Progress or Regress? {Self-Improvement} Reversal in Post-training},
  booktitle = {Proceedings of the 13th International Conference on Learning Representations (ICLR)},
  year      = {2025},
  note      = {arXiv:2407.05013}
}

@inproceedings{yue2025rlvr,
  author    = {Yue, Yang and Chen, Zhiqi and Lu, Rui and Zhao, Andrew and Wang, Zhaokai and Yue, Yang and Song, Shiji and Huang, Gao},
  title     = {Does Reinforcement Learning Really Incentivize Reasoning Capacity in {LLMs} Beyond the Base Model?},
  booktitle = {Advances in Neural Information Processing Systems 38 (NeurIPS)},
  year      = {2025},
  note      = {arXiv:2504.13837}
}

@misc{xu2026phantom,
  author       = {Xu, Cheng and Yan, Nan and Chen, Liming and Kechadi, M-Tahar},
  title        = {Phantom Gains: Auditing Self-Improvement Against a Measured Null},
  howpublished = {arXiv:2608.20290},
  year         = {2026}
}

@inproceedings{zhao2025sirius,
  author    = {Zhao, Wanjia and Yuksekgonul, Mert and Wu, Shirley and Zou, James},
  title     = {{SiriuS}: Self-improving Multi-agent Systems via Bootstrapped Reasoning},
  booktitle = {Advances in Neural Information Processing Systems 38 (NeurIPS)},
  year      = {2025},
  note      = {arXiv:2502.04780}
}

@inproceedings{srivastava2025dte,
  author    = {Srivastava, Gaurav and Bi, Zhenyu and Lu, Meng and Wang, Xuan},
  title     = {{DEBATE, TRAIN, EVOLVE}: Self-Evolution of Language Model Reasoning},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages     = {32764--32810},
  year      = {2025},
  note      = {DOI: 10.18653/v1/2025.emnlp-main.1666, arXiv:2505.15734}
}

@article{shumailov2024collapse,
  author  = {Shumailov, Ilia and Shumaylov, Zakhar and Zhao, Yiren and Papernot, Nicolas and Anderson, Ross and Gal, Yarin},
  title   = {{AI} Models Collapse When Trained on Recursively Generated Data},
  journal = {Nature},
  volume  = {631},
  number  = {8022},
  pages   = {755--759},
  year    = {2024},
  note    = {DOI: 10.1038/s41586-024-07566-y}
}

@inproceedings{ferbach2024curated,
  author    = {Ferbach, Damien and Bertrand, Quentin and Bose, Avishek Joey and Gidel, Gauthier},
  title     = {Self-Consuming Generative Models with Curated Data Provably Optimize Human Preferences},
  booktitle = {Advances in Neural Information Processing Systems 37 (NeurIPS)},
  year      = {2024},
  note      = {DOI: 10.52202/079017-3256, arXiv:2407.09499}
}

@misc{roe2026idempotent,
  author       = {Roe, Zephaniah and Sanderson, Jack and Nguyen, Dang and Huang, Julian and Nief, Todd and Shrivastava, Aryan and Tan, Chenhao and Holtzman, Ari},
  title        = {Iterative Finetuning is Mostly Idempotent},
  howpublished = {arXiv:2605.01130},
  year         = {2026}
}
```

補足:
- rendell2010copy の著者名は、Crossref の登録どおりイニシャルで書いた。
- yue2025rlvr の著者には「Yang Yue」が2人いる（NeurIPS の proceedings ページで確認）。誤記ではない。

---

## 7. 検証記録（確認した一次情報）

| キー | 確認先 |
|---|---|
| vallinder2025cultural | [arXiv abs](https://arxiv.org/abs/2412.10270)、[arXiv HTML（手続き: 12体・上位50%・10世代・5 seed）](https://arxiv.org/html/2412.10270v1)、Semantic Scholar API（DBLP conf/ifaamas/Vallinder025、pp. 2771–2773）、[IFAAMAS PDF](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2771.pdf)（取得が途中で停止） |
| cook2024generational | [arXiv abs](https://arxiv.org/abs/2406.00392)、[NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6df3a719d99bd2479c04114d357003d0-Abstract-Conference.html) |
| ren2024iterated | [arXiv abs](https://arxiv.org/abs/2404.04286)、[arXiv HTML（3段階と選別の記述）](https://arxiv.org/html/2404.04286v2)、[NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4418f6a54f4314202688d77956e731ce-Abstract-Conference.html) |
| laland2004social | [Crossref](https://api.crossref.org/works/10.3758/BF03196002)、Europe PMC（PMID 15161136、要旨） |
| rendell2010copy | [Crossref](https://api.crossref.org/works/10.1126/science.1184719)、Europe PMC（PMID 20378813、要旨） |
| enquist2007critical | [Crossref（要旨を含む）](https://api.crossref.org/works/10.1525/aa.2007.109.4.727) |
| hinton1987learning | [著者のページ（要旨・書誌）](https://www.cs.toronto.edu/~hinton/absps/evolution.htm)、[Complex Systems 誌のページ（1巻3号）](https://www.complex-systems.com/abstracts/v01_i03_a06/) |
| wu2025reversal | [arXiv abs](https://arxiv.org/abs/2407.05013)、[arXiv HTML（モデル・多様性・OOD の記述）](https://arxiv.org/html/2407.05013v1)、[ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1fa0c4e5a7e189729230d018b229abc7-Abstract-Conference.html) |
| yue2025rlvr | [arXiv abs](https://arxiv.org/abs/2504.13837)、[NeurIPS 2025 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/537d5aa768c2d534016a4d06f87bc8fb-Abstract-Conference.html) |
| xu2026phantom | [arXiv abs](https://arxiv.org/abs/2608.20290)、[arXiv HTML（106/88問、5.6pt、帰無の記述）](https://arxiv.org/html/2608.20290v1) |
| zhao2025sirius | [arXiv abs](https://arxiv.org/abs/2502.04780)、[arXiv HTML（骨格モデル・役割ごとの SFT）](https://arxiv.org/html/2502.04780v1)、[NeurIPS 2025 ポスター](https://neurips.cc/virtual/2025/poster/118834) |
| srivastava2025dte | [arXiv abs](https://arxiv.org/abs/2505.15734)、[arXiv HTML（6モデル、+13.92pt、3B 未満の劣化）](https://arxiv.org/html/2505.15734v2)、[ACL Anthology](https://aclanthology.org/2025.emnlp-main.1666/) |
| shumailov2024collapse | [Crossref](https://api.crossref.org/works/10.1038/s41586-024-07566-y)、Europe PMC（PMID 39048682、要旨）、[関連 preprint](https://arxiv.org/abs/2305.17493) |
| ferbach2024curated | [arXiv abs](https://arxiv.org/abs/2407.09499)、[NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b9e88ae0308cf82d0b0f634ddbdf809a-Abstract-Conference.html) |
| roe2026idempotent | [arXiv abs](https://arxiv.org/abs/2605.01130)、[arXiv HTML（Qwen3-4B-Instruct、LoRA rank 16、20周）](https://arxiv.org/html/2605.01130v1) |
| yuan2023scaling（既存） | [arXiv abs（要旨の2点）](https://arxiv.org/abs/2308.01825) |
