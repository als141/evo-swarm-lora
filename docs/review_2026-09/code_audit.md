# コード監査報告（評価・学習・進化パイプライン）— 2026-09-30

対象: `src/evalx/`、`src/evolve/loop.py`、`src/models/lora_ops.py`、`scripts/run_eval*.py`、`scripts/run_evolution.py`、学習スクリプト、`cloud/`・`scripts/cloud/` のジョブ定義。
目的: 次の再実験（新GCPプロジェクト、進化v4・チーム選抜実験）の前に、結果を歪めうるバグと設計上の問題を洗い出す。
方法: コード精読と、手元の生データ（`results/gcs/` の per_item・`llm_calls` 完全ログ・transcripts、`artifacts_local/` のアダプタ）を使った定量確認。GPU・クラウドは使っていない。
テスト: `uv run pytest -q` → **58 passed**（ただし §9 の穴があり、以下の問題はどれもテストで検出されない）。

重大度の定義: **致命**＝その実験の結論が成り立たない／**重大**＝数値や比較の解釈が変わる、または次の実験の設計前提が崩れる／**中**＝特定条件で歪む・再現性を損なう／**軽微**＝実害は小さいが直すべき。

---

## 0. 要点（重大度順）

| # | 重大度 | 問題 | 定量的根拠 | 対処 |
|---|---|---|---|---|
| E1 | 致命（進化実験） | 初期集団が「SFT個体＋2%ノイズの複製」で、遺伝的多様性がない | 変異1回の相対変化 2.3%、6世代で 2.8%。独立学習LoRA同士は cos=0.01・相対距離 140%（約50倍） | 独立学習した多様な初期集団を作る（§4.1） |
| E2 | 致命（進化実験） | 選抜がノイズに支配されている（共通乱数なし、再測定SD 3.1pt、最大差10pt） | 同一連合の世代間再測定 SD=3.07pt、候補間 Shapley 差は 0.002〜0.034 | round0共有による共通乱数化と大きな適応度セット（§2.2・§4.2） |
| S1 | 重大 | vLLM の seed 指定で出力が再現しない | 同一 model+messages+seed の重複呼び出し 909件中、一致は **29件（3.2%）** | seed ではなく生成テキストのキャッシュ共有で共通乱数を作る（§2.1） |
| S2 | 重大 | 議論 round1 で**自分の前回回答を見せていない** | 2-1割れで多数派側の変更率が **28〜45%**（c7: 44.8%）。条件付き更新の指示が成立していない | 自発話を提示する新スタイルを追加（§3.1） |
| G1 | 重大 | MATH-500 採点の偽陰性（正解側の `\sqrt2` `\frac59` `^\circ` 単位 `x=` 基数添字 pmatrix 等が正規化されない） | 全予測の **6.49%（623/9600）** が偽陰性。全条件で精度が約 **6pt 過小**。c7 対 SC@9 は −0.78pt → **+0.01pt [−1.07,+1.17]**（同等性成立に変わる） | 正解側の正規化と math-verify 系の同値判定（§1.1） |
| S3 | 重大 | G2集約比較（多数決/重み付き/GenSelect）が条件ごとに round0 を生成し直しており、集約差とサンプリングノイズが交絡 | 同一 round0 のはずの 900 呼び出しで応答がほぼ全て異なる | 同一トランスクリプトにオフラインで各集約を適用（§2.1） |
| R1 | 重大（再現性） | 評価ジョブが `eval:latest` の可変タグを参照。ダイジェストも結果に記録されない | +6pt の系統差はまさにイメージ再ビルドで発生 | `@sha256` 固定＋結果JSONへ記録（§7.1） |
| V1 | 重大（v4で顕在化） | 進化ジョブの vLLM `max-model-len` 既定が 8192（battery だけ 32768） | 8192トークン生成＋他者発話の埋め込みで溢れ、400→リトライ→全体停止になる | 進化ジョブにも `VLLM_MAX_MODEL_LEN=32768`（§7.4） |
| T1 | 中 | SFT が全トークン（system+user+assistant）に損失を掛けている | TRL 0.24 で `text` 列を渡すとマスクなし | prompt-completion 形式で assistant のみ学習（§5.1） |
| T2 | 中 | 学習 seed が子プロセスに伝わらない（記録は1234、実効は42）。LoRA 初期化は非シード | `train_all_personas_vertex.py` が seed を渡していない | seed を明示的に中継（§5.2、v4の多様性設計で必須） |
| X1 | 中 | 8192トークン打ち切りの非対称性（ANSWER 行なしがベース2.2〜3.7% vs LoRA 0.5〜1.5%） | c1 ベース solo の None 率は MATH 3.3%・SGPQA 1.7%、チームは約0% | `finish_reason` を記録し、16384 を検討（§1.2） |
| C4 | 中（v4で顕在化） | ΔW ブレンド交叉は、異なる親同士だと rank 切り詰めでエネルギーの 11〜13% を失う。直交性による縮みと合わせ、子の更新量は親の約0.66倍になる | float64 で実測 | ランク2rの子、ノルム補正、TIES/DARE（§4.4） |
| O1 | 軽微 | タイブレークが問題に依存せず固定インデックス（seed1〜4・777では常に辞書順最小を選ぶ） | `Random(seed).choice` を毎問同じ seed で呼んでいる | 問題IDを混ぜた seed を使う（§2.3） |
| O2 | 中（G2の結論） | GenSelect の候補順が全問で同一（seed9 では [0,2,1]）。裁定の14%が解析不能（`BEST: J` と選択肢を答える） | g2 の judge 50呼び出しを確認 | 問題ごとのシャッフル、選択肢形式への耐性（§2.4） |

**確認できた健全な点**
- SC@9 をログの9サンプルから再集計すると、保存済み予測と **6000/6000件（100%）一致**した。記録と多数決の実装に誤りはない。
- 選択肢の抽出は 97.3% が素の1文字で、誤抽出は約0.4%にとどまる。
- 数値比較のトークン計測も、呼び出し数（c7=6.0/問、SC=9.0/問）が理論値と一致する。

---

## 1. 採点

### 1.1 MATH-500 の偽陰性 — **重大**
- 箇所: `src/evalx/tasks.py:69-91`（`normalize_math_answer`）、`:261-276`（`is_correct`）
- 原因: 正解（raw LaTeX）側の次の表記を正規化できず、モデルが数学的に同値な答えを書いても文字列が一致しない。
  - 波括弧のない `\sqrt2`、`\frac59`、`\frac9{19}`
  - 角度の `^\circ`
  - 単位の `\text{ cents}`、`\mbox{ inches}^2`
  - `x=5`、`x \in [-2,7]`
  - 基数添字 `2516_8`
  - `\begin{pmatrix}`
  - 小数と分数（`5.5` と `\frac{11}{2}`）
  - `11(1+\sqrt5)` と `11\sqrt5+11`
  - `∪`、`π`、`√` の記号
- 再現: `uv run --no-project --with sympy --with numpy python scripts/analysis/audit_math_grading.py --impact --dump-pairs out.tsv`
  - Hendrycks MATH 式の前処理、sympy による数値同値判定、区間とタプルの要素比較で再採点した。
  - 反転した40種の（正解, 予測）対はすべて目視で同値と確認した。
  - うち8種（49件）は「分数を小数で答えた／約分していない」という形式違いで、値は正しい。
- 定量:
  - 全9600予測中 **623件（6.49%）** が偽陰性だった（形式違いを除いても 5.9%）。
  - 条件別の過小評価は +5.2〜+8.0pt。新環境では c1 +6.0 / SC@9 +6.0 / c5 +6.0 / c7 +6.5pt。
  - SC@9 の多数決で「同値の別表記による票割れ」は起きていなかった（`audit_math_vote_splitting.py`、改良多数決との差 +0.00pt）。
- 主要比較への影響（問題IDクラスタ、[95%CI]）:

| 比較（MATH-500） | 現行採点 | 改良採点 |
|---|---|---|
| c7 vs SC@9（6シード） | −0.78 [−2.38,+0.69] p=0.34 | **+0.01 [−1.07,+1.17]** p=1.0（95%CI ⊂ ±2pt → 同等性成立） |
| c7 vs ベース（3シード） | +3.35 [+0.90,+5.86] p=0.009 | +4.63 [+2.06,+7.34] p=0.0006 |
| c7 vs 旧チームc5 | +6.05 | +7.14 |
| 実験1: 進化の寄与 c5 vs c4 | +4.31 p=0.0004 | +3.73 p=0.003 |
| 実験1: 素の議論 c3 vs c1 | +4.18 | +5.02 |
| 実験1: gen0 c4 vs ベース c1 | −5.53 | −3.73 |

  符号が反転する比較はない。ただし、修論の「MATH-500 で SC@9 と同等性未確定」は「同等性成立」に変わる。3ベンチ統合の c7 vs SC@9（−3.49pt）への影響は約+0.04ptで、無視できる。MATH-500 の絶対値（ベース0.818、SC@9 0.873 等）は約6pt 低すぎる。
- 修正案:
  1. 正解側にも予測側と同じ前処理（Hendrycks `strip_string` 系：`\sqrt`・`\frac` の短縮形、`^\circ`、末尾単位、`x=` と `x\in` の接頭、基数添字、pmatrix→タプル）をかける。
  2. 同値判定は `math-verify`（HuggingFace、`uv run --with math-verify` で動作確認済み）または sympy の数値判定を併用する。
  3. 票の集計キーも同じ正規化にそろえる。
  4. 修論では、現行採点と改良採点を併記するか、改良採点で全面的に差し替える。

### 1.2 選択肢の抽出と 8192 トークン打ち切り — 中
- 箇所: `tasks.py:229-238`（`re.search(r"[A-J]", tail.upper())` が ANSWER 行の**最初の** A〜J 文字を拾う）、`tasks.py:206-216`
- 定量（run002 の選択肢タスク 91,502 呼び出し）:
  - ANSWER 行が素の1文字だったのは 97.3%。
  - 非標準の末尾は 361件（0.4%）。代表例は `None of the above` など約240件で、`NONE` の `E` を拾って **E 票**になる。ほかに `<letter>`→E、`the number of leaf…`→H、`A,B,C,D`（複数回答）→A。
  - ANSWER 行がない応答は 2,068件（2.3%）で、中央値は 19,672 字。**8192トークン上限に達した暴走・反復**が主因で、うち1,405件はフォールバックで途中の「Option D…」を拾っている。
  - **非対称性**: ANSWER 行なしの率はベースで 2.2〜3.7%（recheck 2.2%、remeasure 3.6%、robust 3.7%）、LoRA 個体で 0.5〜1.5%。ペルソナSFTで出力が短くなったため。
  - その結果、per_item の None 率は c1 ベース solo が MATH 3.3%・SGPQA 1.7%・MMLU 0.4%、チームと SC は約0% になり、**ベース単体だけが打ち切りで不利**になる。
  - 「c7 ≈ ベース（同等性成立）」は、ベースが打ち切りで 0.5〜1pt 程度不利な条件での判定である。
- 修正案:
  - ANSWER 行の末尾は `^\**\(?([A-J])\)?\**\.?$` のような独立トークンだけを採用し、`none` や複数回答は None にする。
  - `finish_reason` と `usage` を記録し（§6.1）、打ち切りを明示的に数える。
  - Qwen3-4B-Instruct-2507 の推奨出力長 16,384 を採用するか、少なくとも打ち切り率を条件別に報告する。

---

## 2. シード設計と共通乱数

### 2.1 vLLM の seed は出力を再現しない — **重大**
- 箇所: `client.py:112`（`seed=config.seed`）。この設計は「同じ seed なら同じ出力」を前提にしている。
- 定量:
  - run002 の llm_calls 全112,841件から、model・messages・seed・温度・max_tokens がすべて同一の重複呼び出し **909件** を抽出した。応答が完全一致したのは **29件（3.2%）** だった。
  - バッチ構成に依存する数値差が、長いCoTの途中で分岐を生むためと考えられる。旧実験の「同一設定で12〜14%の予測が変わる」とも整合する。
- 帰結:
  1. **G2 集約 ablation は交絡している。** 3条件（majority/weighted/genselect）は round0 のプロンプトと seed が同一なのに、応答は別物だった（重複 909件の大半がこれ）。「weighted 0.440 vs majority 0.433」はサンプリングノイズと区別できない（150問、SE ≈ 4pt）。
  2. 同じ理由で、seed による**共通乱数（CRN）は成立しない**。進化の候補比較、アブレーション、プロトコル A/B は、どれも独立サンプルどうしの比較になっていた。
- 修正案:
  - CRN は**生成テキストのキャッシュ共有**で実現する（§2.2）。集約方式の比較は、同一トランスクリプトにオフラインで各集約を適用して行う（GenSelect のみ追加の judge 呼び出しが要る）。
  - ビット単位の再現が必要なら、vLLM のバッチ不変モード（新しい版にある `VLLM_BATCH_INVARIANT` 系）を検証してから使う（性能コストあり、要確認）。

### 2.2 seed がチーム内の「位置」に依存し、round0 が共有されない — **重大（v4の前提）**
- 箇所: `debate.py:232-237`（`seed*10000 + agent_idx*100 + round_idx`）、`run_eval.py:302-318`（coalitions モードが連合ごとに round0 を生成し直す）、`loop.py:90-113`（`CoalitionEvaluator` のキャッシュは連合単位のみ）
- 事実:
  - 同じエージェントの round0 でも、solo（seed=s）、ペア内の位置0（s·10⁴）、3体内の位置1（s·10⁴+100）で別の生成になる。
  - `solo_answer` と round0 はメッセージが完全に同一なのに、共有されていない。
- 無駄と帰結:
  - 3体の Shapley 評価（7連合）は round0 を12回生成するが、固有のエージェントは3体しかない。
  - 進化の1世代では、代表7連合＋子の固有連合で 66 呼び出し/問 → round0 共有なら **36 呼び出し/問（45%削減）** になる。
  - しかも全連合が同じ round0 サンプルを共有するので、Shapley の各限界貢献に共通乱数が効く。
- 修正案（具体）:
  1. `R0Cache`（キー: `(agent_id, item_id, sample_idx)` → 生成テキストと抽出回答）を導入する。`solo_answer` と `run_debate` の round0 はまずキャッシュを引き、無ければ生成して保存する。
  2. seed は `hash(agent_id, item_id, round, sample_idx)` 由来にし、位置 `agent_idx` を使わない。温度サンプリング対照（同一モデル3体）では `agent_id` を `sampler_0..2` と別名にして多様性を保つ。
  3. round1 のキャッシュキーは `(agent_id, item_id, frozenset(相手のround0 id))`。これで「全チームの landscape」を round0 共有＋round1 の文脈別生成で最小コストで測れる。
  4. キャッシュは JSONL（GCS 上）に追記する。ProgressCache と同じ方式で、プリエンプト耐性がある。

### 2.3 タイブレークが問題に依存しない — 軽微
- 箇所: `debate.py:308-317`（`random.Random(tie_break_seed).choice(winners)`）。`run_eval.py:283,292` と `loop.py`（`CoalitionEvaluator(... args.fitness_seed)`）では、全問に同じ seed が渡される。
- 事実:
  - seed 1〜4・555・777 では、2者・3者の同数は**常に辞書順最小**が選ばれる。seed 5 は最大、seed 6 は2者なら最小・3者なら最大。
  - MMLU-Pro の正解文字は A〜D がやや多い（A 11.7%、J 7.2%）。そのため「最小文字を選ぶ」規則は一様乱択より僅かに有利になる。
  - 同数の頻度は低く（3体では1-1-1が約8%で、多くは weighted で解消）、実害は小さい。
  - ただし 2体連合（Shapley の中間連合）では同数が頻発し、評価値に構造的な偏りを入れる。
- 修正案: `Random(hash((seed, item_id)))` のように問題ごとの seed を使う。`test_tie_is_deterministic` は「問題間で独立」を検査する形に変える。

### 2.4 GenSelect の候補順が全問で固定・解析失敗が多い — 中（G2の結論に影響）
- 箇所: `debate.py:160-161, 295`（`shuffle_seed=tie_break_seed`）、`:171-179`
- 事実:
  - seed 9 の G2 では、全問で候補順が [critic, explorer, pragmatist] に固定されていた。位置バイアスとエージェントの質が交絡する。
  - judge の 50 呼び出しのうち 7件（14%）が `BEST: J` のように選択肢の文字を返した。数字が無いため None になり、weighted にフォールバックしていた。
- 修正案: 問題ごとのシャッフル。候補番号を `[1]〜[3]` の独自記号にし、選択肢の文字と混同しないプロンプトにする。解析失敗率を記録する。

---

## 3. 議論プロトコル

### 3.1 round1 で自分の前回回答を提示していない — **重大**
- 箇所: `debate.py:255`（`others = {… if name != agent.name}`）、`:259-267`、`:94-108`
  - standard の指示は "You may keep or change your previous answer"、conditional の指示は "compare it with your own… keep your original answer" だが、どちらも**前回回答は入力にない**。
  - Du et al. (2023) の原型では、自分の前回応答が会話履歴に残る。
- 定量（`scripts/analysis/audit_debate_transitions.py`）:

| データ | 2-1割れの**多数派**の変更率 | 少数派の変更率 | 全員一致の変更率 | round0多数決→最終 |
|---|---|---|---|---|
| c7 seed1（conditional＋匿名化、1000問） | **44.8%**（正→誤62・誤→正63） | 98.1% | 0.7% | 0.656→0.677 |
| c5 demo（standard、MMLU-Pro 60問） | 34.4% | 80.0% | 1.7% | 0.767→0.750 |
| MATH 診断3条件（standard、120問） | 28.3% | 100% | 0% | 0.900→0.908 |

  多数派のエージェントには「賛成1・反対1」しか見えず、自分が多数派だという情報が消えて解き直している。
  - 条件付き更新（「自分の誤りを特定できたときだけ変更」）は、元の回答が見えないので原理的に機能しない。修論 §5.2 の「条件付き更新＝追従対策」という解釈は、実装と合っていない。
  - 議論の実効果は「少数派が多数派に寄る」ことにほぼ尽きている（c7 では少数派の誤→正が92件、正→誤が51件）。
- 修正案:
  - 新スタイル `du2023` を追加する。自分の前回応答を assistant ターンとして履歴に残すか、"Your previous solution: …" として明示する。standard は v3 再現用に文言固定のまま残す（テスト `test_standard_prompt_unchanged` と整合）。
  - v4 では新スタイルを既定にし、round0 の多数決・standard・du2023 を同じ round0 キャッシュ上で比較する（round1 だけ再生成）。

### 3.2 その他（軽微）
- `anonymize=False` のとき `--- Agent critic ---` のように役割名がそのまま出る（c5 など旧プロトコル）。
- 匿名化のシャッフル seed も問題に依存しない（`debate.py:256-258`）。ある視点から見た他者の提示順は全問で同じになる。
- weighted 投票は 2-1 をほぼ覆さないため、同数解消器としてしか効かない。c7 seed1 では round1 の多数決 0.679 に対し weighted は 0.677 で、差は事実上ゼロ（G2 の +0.7pt は §2.1 のノイズ内）。
- `tail_confidence` は末尾64トークンの幾何平均で、`ANSWER:` などの定型トークンが支配する。答えのトークン自体の確率ではない。
- vLLM の版とエンジンによって、logprob が温度・top-k/p の適用前（raw）か適用後（processed）かが異なる。新しい vLLM では明示指定できるので、確信度を使うなら版を固定し、mode を記録すること（要確認）。

---

## 4. 進化（run001、`results/gcs/run001/evolution`・`artifacts_local`）

### 4.1 初期集団に遺伝的多様性がない — **致命**
- 箇所: `run_evolution.py:101-113`（`make_child(original, original, …, mut_ratio=1.0, mut_std=0.02)`＝自分自身とのブレンド＋2%変異）
- 定量（float64、`scripts/analysis/audit_evolution_operators.py` → `results/analysis_evolution_operators.json`）:

| 比較 | cos | 相対変化 ‖ΔW₁−ΔW₀‖/‖ΔW₀‖ |
|---|---|---|
| 変異1回（persona_b→gen0変異体） | 0.99973 | 2.31% |
| 交叉＋変異（persona_b→gen1子） | 0.99979 | 2.05% |
| 6世代（persona_b→最終gen5子） | 0.99962 | **2.77%** |
| 別ペルソナ（gen0 persona_a vs b） | **0.0118** | **140.2%** |
| 別ペルソナ（run002 persona_a vs b） | 0.0070 | 140.9% |

  独立に学習した LoRA 同士は ΔW 空間でほぼ直交している（√2≈141%）。進化の全摂動は個体間の距離の約1/50しかない。同じ親の近縁同士を交叉しても子は親と変わらず、選抜の対象になる「差」が存在しなかった。
- 補足: 既存の `delta_w_similarity.py` は float32 の素朴な総和で cos を計算しており、**cos>1（最大1.0005）**が出ていた。精度は約1e-4 で、報告値「0.9997」と「1.0000」の差は桁落ちと同じ桁である。結論（ほぼ同一）は変わらないが、表記は float64 の値（相対変化 2〜3%）に差し替えるのがよい。
- 修正案（v4）: 初期集団を**独立学習した多様な LoRA** で構成する。例: 役割ごとに seed・データ部分集合・ハイパラ・推論戦略型ペルソナを変えて K≥4 個体。変異は相対2%のガウスではなく、「再学習ステップ（ラマルク型）」「DARE の drop&rescale」「タスクベクトルの外挿」など意味のある大きさの演算子にする。

### 4.2 選抜がノイズに支配されている — **致命**
- 箇所: `loop.py:90-113, 232-249`（世代ごとに独立サンプルで再測定し、`max` で選抜。統計的な判定はない）
- 定量:
  - 同じ連合（名前集合が同一）を別の世代で測り直した23連合の、世代間の標準偏差をプールすると **3.07pt**（平均絶対差3.5pt、最大10pt）。
  - 例: gen0_critic_base の solo は g0:0.53 → g1:0.46 → g2:0.56 と振れた。
  - 一方、同じ役割内の候補間の Shapley 差は 0.002〜0.034（18ケース、適応度の値域は 0.158〜0.233）。
- 構造の注意:
  - 同じ文脈での候補 c と c′ の Shapley 差は、c を含まない連合の項が打ち消し合い、次の式になる。
    (1/3)Δsolo + (1/6)Δpair₁ + (1/6)Δpair₂ + (1/3)Δteam
  - 独立サンプルで測った4つの精度差の加重和なので、共通乱数なしでは差の標準誤差が大きい（100問で約3〜4pt）。
- 修正案:
  - §2.2 の round0 共有で、全連合と全候補の共通乱数化を行う。
  - 適応度セットを拡大・層化する。
  - 逐次半減や racing（差の信頼区間で打ち切り）を使う。
  - エリートを再評価したときは、同じ round0 サンプルでの対比較にする。
  - 世代の最後に、選ばれた新代表の組をチームとして同時評価する。今の実装では、選抜後の3代表の組はその世代では一度もチームとして測られていない。最終チーム（gen5 の選抜結果）は最終評価まで未測定である。

### 4.3 fitness sharing — 重大（K≥3 で顕在化する符号問題を含む）
- 箇所: `loop.py:130-133, 236-246`
- 事実:
  - K=2 では距離が対称なので、2個体の係数は常に等しく、選抜には中立になる（修論で既に指摘済み）。
  - 加えて、行動距離（solo 予測の不一致率 0.28〜0.45）は§2.1 のサンプリングノイズが支配しており、ほぼ同一の重み同士でもσ=0.3 を超える。つまり距離は「行動の違い」ではなく「サンプリングの違い」を測っている。
  - K≥3 にしても、`fitness = raw × penalty` は **raw（Shapley）が負のとき**、割り引くほど適応度が上がる（負値×1未満）。
- 修正案: 距離は共通乱数で測る（同じ round0 サンプル、または温度0の greedy solo 予測）。適応度を非負化する（例: Shapley−min）か、順位ベースの sharing にする。

### 4.4 ΔW ブレンド交叉の切り詰め損失と縮み — 中（v4で顕在化）
- 箇所: `lora_ops.py:62-114`
- 定量（α=0.5、厳密SVD、全252モジュール）:
  - 近縁同士の rank r 切り詰め損失は **0.00%** で、今までの実験では問題が表に出なかった。
  - 別ペルソナ同士では、gen0 で **11.3%**、run002 で **13.2%** のエネルギーを失う。
  - さらに親同士がほぼ直交しているので、α=0.5 のブレンド自体でノルムが約0.71倍になる。切り詰めと合わせると、子の更新量は親の**約0.66倍**になる（推定: 0.71×√0.887）。
  - 多様な親の交叉は「両親の特徴を持つ子」ではなく「薄まった子」を作りやすい。
- 修正案: 子のランクを 2r にする（vLLM の `--max-lora-rank` を 64 へ）、ノルムを保存する再スケール、TIES の符号合意や DARE の再スケール。交叉と変異の効果は、ΔW の差分ではなく行動（共通乱数での予測差）で検収する。

### 4.5 その他（軽微）
- `torch.svd_lowrank`（`lora_ops.py:97`）の乱数がシードされていないため、交叉の子はビット単位で再現しない（変異は seed で再現する）。修論の「完全に再現可能」は交叉には当てはまらない。
- `delta_blend_lora` は親の rank の一致しか検査しない。`lora_alpha` の一致は検査していない（γ が違う親をブレンドすると誤る）。
- 同じ適応度になったとき、`max` は先頭（エリート）を返す。これが僅かなエリート寄りのバイアスになる。

---

## 5. 学習

### 5.1 全トークンに損失が掛かる（assistant のみのマスクなし）— 中
- 箇所: `train_lora_persona.py:19-25, 98, 110-125`
  - `apply_chat_template(..., tokenize=False)` で `text` 列にしてから、`dataset_text_field="text"` で SFTTrainer に渡している。
  - TRL 0.24（uv.lock、`sft_trainer.py:338-361`）は、`text` 列には completion_mask も assistant_masks も作らない。そのため **system（日本語ペルソナ文）・user（設問）・assistant の全トークンが学習対象**になっている。
- 影響:
  - 60〜96例 × 2〜3エポックで、固定のペルソナ文と設問文（リプレイでは MATH train の問題文）を予測する学習が混ざっている。能力毀損の直接原因とまでは言えないが、「ペルソナSFT」の実態は仕様とずれている。
  - v4 で多様な個体を学習するなら、ここを直してから行うべき。
- 修正案: `{"prompt": [system, user], "completion": [assistant]}` の prompt-completion 形式にする（TRL の既定 `completion_only_loss=True` で assistant だけに損失が掛かる）。

### 5.2 学習 seed が伝わっていない — 中（v4 の多様性設計で重要）
- 箇所: `train_all_personas_vertex.py:240`（`seed_everything` は親プロセス内だけ）、`:285-308`（子プロセスに `--seed` を渡していない）、`train_lora_persona.py`（seed 引数なし、SFTConfig に seed を渡していない）
- 事実:
  - metadata と修論の「シード1234」に対し、実際のデータ順や dropout は TrainingArguments 既定の **seed=42** で決まっていた。
  - LoRA A の初期化は `get_peft_model` 時点で、Trainer が seed を設定する前に行われるため、**シードされていない**。
- 修正案: `--seed` を追加して `transformers.set_seed(seed)` を `get_peft_model` より前に呼び、`SFTConfig(seed=...)` も渡す。v4 で「seed 違いの個体」を多様性の源にするなら必須。

### 5.3 trainer イメージの依存が固定されていない — 中
- 箇所: `Dockerfile:21-23`（`COPY pyproject.toml README.md` だけで **uv.lock をコピーしていない**）
- 事実: `uv sync` がビルド時点の最新版を解決するため、trl・transformers・peft の実際の版は uv.lock（trl 0.24.0、transformers 4.57.1、peft 0.17.1）と違いうる。
- 修正案: uv.lock をコピーして `uv sync --frozen` にし、学習メタデータに `pip freeze` を残す。

### 5.4 その他
- `gradient_checkpointing=True` を、`prepare_model_for_kbit_training` や `use_reentrant=False` なしで使っている（軽微、要確認）。学習は実際に効いているので致命ではないが、v4 では明示するのが安全。
- リプレイの36例は3ペルソナで**同一**である（`make_run002_datasets.py`）。能力は守れるが、個体間の多様性を均す方向に働く。v4 で集団の多様性を狙うなら、個体ごとに別のリプレイ部分集合を使う。
- リプレイの正誤判定にも §1.1 の偽陰性が効く。角度・単位・`\sqrt2` などを含む問題は選ばれにくい（軽微な選択バイアス）。

---

## 6. トークン計測・確信度・記録

### 6.1 `finish_reason` と `usage` を記録していない — 中
- 箇所: `client.py:128-141`
- 影響:
  - 打ち切り（§1.2）を直接数えられない。
  - トークン数は再トークン化による推定になる（`token_accounting.py`。値の妥当性は呼び出し数で確認済み）。
- 修正案: `choice.finish_reason` と `response.usage.{prompt,completion}_tokens` を llm_calls に追記する。

### 6.2 SC の各サンプル回答を per_item に残していない — 軽微（ただし有用な再解析がある）
- 箇所: `run_eval.py:111-120`（多数決の結果だけをキャッシュ）
- llm_calls から再集計した結果（`scripts/analysis/audit_sc_from_logs.py`、新環境6シード）:

| | SC@1 | SC@3 | SC@6 | SC@9 |
|---|---|---|---|---|
| MMLU-Pro | 0.722 | 0.733 | 0.735 | 0.740 |
| MATH-500 | 0.848 | 0.865 | 0.873 | 0.873 |
| SuperGPQA | 0.449 | 0.471 | 0.481 | 0.486 |

  - **c7（1問6生成）− SC@6（同じ6生成）= −2.94pt [−3.90,−2.01] p<10⁻⁴**
  - c7 − SC@3（半分の3生成）= −2.35pt [−3.31,−1.40]
  - 「生成回数でチームが不利だった」という留保は不要になる（SC@6 との厳密な生成回数マッチでも有意に負ける）。修論の比較表に SC@k 曲線を加えることを推奨する。
- 修正案: SC の per_item に `samples: [...]` を保存する。今後の計算量マッチは、ログから SC@k をオフラインで作る。

---

## 7. 評価環境の再現性（系統差の温床）

### 7.1 可変タグ `:latest` とダイジェスト未記録 — **重大**
- 箇所: `scripts/cloud/submit_job.sh:45,72,109,142`（`imageUri: …/eval:latest`）、`cloud/cloudbuild*.yaml`（`:latest` を上書き push）
- 事実: +6pt の系統差は、イメージ再ビルドの前後でまさに発生した。結果 JSON にイメージのダイジェストも vLLM の版も残っていないので、事後の特定は難しい。
- 修正案:
  - ビルドごとに不変タグ（日付＋git sha）と `@sha256` ダイジェストでジョブを投入する。
  - `run_eval.py` の payload に次を記録する: イメージダイジェスト（環境変数で注入）、`vllm.__version__`、`pip freeze` のハッシュ、モデルと データセットの revision、top_p/top_k、debate_style、anonymize、aggregation。
  - 現状の payload（`run_eval.py:263-271`）は温度と max_tokens しか記録していない。

### 7.2 依存とデータの版固定 — 中
- `cloud/Dockerfile.eval:13` の `pip install "datasets>=3.0.0" "httpx>=0.27.0"` は下限指定だけ。共有依存（numpy、fsspec、pyarrow など）をビルド時点の版に動かしうる。→ constraints で固定する（`cloud/v4/constraints.txt` が既に作られつつあるようなので、そちらで吸収されるか確認）。
- HF のモデルとデータの revision は未指定。確認した範囲では、実験期間（2026-07）の前後で変化はない（Qwen3-4B-Instruct-2507 の最終コミットは 2025-09-17、MMLU-Pro のデータ更新は 2026-01-19、SuperGPQA は 2025-04-30）。今後の再実験に備え、`load_dataset(..., revision=<sha>)` と `--revision` で固定する。

### 7.3 失敗時の挙動 — 中
- `parallel.py:23-28` は例外をそのまま伝播させるので、1問の失敗でエントリ全体が落ちる。→ 問題単位で失敗を記録して続行し、最後にまとめて再試行する。
- `run_eval.py:131-134` の team 進捗キャッシュのラベルに debate_style・anonymize・rounds が入っていない。battery はエントリ名で dir を分けているので実害はなかったが、手動実行で同じ dir を使うと別条件の回答を黙って再利用する。

### 7.4 vLLM サーバ設定（v4 で顕在化）— 重大／中
- `cloud/entrypoint_eval.sh:8` の `MAX_MODEL_LEN` 既定は 8192。battery には `VLLM_MAX_MODEL_LEN=32768` が付いているが、**evolution ジョブ（`submit_job.sh:60-92`）には付いていない**。v4 で8192トークン生成＋他者発話の埋め込みを行うと 400 エラー → 3回リトライ → 例外 → ジョブ停止になる。
- `:9-10` の `MAX_LORA_RANK=32`、`MAX_LORAS=8`。個体数が8を超える集団や rank 64 の子を扱う v4 では、`--max-loras`・`--max-cpu-loras`・`--max-lora-rank` を引き上げる必要がある。

---

## 8. 新プロジェクト（pro-plasma-510112-m7）への移行で変える箇所

| ファイル | 内容 | 対応 |
|---|---|---|
| `scripts/cloud/submit_job.sh:16-20,28` | `PROJECT="research-501308"`、`REGION`、`BUCKET="gs://evo-swarm-lora-usc1-research-501308"`、`IMAGE_PREFIX`、`GCS_MOUNT` の固定パス | 環境変数化し、新バケット（us-central1 必須）へ |
| `scripts/cloud/submit_job.sh:13` | `CLOUDSDK_ACTIVE_CONFIG_NAME=evo-swarm` 前提 | 新方式（ディレクトリ専用の `CLOUDSDK_CONFIG`）に合わせる |
| `scripts/cloud/submit_job.sh:45,72,109,142` | `:latest` の可変タグ | ダイジェスト固定（§7.1） |
| `scripts/cloud/submit_job.sh:60-92` | evolution に `VLLM_MAX_MODEL_LEN` と `EVALX_LOG_DIR` 以外の設定なし | 32768、max_loras、rank を追加（§7.4） |
| `cloud/cloudbuild*.yaml` | `_REPO: evo-swarm`（新プロジェクトに Artifact Registry リポジトリが必要）、`:latest` | AR リポジトリ作成、不変タグ |
| `cloud/eval_battery_*.json`、`results/gcs/*/configs/*.json` | `/gcs/evo-swarm-lora-usc1-research-501308/...` のアダプタパスと `--exclude-items-file …/run_log.json` | 生成スクリプトで再生成する。除外リスト（fitness_items）はリポジトリ内のコピーを参照する形に |
| `results/evolution_run_log.json`、`results/gcs/run001/evolution/run_log.json` | 旧 GCS パスを含む（記録なので変更不要） | — |
| `AGENTS.md`、`results/gcs/README.md` | 旧プロジェクトの記述 | 「クラウド環境の現状」を更新 |
| `Dockerfile:21` | uv.lock を未コピー | §5.3 |
| アダプタの実体 | 旧 GCS（課金停止で消える可能性）→ `artifacts_local/` にある | 新バケットへアップロード |

補足: 作業中に `cloud/v4/`（Dockerfile・constraints・pip freeze）と `scripts/v4/audit_extractor.py` が並行して作られているのを確認した。本監査の §1〜§2・§7 と重なる部分は、そちらの実装で吸収されるかを突き合わせるとよい。

---

## 9. テストの穴（どれも現行の58テストでは検出されない）

1. MATH の正解側表記（`\sqrt2`、`\frac59`、`\frac9{19}`、`^\circ`、`\text{ cents}`、`x=5`、`x\in[…]`、`2516_8`、pmatrix、`5.5` と `\frac{11}{2}`）の同値判定
2. 選択肢抽出で `None of the above` → None、`A,B` → None、`<letter>` → None
3. round1 のプロンプトに自分の前回回答が含まれること（新スタイル）
4. round0 キャッシュの共有: 同じエージェント・同じ問題の round0 が、solo と全連合で同じテキストになること
5. タイブレークと GenSelect の順序が問題ごとに変わること
6. sharing が K≥3 で順位を変えること、負の Shapley で符号が逆転しないこと
7. 学習の seed の中継（同じ seed → 同じ LoRA 初期値）と、assistant のみの損失マスク
8. vLLM 設定の整合（`max_tokens` + プロンプト長 ≤ `max-model-len`）の事前検査

---

## 10. 修論の記述への影響（数値・主張が変わる箇所）

- MATH-500 の絶対精度は全条件で約6pt 高く直る。比較の符号は不変。**「MATH-500 で SC@9 と同等性未確定」→「同等性成立（+0.01pt [−1.07,+1.17]）」**。実験1の効果量は ±1pt 程度動く（c4 vs c1: −5.5→−3.7pt 等）。
- **G2（集約方式の選定）の結果は交絡している**（§2.1）。「weighted を採用」は根拠が薄く、実際に c7 seed1 でも weighted と多数決の差はない（§3.2）。
- **条件付き更新と匿名化の「効果」**は、自分の前回回答を見せない実装の上での話である（§3.1）。修論の解釈（追従対策）は、実装と対応するよう書き直しが必要。
- **進化の機能不全**の説明は、「演算子の摂動が小さい」だけでなく、「初期集団が同一個体の複製だった（多様性ゼロ）」と「共通乱数のない再測定ノイズ（SD 3.1pt）」を主因として書くのが正確（§4.1・§4.2）。ΔW 類似度は float64 の値（相対変化 2〜3%）に差し替える。
- 学習の seed 表記（1234）は実効値と違う（§5.2）。SFT の損失は全トークンに掛かっていた（§5.1）。手法の節で正確に書く。
- **SC@6（生成回数が厳密に同じ）でも c7 は −2.94pt と有意に負ける**（§6.2）。主結論は強化される。

---

## 付録: 本監査で追加したスクリプト（すべて GPU 不要・再実行可能）

| スクリプト | 内容 | 実行 |
|---|---|---|
| `scripts/analysis/audit_math_grading.py` | MATH 偽陰性の条件別集計と主要比較の再計算（`--impact`）、反転対の出力（`--dump-pairs`） | `uv run --no-project --with sympy --with numpy python … --impact` |
| `scripts/analysis/audit_math_vote_splitting.py` | SC@9 の票割れ（同値別表記）の影響 | `uv run --with sympy python …` |
| `scripts/analysis/audit_debate_transitions.py` | round0→round1 の回答遷移（多数派・少数派の変更率） | `uv run python …` |
| `scripts/analysis/audit_evolution_operators.py` | 進化演算子の実効摂動と ΔW 交叉の切り詰め損失（float64）→ `results/analysis_evolution_operators.json` | `uv run python …` |
| `scripts/analysis/audit_sc_from_logs.py` | SC@9 のログ整合性（6000/6000一致）、SC@k 曲線、c7 vs SC@3/6/9 | `uv run python …` |

MATH-500 の raw データは、未指定なら `~/.cache/evo_swarm_lora/math500_test.jsonl` に自動で取得する（HF `HuggingFaceH4/MATH-500`、トークン不要）。
