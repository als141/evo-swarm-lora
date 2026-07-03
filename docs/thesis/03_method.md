# 第3章 提案手法

本章では、マルチエージェント議論（Multi-Agent Debate; MAD）のチームレベル適応度を選択圧としてペルソナ LoRA 集団を進化させる提案手法を定式化する。まず 3.1 節で問題を形式的に定義し、3.2 節で議論プロトコル、3.3 節で適応度関数、3.4 節で進化アルゴリズムを述べ、3.5 節で計算量を分析する。本章の記述はすべて公開実装（`src/evalx/debate.py`、`src/evalx/shapley.py`、`src/evolve/loop.py`、`src/models/lora_ops.py`）と一対一に対応する。

## 3.1 問題定式化

### 3.1.1 ベースモデルと LoRA 個体

凍結されたベースモデル $\mathcal{M}_{\theta_0}$（本研究では Qwen3-4B-Instruct-2507）を共有し、その上に付加する LoRA（Low-Rank Adaptation）[Hu+ 2021] アダプタを進化の対象となる「個体」とする。LoRA は対象モジュール $m \in \mathcal{L}$（本実装では各 Transformer 層の $q, k, v, o$ 射影および $\mathrm{gate}, \mathrm{up}, \mathrm{down}$ 射影）の重み $W_0^{(m)} \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$ に対し、低ランク行列対 $B^{(m)} \in \mathbb{R}^{d_{\mathrm{out}} \times r}$、$A^{(m)} \in \mathbb{R}^{r \times d_{\mathrm{in}}}$ による更新

$$
W^{(m)} = W_0^{(m)} + \gamma\, B^{(m)} A^{(m)}, \qquad \gamma = \frac{\alpha_{\mathrm{LoRA}}}{r}
$$

を与える（$r = 32$、$\alpha_{\mathrm{LoRA}} = 2r$、したがって $\gamma = 2$ で全個体共通・固定）。個体 $x$ は行列対の集合 $x = \{(A^{(m)}, B^{(m)})\}_{m \in \mathcal{L}}$ として定義され、その実効的な表現型は各モジュールの更新行列（タスクベクトル [Ilharco+ 2022]）

$$
\Delta W^{(m)}(x) = B^{(m)} A^{(m)}
$$

の集合である。全個体は同一のベースモデル・同一の rank・同一の対象モジュール集合を共有するため、後述の交叉演算が層ごとに well-defined になる。

### 3.1.2 役割・エージェント・チーム

役割集合を $\mathcal{R} = \{\text{critic}, \text{pragmatist}, \text{explorer}\}$（批判的検証者・実務的意思決定者・発散的探索者）とし、各役割 $r$ に固有のペルソナプロンプト $\pi_r$（システムプロンプトとして与える短い役割記述）を割り当てる。エージェントは個体と役割プロンプトの組 $a = (x, \pi_r)$ である。役割ごとのペルソナは、初期化時に役割別 SFT データで重み空間にも焼き込まれる（第4章）。

世代 $t$ における役割 $r$ のサブ集団を $P_r^{(t)} = \{c_{r,1}^{(t)}, \dots, c_{r,K}^{(t)}\}$（本実装では $K = 2$）、その代表個体を $\rho_r^{(t)} \in P_r^{(t)}$ と書く。候補個体 $c \in P_r^{(t)}$ の評価は、他役割の代表と組んだ**代表チーム文脈**

$$
T(c) = \{c\} \cup \{\rho_{r'}^{(t)} \mid r' \in \mathcal{R},\ r' \neq r\}
$$

で行う。これは協調的共進化（Cooperative Coevolution）[Potter & De Jong 1994] における「協力者（collaborator）を固定した部分解評価」に相当する。

### 3.1.3 議論プロトコルと特性関数

質問 $q$ と正解 $y^*$ の組からなる固定の適応度セットを $D_{\mathrm{fit}} = \{(q_j, y_j^*)\}_{j=1}^{N_{\mathrm{fit}}}$（$N_{\mathrm{fit}} = 100$）とする。エージェント集合（連合）$S$ に対し、3.2 節の議論プロトコルを $R$ ラウンド実行して多数決回答を返す演算子を $\mathcal{D}_R(S, q)$ と書く（$|S| = 1$ のときは単独の Chain-of-Thought 回答）。連合 $S$ の**特性関数**（characteristic function）を適応度セット上の精度

$$
v(S) = \frac{1}{N_{\mathrm{fit}}} \sum_{j=1}^{N_{\mathrm{fit}}} \mathbb{1}\left[\mathcal{D}_R(S, q_j) = y_j^*\right], \qquad v(\emptyset) = 0
$$

と定義する。定義より $v(\{i\})$ は個体 $i$ の solo 精度に一致する。以上の記号を表 3.1 にまとめる。

**表 3.1: 記号表**

| 記号 | 意味 |
|---|---|
| $\mathcal{M}_{\theta_0}$ | 凍結ベースモデル（Qwen3-4B-Instruct-2507） |
| $x = \{(A^{(m)}, B^{(m)})\}$ | LoRA 個体（rank $r=32$、$\gamma = 2$） |
| $\Delta W^{(m)}(x) = B^{(m)}A^{(m)}$ | モジュール $m$ の実効更新行列 |
| $\mathcal{R}$, $\pi_r$ | 役割集合（3 役割）、役割別ペルソナプロンプト |
| $P_r^{(t)}$, $K$ | 世代 $t$・役割 $r$ のサブ集団、その規模（$K=2$） |
| $\rho_r^{(t)}$ | 役割 $r$ の代表個体（前世代の役割内最良） |
| $T(c)$ | 候補 $c$ の代表チーム文脈（3 エージェント） |
| $D_{\mathrm{fit}}$, $N_{\mathrm{fit}}$ | 適応度セット、その問題数（100） |
| $\mathcal{D}_R(S, q)$ | 連合 $S$ による $R$ ラウンド議論の多数決回答 |
| $v(S)$ | 特性関数（連合 $S$ の適応度セット精度、$v(\emptyset)=0$） |
| $\phi_c$ | 候補 $c$ の厳密 Shapley 値 |
| $d(c, c')$ | 行動距離（solo 予測の不一致率） |
| $\sigma_{\mathrm{share}}$ | fitness sharing の niche 半径（0.3） |
| $F(c)$ | 最終適応度 $\phi_c \cdot s(c)$ |
| $G$, $R$ | 世代数（6）、議論ラウンド数（1） |

## 3.2 議論プロトコル

議論プロトコルは Du らの MAD 原型 [Du+ 2023] に準拠し、近年の系統的検証 [Choi+ 2025; Kaesberg+ 2025; Demystifying MAD 2026] の知見に基づいて簡素化した以下の構造を採る。

**ラウンド構造**。各問題 $q$ に対し:

- **round 0（独立回答）**: 各エージェント $a_i = (x_i, \pi_{r_i})$ は、システムプロンプト（ペルソナ $\pi_{r_i}$ ＋回答形式指示）と質問のみを入力として独立に段階的推論と回答を生成する。初期回答の多様性が MAD 成功の主要条件であるとの報告 [Demystifying MAD 2026] に従い、独立生成を厳格に保つ。
- **round $1 \dots R$（批判的更新）**: 各エージェントに、**他エージェントの直前ラウンドの発話全文**を提示し、「他者の推論の誤りを指摘した上で、自身の解答を更新（または維持）せよ」と指示して回答を再生成する。自己の過去発話は再提示せず、各ラウンドは対話履歴を持たない独立の単一ターン呼び出しとして実装する。ラウンド数はラウンド追加が逆効果になるとの報告 [Kaesberg+ 2025] に基づき $R \in \{1, 2\}$、既定 $R = 1$ とする。

各生成には温度 $\tau = 0.7$・top-p $0.9$ のサンプリングを用い、乱数シードはエージェント添字 $i$ とラウンド添字 $\rho$ に応じて $\mathrm{seed}' = \mathrm{seed} \cdot 10^4 + 100 i + \rho$ と系統的にずらす。これにより同一重み・同一プロンプトの構成（第4章の温度サンプリング対照条件）でも初期回答の多様性が確保され、かつ全実行が再現可能になる。

**回答抽出と集約**。各エージェントの最終ラウンド発話から、厳密な書式指示 `ANSWER: <値>` に対する正規表現で最終回答を抽出する（抽出不能な発話は無効票とする）。集約は**最終ラウンド回答の多数決**であり、同数の場合は候補を辞書順に整列した上でシード付き乱数により一様に選ぶ。

judge（審判）エージェントによる集約を採用しない根拠は二つある。第一に、MAD の性能利得の大半は議論そのものではなく多数決集約で説明されるという分析 [Choi+ 2025]、および推論タスクでは合意形成プロトコルより投票が一貫して優れるという比較実験 [Kaesberg+ 2025] があり、集約規則としての多数決の優位が示されている。第二に、judge を置いた場合は judge 自身の判定能力がチーム性能のボトルネックになることが報告されており [arXiv:2511.11040]、本研究のように 4B 級の小型モデルのみでチームを構成する設定では特に深刻である。また、言語化された確信度（confidence）は過信バイアスを持つため主集約には用いない。

なお $|S| = 2$ の連合では多数決が常に同数となり得るため、実効的には「一致すればその回答、不一致ならシード付き乱数で一方を選択」という規則になる。これは Shapley 値算出（3.3 節）に必要な中間連合の評価としてプロトコルの一貫性を優先した設計である。

## 3.3 適応度関数

### 3.3.1 特性関数と厳密 Shapley 値

候補 $c$（役割 $r$）の適応度の中核は、代表チーム文脈 $T(c) = \{c, \rho_1, \rho_2\}$（$\rho_1, \rho_2$ は他 2 役割の代表）上の**厳密 Shapley 値** [Shapley 1953] である:

$$
\phi_c = \sum_{S \subseteq T(c) \setminus \{c\}} \frac{|S|!\,(n - |S| - 1)!}{n!} \Big( v(S \cup \{c\}) - v(S) \Big), \qquad n = 3.
$$

$n = 3$ を展開すると、係数は $|S| = 0, 1, 2$ に対しそれぞれ $\tfrac{1}{3}, \tfrac{1}{6}, \tfrac{1}{3}$ であり、

$$
\phi_c = \frac{1}{3} v(\{c\}) + \frac{1}{6} \Big[ v(\{c,\rho_1\}) - v(\{\rho_1\}) \Big] + \frac{1}{6} \Big[ v(\{c,\rho_2\}) - v(\{\rho_2\}) \Big] + \frac{1}{3} \Big[ v(T(c)) - v(\{\rho_1,\rho_2\}) \Big]
$$

となる。必要な特性関数値は 3 エージェントの全非空連合 $2^3 - 1 = 7$ 個であり、本研究ではこれを**すべて実測**する。すなわち Monte Carlo 近似や Leave-One-Out（LOO）近似 [Rozemberczki+ 2022] を一切用いず、公理的に一意な貢献配分を厳密に得る。エージェント数を 3 に限定した設計判断は、（i）Du らの標準構成であり奇数のため多数決が機能しやすいこと、（ii）$2^n$ に指数増加する連合評価が $n = 3$ でのみ実測可能な規模に収まること、の二点による。

Shapley 値を採る理論的根拠は、効率性（$\sum_i \phi_i = v(N) - v(\emptyset)$）・対称性・null player・線形性の 4 公理を同時に満たす唯一の配分規則である点にある [Shapley 1953; Ghorbani & Zou 2019]。特に LOO 型の寄与度 $v(N) - v(N \setminus \{c\})$ は、行動が同型な個体（複製）が連合に共存すると双方に寄与 0 を割り当てるという既知の欠陥を持つ [Ghorbani & Zou 2019]。進化計算では集団内に類似個体が発生しやすいため、この欠陥は選択圧を系統的に歪める。Shapley 値はすべての部分連合にわたる限界貢献の加重平均であるため複製に対しても頑健である。

また、上式の第 1 項が示すとおり $\phi_c$ は $v(\{c\})$（solo 精度）を重み $\tfrac{1}{3}$ で内包する。したがって個体タスク性能を別項として適応度に加算する必要はなく、「個体性能＋協調寄与」が単一のスカラーに公理的に統合される。これが個体性能のみを適応度とする先行研究（GENOME [Zhang+ 2025]）との本質的な差分であり、第4章のアブレーション A1（適応度を $v(\{c\})$ に置換）で直接検証する。

### 3.3.2 Fitness sharing と行動距離

Shapley 値のみを選択圧とすると、役割内の個体が同一の行動様式に収束し、進化の探索が早期に停滞する恐れがある。そこで Goldberg & Richardson の fitness sharing [Goldberg & Richardson 1987] による乗法ペナルティを導入する。同役割サブ集団内の個体間の**行動距離**を、適応度セット上の solo 予測の不一致率

$$
d(c, c') = \frac{1}{|D_{\mathrm{fit}}|} \sum_{j} \mathbb{1}\left[ \hat{y}_j(c) \neq \hat{y}_j(c') \right]
$$

（$\hat{y}_j(\cdot)$ は問題 $j$ に対する solo 抽出回答）と定義し、sharing 係数を

$$
s(c) = \frac{1}{\displaystyle 1 + \sum_{c' \in P_r^{(t)} \setminus \{c\}} \max\!\left(0,\ 1 - \frac{d(c, c')}{\sigma_{\mathrm{share}}}\right)}, \qquad \sigma_{\mathrm{share}} = 0.3
$$

とする。分母は $c$ の属する niche の実効個体数（niche count）であり、行動距離が $\sigma_{\mathrm{share}}$ 未満の同役割個体が近くに多いほど適応度が割り引かれる。最終的な適応度は

$$
F(c) = \phi_c \cdot s(c)
$$

である。行動距離の計算に必要な solo 予測は Shapley 値算出時の単独連合評価（$v(\{c\})$ の実測）の副産物として得られるため、追加の推論コストは発生しない。重み空間のパラメータ距離ではなく行動空間の距離を用いるのは、LoRA の再パラメータ化不変性（3.4.3 節）によりパラメータ距離が表現型の差を反映しないためである。

### 3.3.3 多様性を加算しない設計判断

多様性の扱いとして「適応度 $=$ 性能 $+ \lambda \cdot$ 多様性」という加重和がしばしば用いられるが、本研究はこれを採らない。根拠は二系統ある。第一に、Quality-Diversity 文献の中核である MAP-Elites [Mouret & Clune 2015] は、多様性を目的関数に加算するのではなく行動記述子による niche 分離（アーカイブ）として性能最適化から構造的に分離すべきであると論じており、Novelty Search [Lehman & Stanley 2011] 以来の知見として、加重和は探索の欺瞞性（deception）を単に係数 $\lambda$ の調整問題にすり替えるに過ぎない。第二に、アンサンブル学習の統一理論 [Wood+ 2023] は、多様性が bias・variance と並ぶ第三の項として汎化誤差分解に現れる**トレードオフの管理対象**であり、無条件に最大化・加算すべき量ではないことを数学的に示した。加重和はさらに、目的間のスケール不整合と係数の恣意性、非凸 Pareto 前線を表現できないという多目的最適化上の既知の限界 [Deb+ 2002] も抱える。

そこで本研究は、（i）選択圧そのものはチーム貢献 $\phi_c$ に一元化し、（ii）多様性は fitness sharing の**乗法ペナルティ**として「同一 niche への密集」を抑制する形でのみ作用させる、という分離設計を採る。乗法形式は性能と多様性の単位の混合を避け、$\phi_c$ の順位構造を niche 内でのみ再調整する。sharing の寄与自体はアブレーション A2（sharing 無効化）で分離検証する。

## 3.4 進化アルゴリズム

### 3.4.1 協調的共進化の構造

全体構造は Potter & De Jong の協調的共進化 [Potter & De Jong 1994] に従い、3 役割それぞれに独立のサブ集団 $P_r^{(t)}$（各 $K = 2$ 個体、計 6 LoRA）を保持する。各候補の評価は 3.1.2 節の代表チーム文脈で行う。すなわち候補 $c$ は「他役割の現代表 $\rho_{r'}$ と組んだときにチームへどれだけ貢献するか」で測られ、役割間の相互適応がチームレベルの選択圧として実現される。代表を固定する評価方式は、Population Based Training [Jaderberg+ 2017] の exploit（良い個体の活用）に対応し、評価対象の組合せ爆発を防ぐ。

**初期化（世代 0）**。役割別 SFT データ（各 60 例）で QLoRA 学習した 3 個体を各役割の初期代表とし、さらに各個体に全テンソル対象のガウス突然変異（変異率 $\rho_{\mathrm{mut}} = 1.0$、3.4.4 節と同一の相対ノイズ幅）を施した変異体を 1 個体ずつ加え、各役割 $K = 2$ の初期サブ集団を構成する。

### 3.4.2 選抜とエリート保存

世代 $t$ において、役割 $r$ の全候補の適応度 $F(c)$ を計算した後、役割内で最良の個体を次期代表に選ぶ:

$$
\rho_r^{(t+1)} = \operatorname*{arg\,max}_{c \in P_r^{(t)}} F(c).
$$

選ばれた代表は**エリート保存**により無変更のまま次世代サブ集団に残る。したがって各役割の代表系列の適応度（同一評価条件下）は非減少であり、突然変異による劣化がチーム全体へ伝播することを防ぐ。

### 3.4.3 交叉: ΔW 空間ブレンドとランダム化 SVD 再分解

**交差項問題**。LoRA 個体の交叉として最も素朴なのは、行列 $A$、$B$ を別々に線形補間する方式（naive 方式）である:

$$
A' = (1-\alpha) A_1 + \alpha A_2, \qquad B' = (1-\alpha) B_1 + \alpha B_2.
$$

しかしこのとき実効更新は

$$
\Delta W_{\mathrm{naive}} = B'A' = (1-\alpha)^2 B_1 A_1 + \alpha^2 B_2 A_2 + \alpha(1-\alpha)\big(B_1 A_2 + B_2 A_1\big)
$$

となり、意図した補間 $(1-\alpha)\Delta W_1 + \alpha \Delta W_2$ とは一致しない。両親の寄与が二乗係数で縮み（$\alpha = 0.5$ で各 $0.25$）、さらに意味を持たない交差項 $B_1 A_2$、$B_2 A_1$ が混入する。加えて LoRA 分解には再パラメータ化の自由度があり、任意の正則行列 $R$ に対して $BA = (BR)(R^{-1}A)$ が同一の $\Delta W$ を与えるため、$A$、$B$ を別々に扱う演算は同一個体の等価な表現の間でも結果が変わってしまう。KnOTS [Stoica+ 2025] は、独立に学習された LoRA 由来の $\Delta W$ は互いの整合（alignment）が低く、この種の素朴なパラメータ空間マージが性能劣化の主因になることを示している。

**提案する交叉（ΔW 空間ブレンド＋SVD 再分解）**。本研究の主方式は、補間を再パラメータ化不変な $\Delta W$ 空間で行う。各モジュール $m$ について:

1. 実効更新行列を補間する:
$$
\Delta W'^{(m)} = (1-\alpha)\, B_1^{(m)} A_1^{(m)} + \alpha\, B_2^{(m)} A_2^{(m)}, \qquad \alpha \sim \mathcal{U}(0.3,\, 0.7).
$$
2. $\Delta W'^{(m)}$ をランダム化 SVD [Halko+ 2011]（oversampling $q = \min(r + 8, \min(d_{\mathrm{out}}, d_{\mathrm{in}}))$、power iteration 4 回）で分解し、上位 $r$ 成分 $U_r \Sigma_r V_r^\top$ に切り詰めて子の LoRA 行列対を再構成する:
$$
B'^{(m)} = U_r \Sigma_r^{1/2}, \qquad A'^{(m)} = \Sigma_r^{1/2} V_r^\top.
$$

これにより子の実効更新 $B'A' = U_r \Sigma_r V_r^\top$ は、補間された $\Delta W'$ の Frobenius ノルム最良 rank-$r$ 近似（Eckart–Young の意味で、ランダム化 SVD による近似精度の範囲内）となり、交差項の混入と再パラメータ化依存性の両方が排除される。特異値の平方根を $A$、$B$ に均等配分するのは、両行列のスケールを揃え後続の突然変異ノイズ（テンソルの標準偏差に対する相対スケール）が偏らないようにするためである。親同士の rank と対象モジュール集合の一致は実行時に検証する。naive 方式は実装内に保持し、アブレーション A3 として比較する。

**交叉の親選択**。役割 $r$ の子は、次期代表 $\rho_r^{(t+1)}$ を第一親、同役割サブ集団の残余個体のうち適応度最良のものを第二親として生成する（$K = 2$ では残余は 1 個体）。混合比 $\alpha$ は世代・役割ごとに $\mathcal{U}(0.3, 0.7)$ から再抽出する。

### 3.4.4 ガウス突然変異

交叉で得た子アダプタの各テンソル $W_k$ に対し、独立に確率 $\rho_{\mathrm{mut}} = 0.3$ で加法的ガウスノイズを付加する:

$$
W_k \leftarrow W_k + \varepsilon_k, \qquad \varepsilon_k \sim \mathcal{N}\!\big(0,\ (\sigma_{\mathrm{mut}} \cdot \mathrm{std}(W_k))^2 I\big), \qquad \sigma_{\mathrm{mut}} = 0.02.
$$

ノイズ幅を各テンソルの経験標準偏差 $\mathrm{std}(W_k)$ に対する相対値とすることで、層・モジュールごとに大きく異なるパラメータスケールに自動適応する。突然変異はシード指定により完全に再現可能である。

### 3.4.5 世代ループ全体

以上を統合した手続きを Algorithm 1 に示す。世代数は $G = 6$ とする。次世代サブ集団は各役割とも「エリート代表 1 ＋交叉・突然変異による子 1」の $K = 2$ 構成である。

```
Algorithm 1: チームレベル Shapley 適応度による協調的共進化
─────────────────────────────────────────────────────────────
入力: 役割別 SFT 済み LoRA {x_r}_{r∈R}, 適応度セット D_fit,
      世代数 G, ラウンド数 R, σ_share, ρ_mut, σ_mut
出力: 最終世代の代表チーム {ρ_r^(G)}_{r∈R}

1:  for r ∈ R do                                   ▷ 世代 0 の初期化
2:      P_r ← { x_r, Mutate(x_r; ρ=1.0, σ_mut) };  ρ_r ← x_r
3:  for t = 0, …, G−1 do
4:      連合評価キャッシュ C ← ∅
5:      for r ∈ R, c ∈ P_r do                      ▷ 適応度評価
6:          T(c) ← {c} ∪ {ρ_r' | r' ≠ r}
7:          for S ⊆ T(c), S ≠ ∅ do
8:              if S ∉ C then C[S] ← v(S)          ▷ debate/solo を実測
9:          φ_c ← Shapley(c, {C[S]})                    (式 3.3.1)
10:     for r ∈ R, c ∈ P_r do                      ▷ fitness sharing
11:         d(c,c') ← solo 予測不一致率 (C の副産物)
12:         F(c) ← φ_c · s(c)                           (式 3.3.2)
13:     for r ∈ R do                               ▷ 選抜（エリート保存）
14:         ρ_r ← argmax_{c∈P_r} F(c)
15:     if t < G−1 then                            ▷ 次世代生成
16:         for r ∈ R do
17:             p ← argmax_{c∈P_r∖{ρ_r}} F(c)
18:             α ~ U(0.3, 0.7)
19:             child ← Mutate(DeltaBlendSVD(ρ_r, p, α); ρ_mut, σ_mut)
20:             P_r ← { ρ_r, child }
21: return {ρ_r}_{r∈R}
─────────────────────────────────────────────────────────────
```

全個体は vLLM の動的 LoRA ロード API により同一サーバへ世代ごとに登録され、進化は勾配計算を一切伴わない（重み操作は交叉・突然変異のみ）。世代ごとの評価ログ（各候補の $\phi_c$、solo 精度、チーム精度、全連合精度、sharing 距離、選抜結果）は JSON として逐次永続化され、Spot インスタンスのプリエンプション後も再開可能である。

## 3.5 計算量の分析

**連合評価数**。本手法の支配的コストは LLM 推論による連合評価である。候補 1 個体あたり必要な連合は $2^3 - 1 = 7$ だが、代表チーム文脈の構造上、多くの連合が候補間で共有される。世代内の連合評価キャッシュを用いると、代表のみからなる連合（solo 3 ＋ 対 3 ＋ 3 体 1 の計 7 個）は全候補で共有され、代表でない候補（各役割の子、計 $m$ 個体）だけが固有の連合（自身の solo 1、代表との対 2、3 体 1 の計 4 個）を追加する。したがって世代あたりの実評価数は

$$
N_{\mathrm{coal}} = 7 + 4m
$$

であり、$K = 2$（役割あたり非代表候補 1）では $m = 3$、$N_{\mathrm{coal}} = 19$ となる。キャッシュがなければ $6 \times 7 = 42$ 評価が必要であり、共有キャッシュにより約 55% が削減される。なおエリート代表自身も候補として評価されるが、その代表チーム文脈は代表のみの 7 連合と完全に一致するため追加コストを生まない。

**LLM 呼び出し数**。連合 $S$ の 1 問題あたりの生成回数は $|S| \cdot (R+1)$（round 0 ＋ $R$ ラウンド、各ラウンドで全員が 1 生成）である。$R = 1$、$N_{\mathrm{fit}} = 100$ のとき、世代あたりの呼び出し数は

$$
\underbrace{100 \times (3 \cdot 1 + 3 \cdot 4 + 1 \cdot 6)}_{\text{代表のみの 7 連合}} + \underbrace{3 \times 100 \times (1 \cdot 1 + 2 \cdot 4 + 1 \cdot 6)}_{\text{子 3 個体の固有 12 連合}} = 2{,}100 + 4{,}500 = 6{,}600
$$

であり、$G = 6$ 世代の進化全体で約 $4.0 \times 10^4$ 回（各生成は最大 512 トークン）となる。問題間は独立なため、クライアント側で問題単位に並列化（既定 16 並列）し、vLLM サーバの連続バッチングで吸収する。

**その他のコスト**。厳密 Shapley 値の計算自体は $n = 3$ の全部分集合和で $O(2^n)$ だが $n = 3$ では無視できる。fitness sharing の行動距離は solo 評価の per-item 予測を再利用するため追加推論はゼロである。交叉のランダム化 SVD は各モジュールの $d_{\mathrm{out}} \times d_{\mathrm{in}}$ 行列に対する rank-$(r+8)$ 分解であり、全体で CPU 数十秒程度と、推論コストに対して無視できる。進化全体で学習（勾配更新）は世代 0 の QLoRA 学習のみに限られる点が、RL による議論最適化（MAPoRL [Park+ 2025]）と対照的な本手法の計算的特徴である。

## 参考文献（第3章）

- Choi, S. et al. (2025). *Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?* arXiv:2508.17536.
- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2).
- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). *Improving Factuality and Reasoning in Language Models through Multiagent Debate.* arXiv:2305.14325 (ICML 2024).
- Ghorbani, A., & Zou, J. (2019). *Data Shapley: Equitable Valuation of Data for Machine Learning.* arXiv:1904.02868 (ICML 2019).
- Goldberg, D. E., & Richardson, J. (1987). Genetic Algorithms with Sharing for Multimodal Function Optimization. *Proc. 2nd International Conference on Genetic Algorithms*.
- Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions. *SIAM Review*, 53(2). arXiv:0909.4061.
- Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685 (ICLR 2022).
- Ilharco, G. et al. (2022). *Editing Models with Task Arithmetic.* arXiv:2212.04089 (ICLR 2023).
- Jaderberg, M. et al. (2017). *Population Based Training of Neural Networks.* arXiv:1711.09846.
- Kaesberg, L. B. et al. (2025). *Voting or Consensus? Decision-Making in Multi-Agent Debate.* arXiv:2502.19130 (ACL 2025 Findings).
- Krogh, A., & Vedelsby, J. (1994). Neural Network Ensembles, Cross Validation, and Active Learning. *NeurIPS 7*.
- Lehman, J., & Stanley, K. O. (2011). Abandoning Objectives: Evolution through the Search for Novelty Alone. *Evolutionary Computation*, 19(2).
- Mouret, J.-B., & Clune, J. (2015). *Illuminating Search Spaces by Mapping Elites.* arXiv:1504.04909.
- Park, C. et al. (2025). *MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning.* arXiv:2502.18439 (ACL 2025).
- Potter, M. A., & De Jong, K. A. (1994). A Cooperative Coevolutionary Approach to Function Optimization. *PPSN III*.
- Rozemberczki, B. et al. (2022). The Shapley Value in Machine Learning. *IJCAI 2022 (Survey Track)*.
- Shapley, L. S. (1953). A Value for n-Person Games. In *Contributions to the Theory of Games II*, Princeton University Press.
- Stoica, G. et al. (2025). *Model Merging with SVD to Tie the Knots (KnOTS).* arXiv:2410.19735 (ICLR 2025).
- Wood, D., Mu, T., Webb, A., Reeve, H., Luján, M., & Brown, G. (2023). A Unified Theory of Diversity in Ensemble Learning. *JMLR*, 24. arXiv:2301.03962.
- Zhang, Y. et al. (2025). *GENOME: GenerativE Neuro-symbOlic Model Evolution.*（LoRA 集団の進化的最適化）arXiv:2503.01155.
- （judge ボトルネックの報告）arXiv:2511.11040.
- （Demystifying MAD: 初期多様性と confidence 条件付き更新が成功条件）arXiv:2601.19921.

> **注**: 提出前に全文献の原典・書誌情報を確認すること（docs/literature_notes.md 冒頭の注意書き参照）。
