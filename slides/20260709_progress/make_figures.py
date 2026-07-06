# -*- coding: utf-8 -*-
"""進捗スライド用の図版を生成する。源暎ゴシックP・落ち着いた学術配色・フラットデザイン。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

AST = Path(__file__).parent / "assets"
AST.mkdir(exist_ok=True)

# ---- フォント登録 ----
GENEI = "/home/als0028/.local/share/fonts/genei"
for f in Path(GENEI).glob("*.otf"):
    fm.fontManager.addfont(str(f))
plt.rcParams["font.family"] = "GenEi Gothic P"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"

# ---- パレット（AIっぽくない、インク基調のフラット）----
INK     = "#22303C"   # 濃紺グレー：軸・見出し
SUB     = "#5A6B78"   # 補助テキスト
GRID    = "#DCE2E7"
PAPER   = "#FBFAF7"   # 生成りの紙色
BASE    = "#B7BEC6"   # ベース単体：薄グレー
SC9     = "#6E7B8A"   # SC@9：中グレー
TEAM_O  = "#C67A4E"   # 旧チーム：くすんだ橙茶
TEAM_N  = "#2E6E8E"   # 新チーム：ティールブルー（主役）
POS     = "#3F7D5A"   # プラス：深緑
NEG     = "#B4553F"   # マイナス：レンガ
CRITIC  = "#3B6EA5"
PRAG    = "#4E8D6E"
EXPLORE = "#8E6BB0"

plt.rcParams["text.color"] = INK
plt.rcParams["axes.edgecolor"] = INK
plt.rcParams["axes.labelcolor"] = INK
plt.rcParams["xtick.color"] = INK
plt.rcParams["ytick.color"] = INK


def _style(ax):
    ax.set_facecolor("none")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(INK)
        ax.spines[sp].set_linewidth(1.1)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(AST / name, dpi=200, bbox_inches="tight",
                facecolor="none", transparent=True)
    plt.close(fig)
    print("saved", name)


# ================================================================
# 図1: 最終結果（実験2・新環境）4条件×3ベンチ グループ棒
# ================================================================
def fig_final():
    benches = ["一般知識・推論\n(MMLU-Pro)", "数学\n(MATH-500)", "大学院級科学\n(SuperGPQA)"]
    data = {  # %
        "ベース単体":        ([72.7, 81.8, 43.1], BASE),
        "多数決9回 (SC@9)":  ([74.0, 87.3, 48.6], SC9),
        "旧チーム":          ([68.5, 79.3, 42.0], TEAM_O),
        "新チーム (処方後)":  ([71.6, 86.7, 43.1], TEAM_N),
    }
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    x = np.arange(len(benches)); w = 0.2
    for i, (lab, (vals, col)) in enumerate(data.items()):
        off = (i - 1.5) * w
        bars = ax.bar(x + off, vals, w, label=lab, color=col,
                      edgecolor="white", linewidth=0.8, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(benches, fontsize=11)
    ax.set_ylabel("正答率（％）", fontsize=11)
    ax.set_ylim(30, 95)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    _style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4,
              frameon=False, fontsize=9.5, handlelength=1.2, columnspacing=1.3)
    save(fig, "fig_final_results.png")


# ================================================================
# 図2: プールでの勝敗（新チームを基準にした差, pt）
# ================================================================
def fig_verdict():
    rows = [  # (ラベル, 差, 判定, 色)
        ("多数決9回（SC@9）に対して", -3.0, "負け  ( p<0.001 )", NEG),
        ("旧チームに対して（処方の効果）", +3.2, "勝ち  ( p<0.001 )", POS),
        ("ベース単体に対して", +0.3, "互角  ( 差なし )", SUB),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 3.3))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, d, tag, col) in zip(y, rows):
        ax.barh(yi, d, color=col, height=0.46, zorder=3,
                edgecolor="white", linewidth=0.8)
        # 値ラベルはバーと反対側に置いて重なりを避ける
        ax.text(d + (0.18 if d >= 0 else -0.18), yi, f"{d:+.1f}pt",
                va="center", ha="left" if d >= 0 else "right",
                fontsize=13, color=col)
    ax.axvline(0, color=INK, linewidth=1.1, zorder=2)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    # 判定タグを右の余白（軸外)へ
    for yi, (_, d, tag, col) in zip(y, rows):
        ax.annotate(tag, xy=(1.02, yi), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=10, color=col,
                    annotation_clip=False)
    ax.set_xlabel("新チームの正答率の差（ポイント, 6,000問プール）", fontsize=10)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(length=0)
    fig.subplots_adjust(left=0.28, right=0.80)
    save(fig, "fig_verdict.png")


# ================================================================
# 図3: 領域依存の反転（素の議論 − ベース, pt）
# ================================================================
def fig_domain():
    labs = ["一般知識・推論\n(MMLU-Pro)", "数学\n(MATH-500)", "大学院級科学\n(SuperGPQA)"]
    deltas = [-4.1, +4.0, +3.1]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(labs))
    cols = [NEG if d < 0 else POS for d in deltas]
    bars = ax.bar(x, deltas, 0.5, color=cols, zorder=3,
                  edgecolor="white", linewidth=1)
    for b, d in zip(bars, deltas):
        ax.text(b.get_x()+b.get_width()/2, d + (0.25 if d > 0 else -0.25),
                f"{d:+.1f}pt", ha="center",
                va="bottom" if d > 0 else "top", fontsize=12,
                color=POS if d > 0 else NEG)
    ax.axhline(0, color=INK, linewidth=1.2, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=11)
    ax.set_ylabel("議論による正答率の変化（pt）", fontsize=10.5)
    ax.set_ylim(-6, 6)
    ax.text(0, -5.3, "議論すると下がる", ha="center", fontsize=9.5, color=NEG)
    ax.text(1.5, 5.2, "議論すると上がる", ha="center", fontsize=9.5, color=POS)
    _style(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    save(fig, "fig_domain_flip.png")


# ================================================================
# 図4: 能力毀損と修復（思考の長さ）
# ================================================================
def fig_cot():
    # 左：思考の長さの圧縮 / 右：その結果の数学正答率の低下（いずれも同一環境の実測）
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"wspace": 0.42})

    ax = axes[0]
    labs = ["元のモデル", "性格を\n教えた後"]
    vals = [2423, 1188]
    bars = ax.bar([0, 1], vals, 0.5, color=[BASE, NEG], zorder=3,
                  edgecolor="white", linewidth=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+50, f"{v:,}字",
                ha="center", va="bottom", fontsize=11, color=INK)
    ax.annotate("", xy=(1, 1300), xytext=(0, 2350),
                arrowprops=dict(arrowstyle="-|>", color=SUB, lw=1.4,
                                connectionstyle="arc3,rad=-0.2"))
    ax.text(0.5, 2050, "約半分に圧縮", ha="center", fontsize=9.5, color=NEG)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labs, fontsize=10.5)
    ax.set_ylabel("思考（途中式）の長さ・中央値", fontsize=10)
    ax.set_ylim(0, 2900); ax.set_yticks([])
    ax.set_title("① 思考が短くなり", fontsize=11, color=INK, pad=10)
    _style(ax); ax.spines["left"].set_visible(False)

    ax = axes[1]
    vals2 = [83.0, 69.8]
    bars = ax.bar([0, 1], vals2, 0.5, color=[BASE, NEG], zorder=3,
                  edgecolor="white", linewidth=1)
    for b, v in zip(bars, vals2):
        ax.text(b.get_x()+b.get_width()/2, v+1.2, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=11, color=INK)
    ax.text(0.5, 40, "−13pt", ha="center", fontsize=13, color=NEG)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labs, fontsize=10.5)
    ax.set_ylabel("数学の正答率（単体）", fontsize=10)
    ax.set_ylim(0, 95); ax.set_yticks([])
    ax.set_title("② 数学が解けなくなる", fontsize=11, color=INK, pad=10)
    _style(ax); ax.spines["left"].set_visible(False)
    save(fig, "fig_cot_repair.png")


# ================================================================
# 図5: 計算量の使い方（SC@9 vs チーム）概念図
# ================================================================
def fig_budget():
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")

    # SC@9: 9個の独立サンプル
    ax.text(2.3, 3.9, "多数決9回（SC@9）", ha="center", fontsize=12, color=INK,
            fontweight="bold")
    ax.text(2.3, 3.5, "同じモデルが9回バラバラに解いて多数決", ha="center",
            fontsize=9, color=SUB)
    for i in range(9):
        cx = 0.55 + (i % 3) * 0.62
        cy = 2.75 - (i // 3) * 0.62
        c = plt.Circle((cx, cy), 0.24, color=SC9, ec="white", lw=1, zorder=3)
        ax.add_patch(c)
    ax.text(2.3, 0.55, "独立な9票（多様だが同じ視点）", ha="center",
            fontsize=9, color=SUB)

    # 仕切り
    ax.plot([4.9, 4.9], [0.3, 3.6], color=GRID, lw=1.4)

    # チーム: 3体 × 2ラウンド
    ax.text(7.5, 3.9, "議論チーム（本研究）", ha="center", fontsize=12, color=INK,
            fontweight="bold")
    ax.text(7.5, 3.5, "3つの性格 × 2ラウンド、途中で答えを見せ合う", ha="center",
            fontsize=9, color=SUB)
    cols = [CRITIC, PRAG, EXPLORE]
    names = ["批判", "実務", "探索"]
    for r in range(2):
        for i in range(3):
            cx = 5.9 + i * 0.85
            cy = 2.75 - r * 0.72
            c = plt.Circle((cx, cy), 0.26, color=cols[i], ec="white", lw=1, zorder=3)
            ax.add_patch(c)
            if r == 0:
                ax.text(cx, cy, names[i], ha="center", va="center",
                        fontsize=7.5, color="white")
        ax.text(5.2, 2.75 - r*0.72, f"R{r+1}", ha="center", va="center",
                fontsize=9, color=SUB)
    # ラウンド間の矢印（3列すべて）
    for i in range(3):
        cx = 5.9 + i * 0.85
        ax.add_patch(FancyArrowPatch((cx, 2.44), (cx, 2.29),
                     arrowstyle="-|>", mutation_scale=10, color=SUB, lw=1.1))
    ax.text(7.5, 0.55, "実質3つの視点（少数だが多様）", ha="center",
            fontsize=9, color=SUB)
    save(fig, "fig_budget.png")


if __name__ == "__main__":
    fig_final()
    fig_verdict()
    fig_domain()
    fig_cot()
    fig_budget()
    print("done")
