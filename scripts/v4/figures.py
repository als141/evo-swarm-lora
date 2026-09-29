"""修論の結果の図（第5章）。stats.py・secondary.py・パイロットの出力 JSON から PDF を作る。

使い方:
  python3 scripts/v4/figures.py --stats results/v4/stats_final.json --secondary results/v4/secondary.json \
      --pilot results/v4/pilot2_summary.json --out thesis/fig
入力のどれかが無い図は作らない。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

FONT = "Noto Sans CJK JP"
BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#718096", "#6b46c1"


def setup() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": FONT if FONT in names else "sans-serif",
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "figure.dpi": 150,
    })


def load(path: str | None):
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    return None


def fig_sc_curve(sec: dict, stats: dict | None, out: Path) -> None:
    """SC@k の精度–出力トークン曲線と、チーム条件の位置（RQ4）。"""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    styles = {"base": ("ベースモデルのSC@$k$", BLUE, "o"), "rft.base": ("RFTのSC@$k$", ORANGE, "s")}
    for agent, curve in sec.get("sc_curves", {}).items():
        if not curve.get("macro_by_k"):
            continue
        tok = curve["mean_out_tokens_per_call"]
        ks = sorted(int(k) for k in curve["macro_by_k"])
        xs = [k * tok for k in ks]
        ys = [100 * curve["macro_by_k"][str(k)] for k in ks]
        label, color, marker = styles.get(agent, (agent, GRAY, "o"))
        ax.plot(xs, ys, marker=marker, color=color, label=label, ms=3.5, lw=1.2)
        for k, x, y in zip(ks, xs, ys):
            if k in (1, 3, 9):
                ax.annotate(f"$k$={k}", (x, y), textcoords="offset points", xytext=(4, -9), fontsize=7,
                            color=color)
    team_style = {"g0": ("世代0の社会", GREEN, "D"), "S_final": ("系統S（最終世代）", RED, "*"),
                  "N_final": ("系統N（最終世代）", PURPLE, "^"), "A1_final": ("系統A1（最終世代）", GRAY, "v"),
                  "c7_july": ("7月のチーム", "black", "x")}
    for name, (label, color, marker) in team_style.items():
        t = sec.get("teams", {}).get(name)
        if not t or not t.get("n"):
            continue
        acc = 100 * (stats["accuracy"][name]["macro"] if stats and name in stats.get("accuracy", {})
                     else t["final_correct"])
        ax.scatter([t["out_tokens_per_item"]], [acc], color=color, marker=marker, s=60 if marker == "*" else 30,
                   label=label, zorder=5)
    ax.set_xlabel("1問あたりの出力トークン数")
    ax.set_ylabel("test の macro 精度（%）")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "sc_curve.pdf")
    plt.close(fig)


def fig_generations(sec: dict, stats: dict | None, out: Path) -> None:
    """系統ごとの世代推移（test macro）と基準の帯（RQ1〜RQ3）。"""
    lin = sec.get("lineages", {})
    if "S" not in lin:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    acc = (stats or {}).get("accuracy", {})

    def val(label, fallback):
        return 100 * acc[label]["macro"] if label in acc else (100 * fallback if fallback is not None else None)

    gens = lin["S"]["generations"]
    ts = sorted(int(t) for t in gens if gens[t].get("test_team") is not None)
    last = max(ts)
    ys = [val("g0" if t == 0 else ("S_final" if t == last else f"S_g{t}"), gens[str(t)]["test_team"]) for t in ts]
    ax.plot(ts, ys, marker="o", color=RED, label="系統S（Shapley選抜）")
    for name, color, marker in (("N", PURPLE, "^"), ("A1", GRAY, "v")):
        if name in lin:
            g = lin[name]["generations"]
            tl = max(int(t) for t in g if g[t].get("test_team") is not None)
            y = val(f"{name}_final", g[str(tl)]["test_team"])
            ax.plot([0, tl], [ys[0], y], ls="--", lw=0.8, color=color)
            ax.scatter([tl], [y], color=color, marker=marker, zorder=5,
                       label="系統N（選抜なし）" if name == "N" else "系統A1（単独精度で選抜）")
    for label, name, color in (("ベース単体", "base_single", BLUE), ("SC@3", "base_sc3", BLUE),
                               ("SC@9", "base_sc9", BLUE)):
        if name in acc:
            y = 100 * acc[name]["macro"]
            ax.axhline(y, color=color, lw=0.7, ls=":")
            ax.text(last + 0.05, y, label, fontsize=7, color=color, va="center")
    ax.set_xticks(ts)
    ax.set_xlabel("世代")
    ax.set_ylabel("test の macro 精度（%）")
    ax.set_xlim(-0.2, last + 0.7)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "generations.pdf")
    plt.close(fig)


def fig_shapley_solo(states: dict, out: Path) -> None:
    """候補ごとの Shapley 値と単独精度（dev）。選ばれた候補を強調する。"""
    st = states.get("S")
    if not st:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    colors = {"parent": GRAY, "a": BLUE, "b": ORANGE, "c": GREEN}
    seen = set()
    for t, g in st["generations"].items():
        chosen = {d["agent_id"] for d in g.get("reps", {}).values()}
        for agent_id, f in (g.get("fitness") or {}).items():
            if "shapley" not in f:
                continue
            variant = agent_id.split(".")[-1] if agent_id.split(".")[-1] in ("a", "b", "c") else "parent"
            label = {"parent": "親", "a": "子a（自己学習）", "b": "子b（社会学習）", "c": "子c（交叉）"}[variant]
            ax.scatter(100 * f["solo"], 100 * f["shapley"], color=colors[variant], s=28,
                       edgecolor="black" if agent_id in chosen else "none", lw=1.0,
                       label=None if label in seen else label)
            seen.add(label)
    ax.set_xlabel("単独の精度（dev, %）")
    ax.set_ylabel("厳密 Shapley 値（dev, pt）")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "shapley_solo.pdf")
    plt.close(fig)


def fig_persona(pilot: dict, out: Path) -> None:
    """段階1の単独精度と段階2の主効果（役割ごと）。"""
    st1, st2 = pilot.get("stage1"), pilot.get("stage2")
    if not st1 or not st2:
        return
    roles = list(st2["main_effect"].keys())
    names = {"plant": "Plant", "monitor_evaluator": "Monitor Eval.", "specialist": "Specialist", "shaper": "Shaper",
             "implementer": "Implementer", "completer_finisher": "Completer Fin.", "coordinator": "Co-ordinator",
             "teamworker": "Teamworker", "resource_investigator": "Resource Inv."}
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.0), sharey=True)
    ys = range(len(roles))
    solo = [100 * st1["solo_macro"][r] for r in roles]
    plain = [100 * st1["solo_macro"][p] for p in ("plain", "plain_b", "plain_c") if p in st1["solo_macro"]]
    axes[0].barh(list(ys), solo, color=BLUE, alpha=0.8)
    for p in plain:
        axes[0].axvline(p, color=GRAY, lw=0.7, ls=":")
    axes[0].set_xlim(min(solo + plain) - 3, max(solo + plain) + 2)
    axes[0].set_xlabel("段階1: 単独の精度（dev-A, %）")
    axes[0].set_yticks(list(ys))
    axes[0].set_yticklabels([names.get(r, r) for r in roles])
    eff = [100 * st2["main_effect"][r] for r in roles]
    axes[1].barh(list(ys), eff, color=[GREEN if e >= 0 else RED for e in eff], alpha=0.8)
    axes[1].axvline(0, color="black", lw=0.6)
    axes[1].set_xlabel("段階2: 主効果（dev-A, pt）")
    for ax in axes:
        ax.grid(alpha=0.3, lw=0.5, axis="x")
    fig.tight_layout()
    fig.savefig(out / "persona_selection.pdf")
    plt.close(fig)


def fig_transitions(sec: dict, out: Path) -> None:
    """議論の正誤遷移: 少数派の正解の維持率と、正→誤・誤→正の率（チーム条件ごと）。"""
    teams = [(n, t) for n, t in sec.get("teams", {}).items() if t.get("n")]
    if not teams:
        return
    label = {"g0": "世代0", "S_final": "S最終", "N_final": "N最終", "A1_final": "A1最終", "c7_july": "7月"}
    teams = [(n, t) for n, t in teams if n in label]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    width = 0.26
    xs = range(len(teams))
    series = [("minority_kept_rate", "少数派の正解の維持率", GREEN),
              ("correct_to_wrong_rate", "正→誤の率", RED), ("wrong_to_correct_rate", "誤→正の率", BLUE)]
    for j, (key, lab, color) in enumerate(series):
        vals = [100 * (t.get(key) or 0) for _, t in teams]
        ax.bar([x + (j - 1) * width for x in xs], vals, width, color=color, alpha=0.85, label=lab)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([label[n] for n, _ in teams])
    ax.set_ylabel("%")
    ax.grid(alpha=0.3, lw=0.5, axis="y")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(out / "transitions.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats")
    parser.add_argument("--secondary")
    parser.add_argument("--pilot")
    parser.add_argument("--state", action="append", default=[], help="系統名=state.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    setup()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats, sec, pilot = load(args.stats), load(args.secondary), load(args.pilot)
    states = {s.split("=", 1)[0]: load(s.split("=", 1)[1]) for s in args.state}
    if sec:
        fig_sc_curve(sec, stats, out)
        fig_generations(sec, stats, out)
        fig_transitions(sec, out)
    if states:
        fig_shapley_solo(states, out)
    if pilot:
        fig_persona(pilot, out)
    print(sorted(p.name for p in out.glob("*.pdf")))


if __name__ == "__main__":
    main()
