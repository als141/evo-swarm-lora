"""修論第5章の表（LaTeX の tabular 断片）をパイロット・最終解析の JSON から作る。

  python3 scripts/v4/tables.py --pilot results/v4/pilot2_summary.json --stats results/v4/stats_final.json \
      --out thesis/tab
出力: persona_stage1.tex / persona_stage2.tex / persona_stage3.tex / main_results.tex / comparisons.tex
各ファイルは表の行をマクロ（\\TabPersonaA 等）として定義する。tabular の中で \\input すると
直後の \\hline が壊れるため、表の前で \\input し、tabular の中ではマクロを置く。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"plant": "Plant", "monitor_evaluator": "Monitor Evaluator", "specialist": "Specialist",
         "shaper": "Shaper", "implementer": "Implementer", "completer_finisher": "Completer Finisher",
         "coordinator": "Co-ordinator", "teamworker": "Teamworker", "resource_investigator": "Resource Investigator",
         "plain": "plain", "plain_b": "plain（複製1）", "plain_c": "plain（複製2）"}
SHORT = {"plant": "PL", "monitor_evaluator": "ME", "specialist": "SP", "shaper": "SH", "implementer": "IMP",
         "completer_finisher": "CF", "coordinator": "CO", "teamworker": "TW", "resource_investigator": "RI",
         "plain": "plain", "plain_b": "plain", "plain_c": "plain"}
COND = {"g0": "世代0の社会", "g0_s1": "世代0の社会（seed 1）", "g0_s12": "世代0の社会（seed 1・2）", "S_final": "系統S（最終世代）", "S_final_s12": "系統S（最終世代，seed 1・2）",
        "N_final": "系統N（最終世代）", "A1_final": "系統A1（最終世代）", "S_g1": "系統S（世代1）", "S_g2": "系統S（世代2）",
        "g0_r0vote": "世代0（議論なしの多数決）", "S_final_r0vote": "系統S最終（議論なしの多数決）",
        "base_single": "ベースモデル単体", "base_sc3": "SC@3",
        "base_sc4": "SC@4", "base_sc4_exp": "SC@4（期待値）", "base_sc5": "SC@5", "base_sc5_exp": "SC@5（期待値）", "base_sc6": "SC@6", "base_sc9": "SC@9",
        "rft_single": "RFT単体", "rft_sc9": "RFTのSC@9", "c7_july": "7月のチーム"}


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}"


def signed(x: float, digits: int = 1) -> str:
    return f"{100 * x:+.{digits}f}".replace("-", "$-$")


def team_label(key: str) -> str:
    return "＋".join(SHORT[m] for m in key.split("+"))


def stage1(pilot: dict) -> str:
    s1 = pilot["stage1"]
    rows = []
    for name, acc in sorted(s1["solo_macro"].items(), key=lambda kv: -kv[1]):
        tax = s1["capability_tax"].get(name)
        exc = s1.get("excess_disagreement_vs_plain", {}).get(name)
        tax_s = (f"{signed(tax['diff_vs_plain'])} [{signed(tax['ci90'][0])}, {signed(tax['ci90'][1])}]"
                 if tax else "--")
        exc_s = signed(exc) if exc is not None else "--"
        rows.append(f"{NAMES[name]} & {pct(acc)} & {tax_s} & {exc_s} \\\\")
    return "\n".join(rows) + "\n"


def stage2(pilot: dict, pool: dict) -> str:
    s2 = pilot["stage2"]
    kind = {"+".join(sorted(t["members"])): ("同質型" if t["type"] == "homogeneous" else "バランス型")
            for t in pool["design_teams"]}
    kind["+".join(sorted(pool["default_trio"]))] = "既定トリオ"
    kind["+".join(pool["plain_team"])] = "plain×3"
    rows = []
    for key, acc in sorted(s2["team_acc_devA"].items(), key=lambda kv: -kv[1]):
        rows.append(f"{team_label(key)} & {kind.get(key, '')} & {pct(acc)} \\\\")
    return "\n".join(rows) + "\n"


def stage3(pilot: dict) -> str:
    s3 = pilot["stage3"]
    label = {"F0_default": "F0（既定トリオ）", "F1_best_observed": "F1（段階2の観測最良）",
             "F2_main_effect_top3": "F2（主効果の上位3）", "F3_solo_top3": "F3（単独の上位3）",
             "C_plain": "C（plain×3）"}
    rows = []
    for name, key in s3["finalists"].items():
        mark = "（採用）" if key == s3["adopted"] else ""
        rows.append(f"{label[name]} & {team_label(key)} & {pct(s3['acc_devB'][key])}{mark} \\\\")
    return "\n".join(rows) + "\n"


def main_results(stats: dict) -> str:
    rows = []
    for name, label in COND.items():
        a = stats["accuracy"].get(name)
        if not a or name.endswith("_s12") or name.startswith("S_g") or name == "g0_s1":
            continue
        seeds = a.get("gen_seeds") or []
        if a.get("type") == "sc_expect":
            note = "全組合せの平均"
        elif a.get("type") in ("sc", "single"):
            note = f"{a.get('k', 9)}本（seed {','.join(map(str, seeds))}）" if a.get("type") == "single" else f"最初の{a.get('k')}本"
        else:
            note = "seed " + ",".join(map(str, seeds))
        rows.append(f"{label} & {pct(a.get('mmlu_pro', float('nan')))} & {pct(a.get('supergpqa', float('nan')))} & "
                    f"{pct(a.get('math', float('nan')))} & {pct(a['macro'])} & {note} \\\\")
    return "\n".join(rows) + "\n"


def comparisons(stats: dict) -> str:
    holm = stats.get("holm_adjusted", {})
    order = {"主要": 0, "副次": 1, "探索": 2}
    rows = []
    items = sorted(stats["comparisons"].items(), key=lambda kv: order.get(kv[1].get("category", "探索"), 3))
    for key, c in items:
        a, b = key.split(" vs ")
        p = c["p"]
        p_s = "$<10^{-4}$" if p < 1e-4 else (f"{p:.4f}" if p < 1e-3 else f"{p:.3f}")
        h = holm.get(key)
        h_s = ("$<10^{-4}$" if h is not None and h < 1e-4 else (f"{h:.3f}" if h is not None else "--"))
        eq = "○" if (c.get("ci90") and -0.02 <= c["ci90"][0] and c["ci90"][1] <= 0.02) else ""
        rows.append(f"{c.get('category', '')} & {COND.get(a, a)} $-$ {COND.get(b, b)} & {signed(c['diff_macro'], 2)} & "
                    f"[{signed(c['ci95'][0], 2)}, {signed(c['ci95'][1], 2)}] & {p_s} & {h_s} & {eq} \\\\")
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot")
    parser.add_argument("--stats")
    parser.add_argument("--pool", default=str(ROOT / "configs/v4/persona_pool.json"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    made = []

    def write(name: str, macro: str, rows: str) -> None:
        (out / f"{name}.tex").write_text(f"\\gdef\\{macro}{{%\n{rows}}}\n")
        made.append(name)

    if args.pilot and Path(args.pilot).exists():
        pilot = json.loads(Path(args.pilot).read_text())
        pool = json.loads(Path(args.pool).read_text())
        if "stage1" in pilot:
            write("persona_stage1", "TabPersonaA", stage1(pilot))
        if "stage2" in pilot:
            write("persona_stage2", "TabPersonaB", stage2(pilot, pool))
        if "stage3" in pilot:
            write("persona_stage3", "TabPersonaC", stage3(pilot))
    if args.stats and Path(args.stats).exists():
        stats = json.loads(Path(args.stats).read_text())
        write("main_results", "TabMainResults", main_results(stats))
        write("comparisons", "TabComparisons", comparisons(stats))
    print(made)


if __name__ == "__main__":
    main()
