"""MATH-500 採点の偽陰性（正解なのに誤答判定）監査。

src/evalx/tasks.py の normalize_math_answer は正解側（raw LaTeX）の
`\\sqrt2` `\\frac59` `\\frac9{19}` `^\\circ` `\\text{ cents}` `x=5` `x \\in [...]` `2516_8`
`\\begin{pmatrix}` 等を正規化できず、モデルが数学的に同値な回答を書いても一致しない。
本スクリプトは per_item に保存された予測（既に正規化済みの文字列）と、
MATH-500 の raw 正解を改良正規化＋sympy 同値判定で再採点し、条件別の偽陰性率を出す。

使い方（プロジェクト依存不要。sympy は任意だが推奨）:
  uv run --no-project --with sympy python scripts/analysis/audit_math_grading.py \
      [--math500-jsonl PATH] [--dump-pairs out.tsv]

MATH-500 の raw データは未指定なら ~/.cache/evo_swarm_lora/math500_test.jsonl に取得する。
"""

from __future__ import annotations

import argparse
import collections
import functools
import glob
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATH500_URL = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"

try:  # 任意依存
    import sympy
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    _TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
except ImportError:  # pragma: no cover
    sympy = None


# --- プロジェクトの現行正規化（src/evalx/tasks.py と同一ロジックを複製; import は datasets 依存のため避ける）
def _normalize_number(text):
    cleaned = text.replace(",", "").replace("$", "").replace("%", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    number = float(match.group())
    return str(int(number)) if number == int(number) else str(number)


def project_normalize(text: str) -> str:
    answer = text.strip()
    boxed = re.search(r"\\boxed\{(.+)\}", answer)
    if boxed:
        answer = boxed.group(1)
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    answer = answer.replace("\\$", "").replace("$", "")
    answer = answer.replace("\\%", "").replace("%", "")
    answer = answer.replace("\\text{", "").replace("\\mathrm{", "")
    answer = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", answer)
    answer = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", answer)
    answer = answer.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    answer = answer.replace("{", "").replace("}", "").replace(" ", "")
    answer = answer.replace("\\", "")
    answer = answer.rstrip(".")
    numeric = _normalize_number(answer)
    if numeric is not None and re.fullmatch(r"-?[\d,.]+", answer):
        return numeric
    return answer.lower()


# --- 改良: raw LaTeX 正解の前処理（Hendrycks MATH の strip_string 系の修正を移植）
def _fix_sqrt(s: str) -> str:
    return re.sub(r"\\sqrt\s*([0-9a-zA-Z])", r"\\sqrt{\1}", s)


def _fix_fracs(s: str) -> str:
    # \frac59 -> \frac{5}{9}, \frac9{19} -> \frac{9}{19}, \frac{270}7 -> \frac{270}{7}
    s = re.sub(r"\\[dt]?frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\[dt]?frac\s*([0-9a-zA-Z])\s*\{", r"\\frac{\1}{", s)
    s = re.sub(r"\\[dt]?frac\s*(\{[^{}]*\})\s*([0-9a-zA-Z])", r"\\frac\1{\2}", s)
    return s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")


def _pmatrix_to_tuple(s: str) -> str:
    m = re.search(r"\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}", s, flags=re.S)
    if not m:
        return s
    body = m.group(1)
    rows = [r.strip() for r in re.split(r"\\\\", body) if r.strip()]
    if all("&" not in r for r in rows):
        return "(" + ",".join(rows) + ")"
    return "[" + ",".join("[" + ",".join(c.strip() for c in r.split("&")) + "]" for r in rows) + "]"


def fix_raw_latex(raw: str) -> str:
    s = raw.strip()
    s = _pmatrix_to_tuple(s)
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\circ", "")
    # 末尾の単位 (\text{ cents}, \mbox{ inches}^2, \text{ degrees})
    s = re.sub(r"(?<=[0-9}])\s*\\(?:text|mbox|mathrm)\{\s*[a-zA-Z ]+\}(\^\d)?\s*$", "", s)
    s = re.sub(r"^\s*[a-zA-Z]\s*\\in\s*", "", s)  # x \in [a,b]
    if len(s.split("=")) == 2 and len(s.split("=")[0].strip()) <= 2:  # x=5
        s = s.split("=")[1]
    s = re.sub(r"_\{?\d+\}?\s*$", "", s)  # 2516_8（基数の添字）
    s = _fix_sqrt(_fix_fracs(s))
    return s


@functools.lru_cache(maxsize=None)
def canon(norm: str) -> str:
    """プロジェクト正規化後の文字列に残る表記ゆれを追加で吸収する。"""
    s = norm
    s = s.replace("π", "pi").replace("°", "").replace("−", "-").replace("∪", "cup")
    s = re.sub(r"(?<=[\)\]])u(?=[\(\[])", "cup", s)  # (2,12)U(12,102)
    s = re.sub(r"√\(?([0-9a-z]+)\)?", r"sqrt(\1)", s)
    s = re.sub(r"sqrt([0-9a-z])", r"sqrt(\1)", s)
    s = re.sub(r"\^circ", "", s)
    s = re.sub(r"\bdegrees?$", "", s)
    s = re.sub(r"^[a-z]in(?=[\[\(])", "", s)  # xin[-2,7]
    if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    # 数値の直後に続く英字の単位・語（12thgrade, 5.4cents, 864mboxinches^2 等）
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)(?:st|nd|rd|th)?(?:mbox)?[a-z]+(?:\^\d)?", s)
    if m:
        s = m.group(1)
    # "27.iinitially...answer:27" のように答えの後に文章が続いたもの
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\.[a-z].*", s)
    if m:
        s = m.group(1)
    s = s.rstrip(".")
    numeric = _normalize_number(s)
    if numeric is not None and re.fullmatch(r"-?[\d,.]+", s):
        return numeric
    return s


def _sym(s: str):
    if sympy is None:
        return None
    # 3^75^10 のような多重べき・巨大指数は評価が止まらないので除外
    if re.search(r"\^\(?-?\d+\)?\^", s) or re.search(r"\^\(?\d{3,}", s):
        return None
    t = s.replace("pi", "(pi)").replace("^", "**")
    if not re.fullmatch(r"[0-9a-z+\-*/().,\s]*", t) or len(t) > 60:
        return None
    try:
        return parse_expr(t, transformations=_TRANSFORMS, local_dict={"pi": sympy.pi})
    except Exception:  # noqa: BLE001
        return None


@functools.lru_cache(maxsize=None)
def equivalent(gold_c: str, pred_c: str) -> bool:
    if gold_c == pred_c:
        return True
    # タプル/区間は要素ごと（括弧の種類は一致を要求）
    if gold_c[:1] in "([" and pred_c[:1] in "([" and gold_c[:1] == pred_c[:1] and gold_c[-1:] == pred_c[-1:]:
        ga, pa = gold_c[1:-1].split(","), pred_c[1:-1].split(",")
        if len(ga) == len(pa) and len(ga) > 1:
            return all(equivalent(g, p) for g, p in zip(ga, pa))
    a, b = _sym(gold_c), _sym(pred_c)
    if a is None or b is None:
        return False
    # simplify は遅いので数値評価で判定（自由変数があれば乱数代入を3点）
    try:
        symbols = sorted(a.free_symbols | b.free_symbols, key=str)
        points = [{}] if not symbols else [
            {s: v for s, v in zip(symbols, vals)}
            for vals in ((0.37, 1.91, 2.53), (1.13, 0.41, 3.07), (2.71, 1.61, 0.53))
        ]
        for point in points:
            va = complex(a.evalf(subs=point))
            vb = complex(b.evalf(subs=point))
            if abs(va - vb) > 1e-9 * max(1.0, abs(va)):
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


@functools.lru_cache(maxsize=None)
def gold_canon(raw_gold: str) -> str:
    return canon(project_normalize(fix_raw_latex(raw_gold)))


def load_math500(path: str | None):
    if path is None:
        cache = Path(os.path.expanduser("~/.cache/evo_swarm_lora/math500_test.jsonl"))
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MATH500_URL, cache)
        path = str(cache)
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def iter_blocks(path: str):
    d = json.load(open(path, encoding="utf-8"))
    for key in ("solo", "sc"):
        for agent, res in d.get(key, {}).items():
            yield f"{key}:{agent}", res
    if "team" in d:
        yield "team", d["team"]


def condition_of(path: str) -> str:
    base = os.path.basename(path).replace(".json", "")
    env = "old(run001)" if "/run001/" in path else "new(run002)"
    cond = re.sub(r"_math500_s\d+$", "", base)
    cond = re.sub(r"^r_", "", cond)
    cond = re.sub(r"_recheck$", "", cond)
    return f"{env}:{cond}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--math500-jsonl", default=None)
    parser.add_argument("--dump-pairs", default=None, help="偽陰性と判定した (gold, pred) 対を TSV で出力")
    parser.add_argument("--impact", action="store_true", help="主要比較の再計算（末尾で実行）")
    args = parser.parse_args()

    rows = load_math500(args.math500_jsonl)
    files = sorted(
        glob.glob(str(ROOT / "results/gcs/run001/final_eval3/*math500*.json"))
        + glob.glob(str(ROOT / "results/gcs/run002/*/*math500*.json"))
    )
    stats = collections.defaultdict(lambda: collections.Counter())
    flips = collections.Counter()
    gold_norm_mismatch = 0
    for f in files:
        cond = condition_of(f)
        for label, res in iter_blocks(f):
            key = cond if label == "team" or label.startswith("sc") else f"{cond}[{label.split(':')[1]}]"
            for iid, r in res["per_item"].items():
                idx = int(iid.split("-")[-1])
                raw_gold = rows[idx]["answer"]
                if project_normalize(raw_gold) != r["gold"]:
                    gold_norm_mismatch += 1
                gold_c = gold_canon(raw_gold)
                pred = r["predicted"]
                ok_orig = bool(r["correct"])
                ok_new = ok_orig or (pred is not None and equivalent(gold_c, canon(pred)))
                s = stats[key]
                s["n"] += 1
                s["orig"] += ok_orig
                s["new"] += ok_new
                s["none"] += pred is None
                if ok_new and not ok_orig:
                    s["fn"] += 1
                    flips[(iid, raw_gold, pred)] += 1

    print(f"[check] stored gold == project_normalize(raw gold): mismatches={gold_norm_mismatch}")
    print(f"{'condition':58s} {'n':>5s} {'orig':>7s} {'fixed':>7s} {'delta':>7s} {'FN':>5s} {'None':>5s}")
    agg = collections.defaultdict(lambda: collections.Counter())
    for key in sorted(stats):
        s = stats[key]
        print(
            f"{key:58s} {s['n']:5d} {s['orig']/s['n']:7.3f} {s['new']/s['n']:7.3f} "
            f"{(s['new']-s['orig'])/s['n']*100:+6.1f}pt {s['fn']:5d} {s['none']:5d}"
        )
        agg[key.split("[")[0]].update(s)
    print("\n--- 条件別（solo は3エージェント合算）---")
    for key in sorted(agg):
        s = agg[key]
        print(f"{key:58s} {s['n']:5d} {s['orig']/s['n']:7.3f} {s['new']/s['n']:7.3f} {(s['new']-s['orig'])/s['n']*100:+6.1f}pt")
    total = collections.Counter()
    for s in stats.values():
        total.update(s)
    print(f"\nTOTAL n={total['n']} FN={total['fn']} ({total['fn']/total['n']*100:.2f}% of predictions)")
    print(f"distinct flipped (item, gold, pred) = {len(flips)}")
    if args.dump_pairs:
        with open(args.dump_pairs, "w", encoding="utf-8") as fh:
            fh.write("count\titem_id\traw_gold\tpredicted_norm\n")
            for (iid, g, p), c in sorted(flips.items(), key=lambda x: -x[1]):
                fh.write(f"{c}\t{iid}\t{g}\t{p}\n")
        print(f"wrote {args.dump_pairs}")
    if args.impact:
        impact(rows)



# ---------------------------------------------------------------------------
# 影響評価: 修論の主要 MATH-500 比較を「現行採点」と「改良採点」で再計算する
#   uv run --no-project --with sympy --with numpy python scripts/analysis/audit_math_grading.py --impact
# ---------------------------------------------------------------------------
def _path_for(cond: str, seed: int) -> Path:
    r2 = ROOT / "results/gcs/run002"
    e3 = ROOT / "results/gcs/run001/final_eval3"
    if cond == "c7":
        if seed == 1:
            return r2 / "g3_team_check/g3_math500.json"
        if seed in (2, 3):
            return r2 / f"final_c7/c7_run002_team_math500_s{seed}.json"
        return r2 / f"robust_c7/c7_run002_team_math500_s{seed}.json"
    if cond == "c2new":
        if seed <= 3:
            return r2 / f"remeasure_v1/r_c2_sc9_math500_s{seed}.json"
        return r2 / f"robust_c2/c2_sc9_math500_s{seed}.json"
    if cond == "c1new":
        return r2 / f"remeasure_v1/r_c1_base_solo_math500_s{seed}.json"
    if cond == "c5new":
        return r2 / f"remeasure_v1/r_c5_evolved_team_math500_s{seed}.json"
    return e3 / f"{cond}_math500_s{seed}.json"


def _correct_maps(cond: str, seed: int, rows):
    d = json.load(open(_path_for(cond, seed), encoding="utf-8"))
    if d.get("team"):
        pi = d["team"]["per_item"]
    elif d.get("sc"):
        pi = next(iter(d["sc"].values()))["per_item"]
    else:
        pi = next(iter(d["solo"].values()))["per_item"]
    orig, fixed = {}, {}
    for iid, r in pi.items():
        g = gold_canon(rows[int(iid.split("-")[-1])]["answer"])
        orig[iid] = bool(r["correct"])
        fixed[iid] = bool(r["correct"]) or (r["predicted"] is not None and equivalent(g, canon(r["predicted"])))
    return orig, fixed


def impact(rows) -> None:
    import numpy as np

    rng = np.random.default_rng(20260930)

    def cluster(a, b, seeds, which):
        per_item = collections.defaultdict(list)
        for s in seeds:
            ma = _correct_maps(a, s, rows)[which]
            mb = _correct_maps(b, s, rows)[which]
            for k in ma.keys() & mb.keys():
                per_item[k].append(int(ma[k]) - int(mb[k]))
        d = np.array([np.mean(v) for v in per_item.values()])
        signs = rng.choice([-1.0, 1.0], size=(20000, len(d)))
        p = float((np.sum(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-15) + 1) / 20001)
        boot = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return d.mean() * 100, p, lo * 100, hi * 100, len(d)

    comps = [
        ("c7 vs SC@9 (new, 6seeds)", "c7", "c2new", range(1, 7)),
        ("c7 vs base (new, 3seeds)", "c7", "c1new", range(1, 4)),
        ("c7 vs old team c5 (new, 3seeds)", "c7", "c5new", range(1, 4)),
        ("base vs SC@9 (new, 3seeds)", "c1new", "c2new", range(1, 4)),
        ("exp1: c5 evolved vs c4 gen0 (old)", "c5_evolved_team", "c4_gen0_team", range(1, 4)),
        ("exp1: c3 plain debate vs c1 base (old)", "c3_base_team", "c1_base_solo", range(1, 4)),
        ("exp1: c5 evolved vs c2 SC@9 (old)", "c5_evolved_team", "c2_sc9", range(1, 4)),
        ("exp1: c4 gen0 vs c1 base (old)", "c4_gen0_team", "c1_base_solo", range(1, 4)),
    ]
    print(f"\n{'comparison (MATH-500)':42s} | {'orig Δpt [95%CI] p':32s} | {'fixed Δpt [95%CI] p':32s}")
    for label, a, b, seeds in comps:
        o = cluster(a, b, seeds, 0)
        f = cluster(a, b, seeds, 1)
        print(
            f"{label:42s} | {o[0]:+5.2f} [{o[2]:+5.2f},{o[3]:+5.2f}] p={o[1]:.4f} | "
            f"{f[0]:+5.2f} [{f[2]:+5.2f},{f[3]:+5.2f}] p={f[1]:.4f}  (n_items={o[4]})"
        )


if __name__ == "__main__":
    sys.exit(main())
