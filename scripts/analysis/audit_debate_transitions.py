"""議論 round0→round1 の回答遷移の監査（自己発話を再提示しないプロトコルの影響の定量）。

src/evalx/debate.py の round1 プロンプトは「他エージェントの発話」だけを提示し、
自分の round0 発話を再提示しない（"You may keep or change your previous answer" と
指示しているのに前回回答は見えない）。3 体チームで round0 が 2-1 に割れたとき、
多数派側のエージェントから見ると提示されるのは「賛成1・反対1」だけで、
自分の答えが多数派であるという情報が消える。本スクリプトは保存済み transcripts から
  - round0 の票構造（全員一致 / 2-1 / 1-1-1）別の変更率
  - 多数派/少数派ごとの変更率と正誤遷移（正→誤, 誤→正）
  - 「round0 多数決」と「round1 後の最終集約」の正答率
を集計する。

実行: uv run python scripts/analysis/audit_debate_transitions.py
"""

from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evalx.debate import majority_vote  # noqa: E402
from src.evalx.tasks import extract_answer, is_correct  # noqa: E402

GCS = ROOT / "results/gcs"

# (ラベル, transcripts パス, per_item を持つ結果 JSON のリスト, プロトコル)
SOURCES = [
    (
        "c7 run002チーム seed1（conditional+匿名化）",
        GCS / "run002/g3_team_check/transcripts_team.json.gz",
        [GCS / f"run002/g3_team_check/g3_{b}.json" for b in ("mmlu_pro", "math500", "supergpqa")],
    ),
    (
        "c5 進化後チーム MMLU-Pro demo（standard）",
        GCS / "run001/transcripts_demo/transcripts_team.json",
        [GCS / "run001/transcripts_demo/c5_team_mmlu_pro_s1_demo.json"],
    ),
    (
        "MATH-500 診断 3条件（standard）",
        GCS / "run001/transcripts_math500_diag/transcripts_team.json",
        [GCS / f"run001/transcripts_math500_diag/diag_{c}_team_math500.json" for c in ("base", "gen0", "evolved")],
    ),
]


def load_json(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def answer_type_of(item_id: str) -> str:
    return "math" if item_id.startswith("math500") else "letter"


def analyze(label: str, transcripts_path: Path, result_paths: list[Path]) -> None:
    gold = {}
    for path in result_paths:
        if not path.exists():
            continue
        d = load_json(path)
        for iid, r in d["team"]["per_item"].items():
            gold[iid] = r["gold"]
    records = load_json(transcripts_path)
    stats = collections.Counter()
    by_struct = collections.defaultdict(collections.Counter)
    for rec in records:
        iid = rec["item_id"]
        if iid not in gold or len(rec["rounds"]) < 2:
            continue
        atype = answer_type_of(iid)
        r0 = {a: extract_answer(t, atype) for a, t in rec["rounds"][0].items()}
        r1 = {a: extract_answer(t, atype) for a, t in rec["rounds"][1].items()}
        g = gold[iid]
        votes = collections.Counter(v for v in r0.values() if v is not None)
        top = votes.most_common(1)[0][1] if votes else 0
        struct = {3: "unanimous", 2: "2-1", 1: "split/none"}.get(top, "other")
        stats["items"] += 1
        stats["r0_majority_correct"] += is_correct(majority_vote(list(r0.values()), 0), g, atype)
        stats["r1_majority_correct"] += is_correct(majority_vote(list(r1.values()), 0), g, atype)
        stats["final_correct"] += is_correct(rec["majority_answer"], g, atype)
        for agent in r0:
            a0, a1 = r0[agent], r1.get(agent)
            side = "n/a"
            if struct == "2-1" and a0 is not None:
                side = "majority" if votes[a0] == 2 else "minority"
            key = f"{struct}:{side}"
            s = by_struct[key]
            s["agents"] += 1
            s["changed"] += a0 != a1
            c0, c1 = is_correct(a0, g, atype), is_correct(a1, g, atype)
            s["c->w"] += c0 and not c1
            s["w->c"] += (not c0) and c1
    n = stats["items"]
    print(f"\n=== {label}  (items={n})")
    print(
        f"  round0多数決 正答率={stats['r0_majority_correct']/n:.3f}  "
        f"round1多数決={stats['r1_majority_correct']/n:.3f}  最終集約={stats['final_correct']/n:.3f}"
    )
    print(f"  {'r0票構造:立場':28s} {'agents':>6s} {'変更率':>7s} {'正→誤':>6s} {'誤→正':>6s}")
    for key in sorted(by_struct):
        s = by_struct[key]
        print(
            f"  {key:28s} {s['agents']:6d} {s['changed']/s['agents']:7.3f} "
            f"{s['c->w']:6d} {s['w->c']:6d}"
        )


def main() -> None:
    for label, tpath, rpaths in SOURCES:
        if tpath.exists():
            analyze(label, tpath, rpaths)


if __name__ == "__main__":
    main()
