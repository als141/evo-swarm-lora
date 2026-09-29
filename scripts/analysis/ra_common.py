"""再解析（2026-09, docs/review_2026-09/reanalysis.md）の共通ローダ。

- 評価結果 JSON（per_item）の読み込みとパス規約（旧環境 final_eval3 / 新環境 run002）
- llm_calls 完全記録の走査と、呼び出し単位の索引キャッシュ（ra0_extract_calls.py が生成）
- 問題文 → 問題ID の索引（ra0_build_index.py が生成）

GCP・GPU は一切使わない。system python3（numpy/scipy/datasets）で動く。
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "results/gcs/run001"
R2 = ROOT / "results/gcs/run002"
E3 = R1 / "final_eval3"
OUT = ROOT / "results/reanalysis_2026-09"
CACHE = OUT / "cache"
QINDEX = CACHE / "question_index.json.gz"
CALLS = CACHE / "calls_index.jsonl.gz"

BENCHES = ["mmlu_pro", "math500", "supergpqa"]
ANSWER_TYPE = {"mmlu_pro": "letter", "math500": "math", "supergpqa": "letter"}
BENCH_OF_PREFIX = {"mmlupro": "mmlu_pro", "math500": "math500", "supergpqa": "supergpqa"}


def import_tasks_without_datasets():
    """src.evalx.tasks を datasets 非依存で import する（抽出・採点関数だけ使う）。"""
    if "datasets" not in sys.modules:
        try:
            import datasets  # noqa: F401
        except Exception:  # tokenizers だけの一時環境など
            stub = types.ModuleType("datasets")
            stub.load_dataset = None
            sys.modules["datasets"] = stub
    sys.path.insert(0, str(ROOT))
    from src.evalx import tasks  # noqa: E402

    return tasks


def qhash(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def bench_of_item(item_id: str) -> str:
    return BENCH_OF_PREFIX[item_id.split("-")[0]]


# ---------------------------------------------------------------- 結果 JSON
def per_item_maps(path: Path) -> Dict[str, Dict[str, dict]]:
    """結果 JSON を {系列名: {item_id: {predicted, gold, correct}}} に正規化する。

    team → {"team": ...}, sc → {"sc": ...}, solo → {agent名: ...}
    """
    d = json.loads(Path(path).read_text())
    if d.get("team"):
        return {"team": d["team"]["per_item"]}
    if d.get("sc"):
        return {"sc": next(iter(d["sc"].values()))["per_item"]}
    if d.get("solo"):
        return {name: v["per_item"] for name, v in d["solo"].items()}
    raise ValueError(f"unknown result format: {path}")


OLD_CONDS = {
    "c1": "c1_base_solo", "c2": "c2_sc9", "c3": "c3_base_team",
    "c3p": "c3p_prompt_persona_team", "c4": "c4_gen0_team",
    "c5": "c5_evolved_team", "c6": "c6_evolved_solo",
}


def old_path(cond: str, bench: str, seed: int) -> Path:
    return E3 / f"{OLD_CONDS[cond]}_{bench}_s{seed}.json"


def new_path(cond: str, bench: str, seed: int) -> Optional[Path]:
    """新環境（run002 系, 2026-07-05 イメージ）の結果ファイル。無ければ None。"""
    if cond == "c7":
        if seed == 1:
            p = R2 / f"g3_team_check/g3_{bench}.json"
        elif seed in (2, 3):
            p = R2 / f"final_c7/c7_run002_team_{bench}_s{seed}.json"
        else:
            p = R2 / f"robust_c7/c7_run002_team_{bench}_s{seed}.json"
    elif cond == "c2":
        if bench == "mmlu_pro" and seed == 1:
            p = R2 / "recheck_c2s1/c2_sc9_mmlu_pro_s1_recheck.json"
        elif seed <= 3:
            p = R2 / f"remeasure_v1/r_c2_sc9_{bench}_s{seed}.json"
        else:
            p = R2 / f"robust_c2/c2_sc9_{bench}_s{seed}.json"
    elif cond == "c1":
        p = R2 / f"remeasure_v1/r_c1_base_solo_{bench}_s{seed}.json"
    elif cond == "c5":
        p = R2 / f"remeasure_v1/r_c5_evolved_team_{bench}_s{seed}.json"
    else:
        raise ValueError(cond)
    return p if p.exists() else None


NEW_SEEDS = {"c7": range(1, 7), "c2": range(1, 7), "c1": range(1, 4), "c5": range(1, 4)}


def all_result_files() -> List[Path]:
    return sorted(p for p in (ROOT / "results/gcs").rglob("*.json")
                  if "battery_summary" not in p.name and "configs" not in p.parts
                  and "transcripts" not in p.name and "run_log" not in p.name
                  and "_demo" not in p.name)


# ---------------------------------------------------------------- 問題索引
def load_question_index() -> Dict[str, list]:
    with gzip.open(QINDEX, "rt") as fh:
        return json.load(fh)


def item_meta() -> Dict[str, dict]:
    """item_id → {bench, gold, category/difficulty/level 等}（索引から逆引き）。"""
    idx = load_question_index()
    meta = {}
    for _h, rec in idx.items():
        bench, item_id, gold, extra = rec
        meta[item_id] = {"bench": bench, "gold": gold, **extra}
    return meta


# ---------------------------------------------------------------- llm_calls
LLM_CALL_DIRS = ["g2_aggregation", "g3_team_check", "final_c7", "robust_c7",
                 "remeasure_v1", "recheck_c2s1", "robust_c2"]


def iter_raw_calls(dirs: List[str] = LLM_CALL_DIRS) -> Iterator[dict]:
    for d in dirs:
        for f in sorted((R2 / d / "llm_calls").glob("*.jsonl.gz")):
            with gzip.open(f, "rt") as fh:
                for line_no, line in enumerate(fh):
                    rec = json.loads(line)
                    rec["_dir"] = d
                    rec["_file"] = f.name
                    rec["_line"] = line_no
                    yield rec


def classify_call(rec: dict) -> dict:
    """seed と model から呼び出し種別（solo/sc/team/judge）と添字を復元する。

    solo: seed = s（1..9） / SC: seed = s*1000 + k / team: seed = s*10000 + 100*i + ρ
    GenSelect 裁定: system が 'candidate solutions' を含む（G2 のみ）
    """
    seed = rec.get("seed")
    model = rec["model"]
    system = rec["messages"][0]["content"] if rec["messages"] else ""
    out = {"kind": None, "run_seed": None, "agent_idx": None, "round": None, "sample_idx": None}
    if "candidate solutions" in system:
        out.update(kind="judge", run_seed=seed)
        return out
    if seed is None:
        return out
    if seed >= 10000:
        out.update(kind="team", run_seed=seed // 10000, agent_idx=(seed % 10000) // 100,
                   round=seed % 100)
    elif seed >= 1000:
        out.update(kind="sc", run_seed=seed // 1000, sample_idx=seed % 1000)
    else:
        out.update(kind="solo", run_seed=seed)
    out["family"] = ("c7" if model.startswith("r2_") else "c5" if model.startswith("ev_")
                     else "base")
    return out


def question_of_call(rec: dict) -> str:
    user = rec["messages"][-1]["content"]
    if user.startswith("Question:\n") and "\nHere are solutions from other agents:" in user:
        return user[len("Question:\n"):user.index("\nHere are solutions from other agents:")]
    if user.startswith("Question:\n"):
        # GenSelect: 'Question:\n{q}\n\n--- Candidate 1 ---'
        body = user[len("Question:\n"):]
        cut = body.find("\n\n--- Candidate")
        return body[:cut] if cut >= 0 else body
    return user


def load_calls_index() -> List[dict]:
    with gzip.open(CALLS, "rt") as fh:
        return [json.loads(line) for line in fh]


def group_calls(calls: List[dict], key_fields) -> Dict[tuple, List[dict]]:
    groups = defaultdict(list)
    for c in calls:
        groups[tuple(c[k] for k in key_fields)].append(c)
    return groups


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float))
