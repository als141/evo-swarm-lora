"""SC@9 の LLM 完全ログからの再集計（記録の整合性チェック＋生成回数マッチの SC@k）。

run002（新環境）の SC@9 は 1 問 9 サンプルの生成全文が llm_calls に残っている
（seed = s*1000 + k, k=0..8）。本スクリプトは
  (1) ログから抽出・多数決を再現し、保存済み per_item の SC@9 予測と一致するか検証
  (2) 先頭 k サンプルでの SC@k（k=1..9）の精度
  (3) チーム c7（1問6生成）と SC@6（同じ生成回数）/SC@9 の問題IDクラスタ差
を出す。GPU 不要。データセットは tasks.load_task で問題文→item_id を引く（HF キャッシュ利用）。

実行: uv run python scripts/analysis/audit_sc_from_logs.py
"""

from __future__ import annotations

import collections
import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evalx.debate import majority_vote  # noqa: E402
from src.evalx.tasks import extract_answer, is_correct, load_task  # noqa: E402

R2 = ROOT / "results/gcs/run002"
BASE = "Qwen/Qwen3-4B-Instruct-2507"
BENCH_TYPE = {"mmlu_pro": "letter", "math500": "math", "supergpqa": "letter"}


def question_index() -> dict:
    index = {}
    for bench in BENCH_TYPE:
        task = load_task(bench, n=10**6, seed=0)
        for item in task.items:
            index[item.question] = (bench, item.item_id)
    return index


def sc_per_item_paths(bench: str, seed: int) -> Path:
    if bench == "mmlu_pro" and seed == 1:
        return R2 / "recheck_c2s1/c2_sc9_mmlu_pro_s1_recheck.json"
    if seed <= 3:
        return R2 / f"remeasure_v1/r_c2_sc9_{bench}_s{seed}.json"
    return R2 / f"robust_c2/c2_sc9_{bench}_s{seed}.json"


def c7_path(bench: str, seed: int) -> Path:
    if seed == 1:
        return R2 / f"g3_team_check/g3_{bench}.json"
    if seed in (2, 3):
        return R2 / f"final_c7/c7_run002_team_{bench}_s{seed}.json"
    return R2 / f"robust_c7/c7_run002_team_{bench}_s{seed}.json"


def per_item(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    block = d.get("team") or next(iter((d.get("sc") or d.get("solo")).values()))
    return block["per_item"]


def main() -> None:
    qidx = question_index()
    samples = collections.defaultdict(dict)  # (bench, item, seed) -> {k: answer}
    for d in ("recheck_c2s1", "remeasure_v1", "robust_c2"):
        for f in glob.glob(str(R2 / d / "llm_calls" / "*.jsonl.gz")):
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec["model"] != BASE or rec["seed"] is None or rec["seed"] < 1000:
                        continue
                    if len(rec["messages"]) != 2 or not rec["messages"][0]["content"].startswith("Think step by step"):
                        continue
                    key = qidx.get(rec["messages"][1]["content"])
                    if key is None:
                        continue
                    bench, iid = key
                    seed, k = divmod(rec["seed"], 1000)
                    slot = samples[(bench, iid, seed)]
                    if k not in slot:  # 再実行による重複は最初の1件を採用
                        slot[k] = extract_answer(rec["response"], BENCH_TYPE[bench])

    # (1) 整合性: ログから再構成した SC@9 多数決 == 保存済み予測か
    agree = total = incomplete = 0
    acc = collections.defaultdict(lambda: collections.Counter())
    sc_correct = {}  # (bench, iid, seed, k) -> bool
    for bench in BENCH_TYPE:
        for seed in range(1, 7):
            stored = per_item(sc_per_item_paths(bench, seed))
            for iid, rec in stored.items():
                slot = samples.get((bench, iid, seed), {})
                if len(slot) < 9:
                    incomplete += 1
                    continue
                answers = [slot[k] for k in range(9)]
                total += 1
                agree += majority_vote(answers, seed) == rec["predicted"]
                for k in range(1, 10):
                    ok = is_correct(majority_vote(answers[:k], seed), rec["gold"], BENCH_TYPE[bench])
                    acc[(bench, k)]["n"] += 1
                    acc[(bench, k)]["c"] += ok
                    sc_correct[(bench, iid, seed, k)] = ok
    print(f"[整合性] ログ再集計のSC@9多数決が保存済み予測と一致: {agree}/{total} "
          f"({agree/max(total,1)*100:.2f}%), 9サンプル揃わず除外={incomplete}")

    # (2) SC@k の精度
    print(f"\n{'bench':10s} " + " ".join(f"SC@{k:<4d}" for k in range(1, 10)))
    for bench in BENCH_TYPE:
        print(f"{bench:10s} " + " ".join(f"{acc[(bench,k)]['c']/acc[(bench,k)]['n']:.3f} " for k in range(1, 10)))

    # (3) c7（6生成/問）vs SC@6 / SC@9（問題IDクラスタ、6シード）
    rng = np.random.default_rng(20260930)
    for k in (3, 6, 9):
        per = collections.defaultdict(list)
        for bench in BENCH_TYPE:
            for seed in range(1, 7):
                c7 = per_item(c7_path(bench, seed))
                for iid, rec in c7.items():
                    key = (bench, iid, seed, k)
                    if key in sc_correct:
                        per[f"{bench}:{iid}"].append(int(bool(rec["correct"])) - int(sc_correct[key]))
        d = np.array([np.mean(v) for v in per.values()])
        signs = rng.choice([-1.0, 1.0], size=(20000, len(d)))
        p = float((np.sum(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-15) + 1) / 20001)
        boot = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"c7(6生成) − SC@{k}: {d.mean()*100:+.2f}pt [{lo*100:+.2f},{hi*100:+.2f}] p={p:.4f} (items={len(d)})")


if __name__ == "__main__":
    main()
