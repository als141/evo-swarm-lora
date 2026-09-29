"""v4 の問題集合（dev / test / train チャンク）を構築し、JSONL に固定する（事前登録用）。

- test: 2026-07 に一度も使っていない新規問題（再解析の test_pool_fresh から）。
    MMLU-Pro 500 / SuperGPQA 500 / MATH（test 全体の Level 4-5 から MATH-500 収録分を除く）300
- dev: 2026-07 に使用済みで難易度が既知の問題から層化抽出（選抜・プロトコル選択に使う）。
    難易度 p_base はベースモデルの新環境の生出力を v4 採点器で採点し直して求める
    （2026-07 の MATH 採点器は約 6pt の偽陰性があったため）。
    MMLU-Pro 150 / SuperGPQA 150 / MATH-500 100。各ベンチとも 0<p<1 の問題を 70%、p=1 を 30%。
- train_g{k}: 社会学習の学習データ生成に使う問題（dev・test・2026-07 使用済みと重複しない）。
    各チャンク MMLU-Pro 400 / SuperGPQA 400 / MATH train split（Level 3-5）400。

使い方: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run ... python scripts/v4/build_items.py
"""

from __future__ import annotations

import collections
import gzip
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset  # noqa: E402

from src.evo4.items import Item, format_mc_question, save_items  # noqa: E402
from src.evo4.scoring import extract_answer, is_correct, last_boxed  # noqa: E402

POOLS = ROOT / "results/reanalysis_2026-09/pools"
INDEX = ROOT / "results/reanalysis_2026-09/cache/calls_index.jsonl.gz"
RAW = ROOT / "results/gcs/run002"
OUT = ROOT / "data/v4/items"
SEED = 20260929
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]
N_TRAIN_CHUNKS = 6


def load_sources():
    mmlu = {f"mmlupro-test-{r['question_id']}": r for r in load_dataset("TIGER-Lab/MMLU-Pro", split="test")}
    sgpqa = {f"supergpqa-{r['uuid']}": r for r in load_dataset("m-a-p/SuperGPQA", split="train")}
    math500 = {f"math500-test-{i}": r for i, r in enumerate(load_dataset("HuggingFaceH4/MATH-500", split="test"))}
    math_test, math_train = {}, {}
    for subject in MATH_SUBJECTS:
        for idx, row in enumerate(load_dataset("EleutherAI/hendrycks_math", subject, split="test")):
            math_test[f"mathfull-test-{subject}-{idx}"] = row
        for idx, row in enumerate(load_dataset("EleutherAI/hendrycks_math", subject, split="train")):
            math_train[f"mathtrain-{subject}-{idx}"] = row
    return mmlu, sgpqa, math500, math_test, math_train


def mmlu_item(item_id, row):
    return Item(item_id, "mmlu_pro", format_mc_question(row["question"], list(row["options"])),
                row["answer"].strip().upper(), "letter", {"category": row["category"]})


def sgpqa_item(item_id, row):
    return Item(item_id, "supergpqa", format_mc_question(row["question"], list(row["options"])),
                row["answer_letter"].strip().upper(), "letter",
                {"discipline": row["discipline"], "difficulty": row["difficulty"]})


def math_item(item_id, problem, gold, meta):
    return Item(item_id, "math", problem, gold.strip(), "math", meta)


def base_difficulty_v4(math500) -> dict:
    """新環境のベース（solo と SC）の生出力を v4 採点器で採点し、問題ごとの正答率を返す。"""
    raw_gold = {k: r["answer"] for k, r in math500.items()}
    by_file = collections.defaultdict(list)
    for line in gzip.open(INDEX, "rt"):
        row = json.loads(line)
        if row.get("family") == "base" and row.get("kind") in ("sc", "solo"):
            by_file[(row["dir"], row["file"])].append(row)
    hits = collections.defaultdict(list)
    for (dirname, filename), rows in by_file.items():
        lines = gzip.open(RAW / dirname / "llm_calls" / filename, "rt").read().splitlines()
        for row in rows:
            text = json.loads(lines[row["line"]])["response"]
            bench = row["bench"]
            answer_type = "math" if bench == "math500" else "letter"
            gold = raw_gold[row["item_id"]] if bench == "math500" else row["gold"]
            hits[row["item_id"]].append(is_correct(extract_answer(text, answer_type), gold, answer_type))
    return {k: (sum(v) / len(v), len(v)) for k, v in hits.items()}


def stratified_dev(ids, p_base, n, rng):
    informative = sorted(i for i in ids if i in p_base and 0.0 < p_base[i][0] < 1.0)
    always = sorted(i for i in ids if i in p_base and p_base[i][0] == 1.0)
    n_inf = min(len(informative), round(n * 0.7))
    chosen = rng.sample(informative, n_inf) + rng.sample(always, n - n_inf)
    return sorted(chosen)


def main() -> None:
    rng = random.Random(SEED)
    mmlu, sgpqa, math500, math_test, math_train = load_sources()
    fresh = json.loads((POOLS / "test_pool_fresh.json").read_text())
    used = set(json.loads((POOLS / "used_items.json").read_text())["used_item_ids"])
    manifest = {"seed": SEED, "files": {}}

    # ---------------- test（新規問題のみ）
    test_ids = {
        "mmlu_pro": fresh["mmlu_pro"]["sample_1000"][:500],
        "supergpqa": fresh["supergpqa"]["sample_1000"][:500],
        "math": [e["item_id"] for e in fresh["math_full_l45"]["sample_1000"][:300]],
    }
    raw_answers = {e["item_id"]: e["raw_answer"] for e in fresh["math_full_l45"]["sample_1000"]}
    test_items = [mmlu_item(i, mmlu[i]) for i in test_ids["mmlu_pro"]]
    test_items += [sgpqa_item(i, sgpqa[i]) for i in test_ids["supergpqa"]]
    for i in test_ids["math"]:
        row = math_test[i]
        gold = last_boxed(row["solution"])
        assert gold.strip() == raw_answers[i].strip(), (i, gold, raw_answers[i])  # 再解析と同じ問題か
        assert row["level"] in ("Level 4", "Level 5"), (i, row["level"])
        test_items.append(math_item(i, row["problem"], gold, {"level": row["level"], "type": row["type"]}))
    assert all(i.item_id not in used for i in test_items)
    assert all(i.gold for i in test_items)

    # ---------------- dev（使用済み・難易度既知から層化）
    p_base = base_difficulty_v4(math500)
    dev_pool = json.loads((POOLS / "dev_pool_known_difficulty.json").read_text())
    dev_ids = {
        "mmlu_pro": stratified_dev([r["item_id"] for r in dev_pool["mmlu_pro"]], p_base, 150, rng),
        "supergpqa": stratified_dev([r["item_id"] for r in dev_pool["supergpqa"]], p_base, 150, rng),
        "math500": stratified_dev([r["item_id"] for r in dev_pool["math500"]], p_base, 100, rng),
    }
    dev_items = [mmlu_item(i, mmlu[i]) for i in dev_ids["mmlu_pro"]]
    dev_items += [sgpqa_item(i, sgpqa[i]) for i in dev_ids["supergpqa"]]
    for i in dev_ids["math500"]:
        row = math500[i]
        dev_items.append(math_item(i, row["problem"], row["answer"],
                                   {"level": row["level"], "subject": row["subject"]}))
    for item in dev_items:
        item.meta["p_base_v4"] = round(p_base[item.item_id][0], 4)
        item.meta["n_base_samples"] = p_base[item.item_id][1]

    # ---------------- train チャンク（dev・test・使用済みと重複しない）
    excluded = used | {i.item_id for i in test_items} | set(fresh["mmlu_pro"]["sample_1000"]) \
        | set(fresh["supergpqa"]["sample_1000"]) | set(raw_answers)
    mmlu_rest = sorted(k for k in mmlu if k not in excluded)
    sgpqa_rest = sorted(k for k in sgpqa if k not in excluded)
    math_train_ok = sorted(
        k for k, r in math_train.items()
        if r["level"] in ("Level 3", "Level 4", "Level 5") and last_boxed(r["solution"])
    )
    rng.shuffle(mmlu_rest)
    rng.shuffle(sgpqa_rest)
    rng.shuffle(math_train_ok)
    per = 400
    train_chunks = []
    for k in range(N_TRAIN_CHUNKS):
        chunk = [mmlu_item(i, mmlu[i]) for i in mmlu_rest[k * per:(k + 1) * per]]
        chunk += [sgpqa_item(i, sgpqa[i]) for i in sgpqa_rest[k * per:(k + 1) * per]]
        for i in math_train_ok[k * per:(k + 1) * per]:
            row = math_train[i]
            chunk.append(math_item(i, row["problem"], last_boxed(row["solution"]),
                                   {"level": row["level"], "type": row["type"]}))
        train_chunks.append(chunk)

    all_ids = [i.item_id for i in test_items] + [i.item_id for i in dev_items]
    for chunk in train_chunks:
        all_ids += [i.item_id for i in chunk]
    assert len(all_ids) == len(set(all_ids)), "dev/test/train overlap"

    manifest["files"]["test.jsonl"] = {"sha256": save_items(test_items, OUT / "test.jsonl"),
                                       "counts": dict(collections.Counter(i.bench for i in test_items))}
    manifest["files"]["dev.jsonl"] = {"sha256": save_items(dev_items, OUT / "dev.jsonl"),
                                      "counts": dict(collections.Counter(i.bench for i in dev_items))}
    for k, chunk in enumerate(train_chunks):
        name = f"train_g{k}.jsonl"
        manifest["files"][name] = {"sha256": save_items(chunk, OUT / name),
                                   "counts": dict(collections.Counter(i.bench for i in chunk))}
    dev_band = collections.Counter(
        (i.bench, "p=1" if i.meta["p_base_v4"] == 1.0 else "0<p<1") for i in dev_items)
    manifest["dev_bands"] = {f"{b}|{band}": n for (b, band), n in sorted(dev_band.items())}
    manifest["dev_mean_p_base_v4"] = {
        b: round(sum(i.meta["p_base_v4"] for i in dev_items if i.bench == b)
                 / max(1, sum(1 for i in dev_items if i.bench == b)), 4)
        for b in ("mmlu_pro", "supergpqa", "math")
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
