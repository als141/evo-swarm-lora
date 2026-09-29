"""問題文 → (bench, item_id, gold, メタ情報) の索引を HF キャッシュから構築する。

src/evalx/tasks.py の各ローダと完全に同じ問題文整形（選択肢付与）を再現し、
sha1(問題文) をキーにする。llm_calls の記録（プロンプト全文）を問題IDへ戻すのに使う。

実行: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python3 scripts/analysis/ra0_build_index.py
出力: results/reanalysis_2026-09/cache/question_index.json.gz
"""
import gzip
import json

from datasets import load_dataset

from ra_common import CACHE, QINDEX, import_tasks_without_datasets, qhash

tasks = import_tasks_without_datasets()


def main():
    index = {}

    mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    for row in mmlu:
        q = tasks._format_mc_question(row["question"], list(row["options"]))
        index[qhash(q)] = ["mmlu_pro", f"mmlupro-test-{row['question_id']}",
                           row["answer"].strip().upper(),
                           {"category": row["category"], "n_options": len(row["options"])}]

    math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for idx, row in enumerate(math):
        index[qhash(row["problem"])] = ["math500", f"math500-test-{idx}",
                                        tasks.normalize_math_answer(row["answer"]),
                                        {"level": int(row["level"]), "subject": row["subject"],
                                         "raw_answer": row["answer"]}]

    sg = load_dataset("m-a-p/SuperGPQA", split="train")
    for row in sg:
        q = tasks._format_mc_question(row["question"], list(row["options"]))
        index[qhash(q)] = ["supergpqa", f"supergpqa-{row['uuid']}",
                           row["answer_letter"].strip().upper(),
                           {"discipline": row["discipline"], "field": row["field"],
                            "difficulty": row["difficulty"],
                            "is_calculation": bool(row["is_calculation"]),
                            "n_options": len(row["options"])}]

    CACHE.mkdir(parents=True, exist_ok=True)
    with gzip.open(QINDEX, "wt") as fh:
        json.dump(index, fh, ensure_ascii=False)
    counts = {}
    for rec in index.values():
        counts[rec[0]] = counts.get(rec[0], 0) + 1
    print("indexed:", counts, "->", QINDEX)


if __name__ == "__main__":
    main()
