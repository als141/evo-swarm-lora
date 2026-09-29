"""【解析8・副産物】旧環境 MMLU-Pro の「イメージ系統差 +6〜9pt」は回答抽出の旧バグで説明できるか。

発見の経緯（ra2b）: 旧環境→新環境（同一 seed・同一問題）でベースモデル系の精度差が
MMLU-Pro だけで +7〜9pt、MATH / SuperGPQA では ±1pt 程度。旧で誤答・新で正答に転じた
問題の旧予測がほぼ全て 'A' だった。

仮説: 旧イメージの extract_answer には「ANSWER: ANSWER: X」型の二重 prefix を剥がす処理が
無く（コミット 9b0fcd6 で追加）、letter 型で単語 ANSWER の 'A' を抽出していた。

検証: 新環境のベースモデル生成全文（llm_calls）に「旧抽出ロジック（剥がし処理なし）」と
現行ロジックを両方適用し、(a) 結果が変わる呼び出しの割合、(b) 旧ロジックで 'A' になる割合、
(c) 旧ロジックで採点した場合の精度低下量を、ベンチ・条件別に出す。
実行: python3 scripts/analysis/ra8_extraction_artifact.py
"""
import re
from collections import defaultdict


from ra_common import ANSWER_TYPE, OUT, dump_json, import_tasks_without_datasets, iter_raw_calls
from ra_common import classify_call, load_question_index, qhash, question_of_call

tasks = import_tasks_without_datasets()


def extract_old(text, answer_type):
    """9b0fcd6 以前の extract_answer（二重 prefix 剥がし無し）。"""
    candidates = tasks.ANSWER_LINE_PATTERN.findall(text)
    tail = candidates[-1].strip() if candidates else None
    if answer_type == "letter":
        if tail:
            m = re.search(r"[A-J]", tail.upper())
            if m:
                return m.group()
        for pattern in tasks.LETTER_FALLBACK_PATTERNS:
            ms = pattern.findall(text)
            if ms:
                return ms[-1].upper()
        return None
    return tasks.extract_answer(text, answer_type)


def main():
    qi = load_question_index()
    stats = defaultdict(lambda: {"n": 0, "changed": 0, "old_A": 0, "ok_new": 0, "ok_old": 0,
                                 "examples": []})
    for rec in iter_raw_calls():
        meta = classify_call(rec)
        if meta["kind"] not in ("solo", "sc", "team"):
            continue
        hit = qi.get(qhash(question_of_call(rec)))
        if not hit:
            continue
        bench, item_id, gold, _ = hit
        atype = ANSWER_TYPE[bench]
        if atype != "letter":
            continue
        key = f"{meta['family']}_{meta['kind']}" + (f"_r{meta['round']}" if meta["kind"] == "team" else "") + f"|{bench}"
        new = tasks.extract_answer(rec["response"], atype)
        old = extract_old(rec["response"], atype)
        s = stats[key]
        s["n"] += 1
        s["changed"] += int(new != old)
        s["old_A"] += int(old == "A" and new != "A")
        s["ok_new"] += int(tasks.is_correct(new, gold, atype))
        s["ok_old"] += int(tasks.is_correct(old, gold, atype))
        if new != old and len(s["examples"]) < 3:
            tail = rec["response"].strip().splitlines()[-1][:120]
            s["examples"].append({"new": new, "old": old, "gold": gold, "last_line": tail})
    out = {}
    for k, s in sorted(stats.items()):
        out[k] = {"n": s["n"], "changed_rate": s["changed"] / s["n"], "old_spurious_A_rate": s["old_A"] / s["n"],
                  "acc_new": s["ok_new"] / s["n"], "acc_old_logic": s["ok_old"] / s["n"],
                  "acc_drop_pt": 100 * (s["ok_new"] - s["ok_old"]) / s["n"], "examples": s["examples"]}
    dump_json(out, OUT / "ra8_extraction_artifact.json")
    print("== 新環境の生成全文に旧抽出ロジックを適用した場合（letter 型ベンチ）")
    for k, v in out.items():
        print(f"  {k:28s} n={v['n']:6d} 抽出変化={v['changed_rate']:.3f} 偽'A'={v['old_spurious_A_rate']:.3f} "
              f"精度 現行{v['acc_new']:.3f} → 旧ロジック{v['acc_old_logic']:.3f} ({-v['acc_drop_pt']:+.1f}pt)")
        for e in v["examples"][:2]:
            print(f"      例: new={e['new']} old={e['old']} gold={e['gold']} | {e['last_line']!r}")


if __name__ == "__main__":
    main()
