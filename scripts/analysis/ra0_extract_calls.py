"""llm_calls 完全記録（run002, 新環境）を 1 呼び出し 1 行の索引に展開する。

各呼び出しについて:
  - 種別（solo/sc/team/judge）・系列（c7=r2_* / c5=ev_* / base）・実行seed・エージェント添字・ラウンド
  - 問題ID・ベンチ・正解（ra0_build_index.py の索引で問題文から逆引き）
  - 入力/出力トークン数（Qwen3-4B 実トークナイザ、チャットテンプレート適用後の入力長）
  - 抽出回答と正誤（全文 / 先頭 512・1024・2048・4096 トークンで打ち切った場合）
    ※ max_tokens は生成分布を変えず出力を切るだけなので、全文の先頭 N トークンで
      抽出し直すことは「max_tokens=N で生成した場合」の厳密なシミュレーションになる
  - ts / elapsed（スループット推定用）、logprob confidence

実行（トークナイザだけの一時環境。プロジェクトの .venv には触れない）:
  uv run --no-project --with tokenizers==0.21.1 --with numpy \
      python scripts/analysis/ra0_extract_calls.py
出力: results/reanalysis_2026-09/cache/calls_index.jsonl.gz
"""
import glob
import gzip
import json
import time

from tokenizers import Tokenizer

from ra_common import (CALLS, ANSWER_TYPE, classify_call, import_tasks_without_datasets,
                       iter_raw_calls, load_question_index, qhash, question_of_call)

tasks = import_tasks_without_datasets()
TRUNC = (512, 1024, 2048, 4096)


def chat_prompt(messages):
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def main():
    tok_path = glob.glob("/home/als0028/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/"
                         "snapshots/*/tokenizer.json")[0]
    tok = Tokenizer.from_file(tok_path)
    qindex = load_question_index()

    t0 = time.time()
    n_written = n_unmapped = 0
    buf = []

    def flush(fh):
        nonlocal n_written
        prompts = [chat_prompt(r["messages"]) for r in buf]
        responses = [r["response"] for r in buf]
        in_enc = tok.encode_batch(prompts, add_special_tokens=False)
        out_enc = tok.encode_batch(responses, add_special_tokens=False)
        for rec, ienc, oenc in zip(buf, in_enc, out_enc):
            meta = classify_call(rec)
            q = question_of_call(rec)
            hit = qindex.get(qhash(q))
            row = {
                "dir": rec["_dir"], "file": rec["_file"], "line": rec["_line"],
                "model": rec["model"], "ts": rec["ts"], "elapsed": rec["elapsed"],
                "max_tokens": rec.get("max_tokens"), "temperature": rec.get("temperature"),
                "seed": rec.get("seed"), **meta,
                "conf_mean": rec.get("mean_confidence"), "conf_tail": rec.get("tail_confidence"),
                "in_tok": len(ienc.ids), "out_tok": len(oenc.ids), "out_chars": len(rec["response"]),
            }
            if hit is None:
                row.update(bench=None, item_id=None, gold=None)
            else:
                bench, item_id, gold, _extra = hit
                row.update(bench=bench, item_id=item_id, gold=gold)
                if meta["kind"] in ("solo", "sc", "team"):
                    atype = ANSWER_TYPE[bench]
                    ans = tasks.extract_answer(rec["response"], atype)
                    row["ans"] = ans
                    row["ok"] = tasks.is_correct(ans, gold, atype)
                    ids = oenc.ids
                    for n in TRUNC:
                        if len(ids) > n:
                            text_n = tok.decode(ids[:n], skip_special_tokens=False)
                            a_n = tasks.extract_answer(text_n, atype)
                        else:
                            a_n = ans
                        row[f"ans{n}"] = a_n
                        row[f"ok{n}"] = tasks.is_correct(a_n, gold, atype)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
        buf.clear()

    with gzip.open(CALLS, "wt") as fh:
        for rec in iter_raw_calls():
            buf.append(rec)
            if len(buf) >= 2000:
                flush(fh)
                print(f"  {n_written:,} calls  ({time.time() - t0:.0f}s)", flush=True)
        if buf:
            flush(fh)
    # 未対応問題の件数を確認
    with gzip.open(CALLS, "rt") as fh:
        for line in fh:
            if json.loads(line)["item_id"] is None:
                n_unmapped += 1
    print(f"done: {n_written:,} calls, unmapped={n_unmapped}, {time.time() - t0:.0f}s -> {CALLS}")


if __name__ == "__main__":
    main()
