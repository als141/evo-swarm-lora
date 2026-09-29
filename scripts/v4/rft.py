"""対照: 単一モデルの自己学習（RFT, rejection-sampling fine-tuning）。

ペルソナも議論も使わず、社会と同じ学習用問題（train_g0..g{G-1}）でベースモデルを k 回サンプリングし、
正解した出力だけで 1 つの LoRA を学習する（例数の上限は社会の1ペルソナが G 世代で使う総数と同じ）。
学習後、test で単独（SC のサンプル）を生成し、単独精度と SC@k をオフラインで求める。
「社会の進化は、単に同じデータで自己学習した場合と比べて何を足すのか」を測るための対照。
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "vendor"))

from src.evo4.items import index_items, load_items  # noqa: E402
from src.evo4.llm import GenConfig, LLMClient  # noqa: E402
from src.evo4.prompts import r0_messages  # noqa: E402
from src.evo4.scoring import extract_answer, is_correct  # noqa: E402
from src.evo4.server import BASE_MODEL, VllmServer  # noqa: E402
from src.evo4.society import Agent, Society, stable_seed  # noqa: E402
from src.evo4.store import CallStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--adapters", required=True)
    parser.add_argument("--train-chunks", default="0,1,2")
    parser.add_argument("--k-train", type=int, default=3)
    parser.add_argument("--max-examples", type=int, default=3000)
    parser.add_argument("--k-test", type=int, default=9)
    parser.add_argument("--test-seeds", default="1")
    parser.add_argument("--workers", type=int, default=192)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chunks = [str(ROOT / f"data/v4/items/train_g{c}.jsonl") for c in args.train_chunks.split(",")]
    test_path = str(ROOT / "data/v4/items/test.jsonl")
    items = index_items(chunks + [test_path])
    train_ids = [i.item_id for c in chunks for i in load_items(c)]
    test_ids = [i.item_id for i in load_items(test_path)]
    store = CallStore(tempfile.mkdtemp(prefix="evo4_store_"), args.store, sync_every=60)
    server = VllmServer(max_loras=4, max_lora_rank=16)
    config = GenConfig()
    base = Agent("base", "base", BASE_MODEL, "")
    adir = Path(args.adapters) / "rft.base"

    if not (adir / "train_info.json").exists():
        server.start()
        soc = Society(LLMClient(server.base_url), store, items, config, workers=args.workers)
        soc.run_tasks(soc.sc_tasks(base, train_ids, gen_seed=1, k=args.k_train), label="rft_sample")
        examples = []
        for item_id in train_ids:
            item = items[item_id]
            for k in range(args.k_train):
                rec = store.get(soc.sc_key(base, item_id, 1, k))
                if rec and rec.get("finish") == "stop" and is_correct(
                        extract_answer(rec["text"], item.answer_type), item.gold, item.answer_type):
                    examples.append(r0_messages("", item.question, item.answer_type)
                                    + [{"role": "assistant", "content": rec["text"]}])
                    break  # 1 問 1 例（社会の r0 と同じ扱い）
        random.Random(0).shuffle(examples)
        examples = examples[:args.max_examples]
        server.stop()
        ex_path = Path("/tmp/rft.examples.jsonl")
        ex_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in examples) + "\n")
        subprocess.run([sys.executable, str(ROOT / "scripts/v4/train_child.py"), "--examples", str(ex_path),
                        "--out", str(adir), "--seed", str(stable_seed("rft.base")), "--lr", "1e-4"], check=True)

    server.start()
    server.load_lora("rft_base", str(adir))
    rft = Agent("rft.base", "base", "rft_base", "")
    soc = Society(LLMClient(server.base_url), store, items, config, workers=args.workers)
    for seed in (int(s) for s in args.test_seeds.split(",")):
        soc.run_tasks(soc.sc_tasks(rft, test_ids, gen_seed=seed, k=args.k_test), label=f"rft_test_g{seed}")
    store.sync()
    server.stop()
    (out / "rft_done.json").write_text(json.dumps({"adapter": str(adir), "train_chunks": args.train_chunks}))


if __name__ == "__main__":
    main()
