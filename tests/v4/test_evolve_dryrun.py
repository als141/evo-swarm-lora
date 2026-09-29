"""進化ループ（evolve.py）の空回し試験: 偽 vLLM ＋ ダミー学習で 3 系統×2 世代を回す。GPU 不要。"""

import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evo4.items import Item, save_items  # noqa: E402


def main() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "tests.v4.fake_vllm:app", "--port", str(port),
                               "--log-level", "warning"], cwd=ROOT)
    try:
        time.sleep(3)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            def mk(prefix, n):
                items = []
                for i in range(n):
                    bench = ("mmlu_pro", "supergpqa", "math")[i % 3]
                    at = "math" if bench == "math" else "letter"
                    items.append(Item(f"{prefix}{i}", bench, f"{prefix} question {i}\nA. a\nB. b", "A" if at == "letter" else "2", at))
                return items
            save_items(mk("dev", 12), tmp / "dev.jsonl")
            save_items(mk("test", 9), tmp / "test.jsonl")
            for k in range(3):
                save_items(mk(f"tr{k}_", 30), tmp / f"train_g{k}.jsonl")
            common = ["--generations", "2", "--store", str(tmp / "store"), "--adapters", str(tmp / "adapters"),
                      "--dev", str(tmp / "dev.jsonl"), "--test", str(tmp / "test.jsonl"), "--train-dir", str(tmp),
                      "--workers", "8", "--external-base-url", f"http://127.0.0.1:{port}/v1", "--fake-train"]
            for lineage, select in (("S", "shapley"), ("N", "none"), ("A1", "solo")):
                subprocess.run([sys.executable, str(ROOT / "scripts/v4/evolve.py"), "--lineage", lineage,
                                "--select", select, "--out", str(tmp / lineage)] + common, check=True)
                state = json.loads((tmp / lineage / "state.json").read_text())
                reps = {t: [state["generations"][t]["reps"][r]["agent_id"] for r in ("critic", "pragmatist", "explorer")]
                        for t in state["generations"]}
                print(lineage, "reps:", reps)
                print(lineage, "test:", {t: round(g.get("test_team", {}).get("macro", -1), 3) for t, g in state["generations"].items()})
            # 共有世代（g1）の子は全系統で同じディレクトリ（一度だけ学習）
            shared = sorted(p.name for p in (tmp / "adapters").iterdir() if p.name.startswith("shared.g1"))
            print("shared g1 adapters:", shared)
            assert len([x for x in shared if x.endswith((".a", ".b", ".c"))]) == 9
            print("DRYRUN OK")
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
