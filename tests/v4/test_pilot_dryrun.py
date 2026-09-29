"""パイロット（pilot.py）の空回し試験: 偽 vLLM で全段を通す。GPU 不要。"""

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
            items = []
            for i in range(60):
                bench = ("mmlu_pro", "supergpqa", "math")[i % 3]
                at = "math" if bench == "math" else "letter"
                items.append(Item(f"d{i}", bench, f"dev question {i}\nA. a\nB. b", "A" if at == "letter" else "2", at))
            save_items(items, tmp / "dev.jsonl")
            pool = json.loads((ROOT / "configs/v4/persona_pool.json").read_text())
            (tmp / "pool.json").write_text(json.dumps(pool))
            subprocess.run([sys.executable, str(ROOT / "scripts/v4/pilot.py"), "--items", str(tmp / "dev.jsonl"),
                            "--pool", str(tmp / "pool.json"), "--out", str(tmp / "out"), "--adapters", str(tmp),
                            "--sweep-workers", "4,8", "--sweep-items", "10", "--sweep-k", "2",
                            "--n-lora", "5", "--workers", "16", "--external-base-url", f"http://127.0.0.1:{port}/v1"],
                           check=True)
            summary = json.loads((tmp / "out" / "pilot_summary.json").read_text())
            for key in ("sweep", "protocol_acc_devA", "protocol_chosen", "stage1", "stage2", "stage3"):
                print(key, "=>", json.dumps(summary[key], ensure_ascii=False)[:600])
            print("selected:", (tmp / "out" / "personas_selected.json").read_text()[:300])
            print("PILOT DRYRUN OK")
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
