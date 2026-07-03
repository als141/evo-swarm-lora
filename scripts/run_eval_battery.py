"""複数の評価を 1 つの vLLM サーバに対して順次実行するバッテリードライバ。

A100 ジョブの起動コスト（イメージ pull + モデルロード）を節約するため、
最終評価の全条件を 1 ジョブにまとめる。各エントリは run_eval.py を
サブプロセスとして呼び出す（進捗キャッシュにより再実行時はスキップ）。

設定 JSON の例:
[
  {"name": "base_solo_mmlupro_s1", "args": ["--task", "mmlu_pro", "--n", "500",
   "--seed", "1", "--mode", "solo",
   "--agents", "base=Qwen/Qwen3-4B-Instruct-2507"]},
  ...
]

使用例:
  python3 scripts/run_eval_battery.py \
    --base-url http://localhost:8000/v1 \
    --config cloud/eval_battery.json \
    --out-dir /gcs/<bucket>/experiments/run001/final_eval
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_EVAL = ROOT / "scripts" / "run_eval.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--config", required=True, help="評価エントリの JSON リスト")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    entries = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for entry in entries:
        name = entry["name"]
        out_path = out_dir / f"{name}.json"
        if out_path.exists():
            print(f"[battery] skip {name} (already complete)")
            summary.append({"name": name, "status": "skipped"})
            continue

        cmd = [
            sys.executable,
            str(RUN_EVAL),
            "--base-url", args.base_url,
            "--out", str(out_path),
            "--progress-dir", str(out_dir / "progress" / name),
            *entry["args"],
        ]
        print(f"[battery] running {name}: {' '.join(cmd)}")
        started = time.time()
        proc = subprocess.run(cmd, cwd=str(ROOT))
        elapsed = time.time() - started
        status = "ok" if proc.returncode == 0 else f"failed({proc.returncode})"
        print(f"[battery] {name}: {status} in {elapsed:.0f}s")
        summary.append({"name": name, "status": status, "elapsed_seconds": elapsed})

        (out_dir / "battery_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if proc.returncode != 0 and args.stop_on_error:
            sys.exit(proc.returncode)

    failures = [s for s in summary if s["status"].startswith("failed")]
    print(f"[battery] done: {len(summary)} entries, {len(failures)} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
