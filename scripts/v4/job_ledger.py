"""Vertex カスタムジョブの稼働時間から費用を推定する台帳（課金エクスポートが反映されるまでのつなぎ）。

GPU ジョブは実効単価（既定 $2.2/h、2026-07 の実測で Spot の約2倍）で保守的に見積もる。
実行中のジョブは現在時刻までの時間で数える。CPU の試験ジョブは $0.2/h とする。
使い方: python3 scripts/v4/job_ledger.py [--rate 2.2] [--jpy 159.4]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess

REGIONS = ("us-central1", "europe-west4", "asia-southeast1")
PROJECT = "pro-plasma-510112-m7"


def parse(ts: str | None):
    if not ts:
        return None
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=2.2, help="GPU ジョブの実効単価（ドル/時）")
    parser.add_argument("--jpy", type=float, default=159.4, help="円/ドル（クレジット ¥47,813 = $300 から）")
    args = parser.parse_args()
    now = dt.datetime.now(dt.timezone.utc)
    rows = []
    for region in REGIONS:
        out = subprocess.run(
            ["gcloud", "ai", "custom-jobs", "list", "--region", region, "--project", PROJECT, "--format=json"],
            capture_output=True, text=True, timeout=120).stdout
        for job in json.loads(out or "[]"):
            start, end = parse(job.get("startTime")), parse(job.get("endTime"))
            if start is None:
                continue
            hours = ((end or now) - start).total_seconds() / 3600
            spec = job["jobSpec"]["workerPoolSpecs"][0]["machineSpec"]
            gpu = spec.get("acceleratorType", "")
            rate = args.rate if "A100" in gpu else (0.4 if gpu else 0.2)
            rows.append((job["displayName"], region, job["state"].replace("JOB_STATE_", ""), hours, rate * hours))
    rows.sort(key=lambda r: -r[4])
    total = sum(r[4] for r in rows)
    for name, region, state, hours, cost in rows:
        print(f"{name:45s} {region:16s} {state:10s} {hours:6.2f} h  ${cost:6.2f}")
    print(f"\n推定合計 ${total:.2f}（¥{total * args.jpy:,.0f}） / 上限 $300（¥47,813）  残り推定 ${300 - total:.2f}")


if __name__ == "__main__":
    main()
