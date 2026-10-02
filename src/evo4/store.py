"""LLM 呼び出し結果の永続ストア（一意キー → 結果）。

- 同じ呼び出し（キー）は二度生成しない。Spot のプリエンプト後も保存済みの呼び出しから再開できる。
- 書き込みはローカルの JSONL シャードに追記し、一定間隔でリモート（GCS FUSE マウント）へ複製する。
  GCS FUSE 上のファイルへの追記は毎回オブジェクト全体を書き直すため、直接は追記しない。
- 起動時はリモートとローカルの全シャードを読み込む。キー重複は既定で先勝ち（シャード名の順）、
  オフライン解析では resolve="earliest"（生成開始時刻 t_start が最も早い記録）を使う。
  感度分析には resolve="latest"（最も遅い記録＝生成し直した記録）を使う。
- 読み込み中に他のジョブがシャードを置き換えると読み込みが失敗しうる（2026-10 に系統 N・A1 が
  系統 S のシャードを黙って読み飛ばし、同じ呼び出しを生成し直した）。失敗は再試行し、最後は警告を出す。
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import socket
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Optional


class CallStore:
    def __init__(self, local_dir: str, remote_dir: Optional[str] = None, sync_every: float = 120.0,
                 resolve: str = "first"):
        self._local_dir = Path(local_dir)
        self._local_dir.mkdir(parents=True, exist_ok=True)
        self._remote_dir = Path(remote_dir) if remote_dir else None
        if self._remote_dir is not None:
            self._remote_dir.mkdir(parents=True, exist_ok=True)
        if resolve not in ("first", "earliest", "latest"):
            raise ValueError(resolve)
        self._resolve = resolve
        self.duplicates = 0  # 同じキーの異なる記録の余分な数（解析で報告する。解決規則に依らない）
        self._seen = set()  # 読み込んだ記録の (key, t_start)。同じ記録の再読み込みを数えないため
        self._records: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._sync_every = sync_every
        self._last_sync = time.time()
        self._dirty = False
        self._load()
        stamp = f"{socket.gethostname()}_{os.getpid()}_{int(time.time())}"
        self._shard = self._local_dir / f"calls_{stamp}.jsonl"

    def _shards(self) -> Iterable[Path]:
        seen = set()
        for directory in (self._remote_dir, self._local_dir):
            if directory is None or not directory.exists():
                continue
            for path in sorted(directory.glob("calls_*.jsonl*")):
                if path.name in seen:
                    continue
                seen.add(path.name)
                yield path

    def _load(self) -> None:
        for path in self._shards():
            opener = gzip.open if path.suffix == ".gz" else open
            for attempt in range(4):
                try:
                    with opener(path, "rt", encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError:
                                continue  # 書き込み途中で切れた末尾行
                            self._add_loaded(record)
                    break
                except OSError as exc:  # 他のジョブがシャードを置き換えた途中など
                    if attempt == 3:
                        print(f"[store] WARNING: shard skipped after retries: {path.name}: {exc}", flush=True)
                    else:
                        time.sleep(5 * (attempt + 1))

    def _add_loaded(self, record: dict) -> None:
        key = record["key"]
        t_start = record.get("t_start")
        ident = (key, t_start, record.get("text") if t_start is None else None)
        if ident in self._seen:
            return  # 同じ記録の再読み込み（再試行時など）
        self._seen.add(ident)
        current = self._records.get(key)
        if current is None:
            self._records[key] = record
            return
        self.duplicates += 1
        if self._resolve == "earliest" and (record.get("t_start") or float("inf")) < (current.get("t_start") or float("inf")):
            self._records[key] = record
        elif self._resolve == "latest" and (record.get("t_start") or float("-inf")) > (current.get("t_start") or float("-inf")):
            self._records[key] = record

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, key: str) -> bool:
        return key in self._records

    def get(self, key: str) -> Optional[dict]:
        return self._records.get(key)

    def put(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            if record["key"] in self._records:
                return
            self._records[record["key"]] = record
            with self._shard.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self._dirty = True
            if time.time() - self._last_sync > self._sync_every:
                self._sync_locked()

    def _sync_locked(self) -> None:
        self._last_sync = time.time()
        if self._remote_dir is None or not self._dirty or not self._shard.exists():
            return
        tmp = self._remote_dir / (self._shard.name + ".tmp")
        shutil.copyfile(self._shard, tmp)
        os.replace(tmp, self._remote_dir / self._shard.name)
        self._dirty = False

    def sync(self) -> None:
        with self._lock:
            self._sync_locked()

    def records(self) -> Iterable[dict]:
        return list(self._records.values())
