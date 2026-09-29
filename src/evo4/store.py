"""LLM 呼び出し結果の永続ストア（一意キー → 結果）。

- 同じ呼び出し（キー）は二度生成しない。Spot のプリエンプト後も保存済みの呼び出しから再開できる。
- 書き込みはローカルの JSONL シャードに追記し、一定間隔でリモート（GCS FUSE マウント）へ複製する。
  GCS FUSE 上のファイルへの追記は毎回オブジェクト全体を書き直すため、直接は追記しない。
- 起動時はリモートとローカルの全シャードを読み込む（キー重複は先勝ち）。
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
    def __init__(self, local_dir: str, remote_dir: Optional[str] = None, sync_every: float = 120.0):
        self._local_dir = Path(local_dir)
        self._local_dir.mkdir(parents=True, exist_ok=True)
        self._remote_dir = Path(remote_dir) if remote_dir else None
        if self._remote_dir is not None:
            self._remote_dir.mkdir(parents=True, exist_ok=True)
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
                        self._records.setdefault(record["key"], record)
            except OSError:
                continue

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
