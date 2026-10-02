"""CallStore の重複キーの解決規則（first / earliest / latest）と重複数の数え方の確認。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evo4.store import CallStore  # noqa: E402


def _write(path: Path, records: list) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_resolve() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        remote = Path(tmp) / "remote"
        remote.mkdir()
        # シャード名の順（a → b）と生成開始時刻の順（b が早い）を逆にしておく
        _write(remote / "calls_a.jsonl", [{"key": "k1", "t_start": 200.0, "text": "late"},
                                          {"key": "k2", "t_start": 10.0, "text": "only"}])
        _write(remote / "calls_b.jsonl", [{"key": "k1", "t_start": 100.0, "text": "early"},
                                          {"key": "k1", "t_start": 100.0, "text": "early"}])  # 同じ記録の再読み込み
        expect = {"first": "late", "earliest": "early", "latest": "late"}
        for rule, text in expect.items():
            store = CallStore(str(Path(tmp) / f"local_{rule}"), str(remote), resolve=rule)
            assert store.get("k1")["text"] == text, (rule, store.get("k1"))
            assert store.get("k2")["text"] == "only"
            assert store.duplicates == 1, (rule, store.duplicates)  # 同じ記録の再読み込みは数えない
        try:
            CallStore(str(Path(tmp) / "local_bad"), str(remote), resolve="random")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown resolve rule must raise")


if __name__ == "__main__":
    test_resolve()
    print("STORE RESOLVE OK")
