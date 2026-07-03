"""問題単位の並列実行ユーティリティ。

vLLM はサーバ側で連続バッチ処理を行うため、クライアントから同時に
複数リクエストを投げることで GPU 利用率が大きく向上する。
問題（item）同士は独立なので、item 単位で並列化する。
debate のラウンド内逐次性は各 item のタスク内で保たれる。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    items: List[T],
    fn: Callable[[T], R],
    max_workers: int = 16,
) -> Dict[int, R]:
    """items を並列に処理し、インデックス -> 結果 の辞書を返す。例外は伝播する。"""
    if max_workers <= 1:
        return {i: fn(item) for i, item in enumerate(items)}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {i: executor.submit(fn, item) for i, item in enumerate(items)}
        return {i: future.result() for i, future in futures.items()}
