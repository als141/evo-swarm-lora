"""vLLM（OpenAI 互換 API）へのストリーミング生成クライアント。

- 生成をストリーミングで受け、完全一致の反復ループを検知したら接続を切って打ち切る。
  2026-07 の実測では max_tokens(8192) 到達の呼び出しは全体の 2〜6% だが出力トークンの
  10〜22% を消費し、正答率はほぼ 0 だった（大半が反復ループ）。
- 生成トークンごとの logprob から確信度（幾何平均確率）を計算する。
- 1 呼び出しの結果は辞書で返し、そのまま CallStore に保存できる形にする。
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI


@dataclass(frozen=True)
class GenConfig:
    """Qwen3-4B-Instruct-2507 の公式推奨（temp 0.7 / top_p 0.8 / top_k 20）。"""

    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_tokens: int = 8192

    def as_dict(self) -> dict:
        return {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k,
                "max_tokens": self.max_tokens}


def find_loop(text: str, probe: int = 64, min_repeats: int = 5, min_span: int = 1500,
              window: int = 12000) -> Optional[int]:
    """末尾が同一文字列の完全反復（周期 p を min_repeats 回以上、合計 min_span 文字以上）なら p を返す。

    完全一致の周期性だけを見るため、正常な推論文を誤検知しない保守的な判定である。
    """
    tail = text[-window:]
    n = len(tail)
    if n < max(min_span, probe * 2):
        return None
    key = tail[-probe:]
    prev = tail.rfind(key, 0, n - 1)
    if prev == -1:
        return None
    period = (n - probe) - prev
    if period <= 0:
        return None
    repeats_needed = max(min_repeats, math.ceil(min_span / period))
    span = period * repeats_needed
    if span > n:
        return None
    # 末尾 span 文字が周期 period を持つか（1 周期ずらした文字列と一致するか）
    if tail[n - span + period:] == tail[n - span:n - period]:
        return period
    return None


class LLMClient:
    def __init__(self, base_url: str, api_key: str = "EMPTY", max_retries: int = 4,
                 loop_check_every: int = 400, loop_detection: bool = True):
        timeout = float(os.environ.get("EVO4_HTTP_TIMEOUT", "900"))
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        self._max_retries = max_retries
        self._loop_check_every = loop_check_every
        self._loop_detection = loop_detection

    def generate(self, model: str, messages: List[dict], config: GenConfig, seed: int) -> dict:
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                return self._generate_once(model, messages, config, seed)
            except Exception as error:  # noqa: BLE001 - サーバ一時エラーは再試行
                last_error = error
                if getattr(error, "status_code", None) == 400:
                    break  # 入力不正（コンテキスト超過など）は再試行しても同じ
                time.sleep(min(60.0, 5.0 * (2 ** attempt)))
        raise RuntimeError(f"generation failed after {self._max_retries} attempts: {last_error}")

    def _generate_once(self, model: str, messages: List[dict], config: GenConfig, seed: int) -> dict:
        started = time.time()
        stream = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            seed=seed,
            stream=True,
            logprobs=True,
            stream_options={"include_usage": True},
            extra_body={"top_k": config.top_k},
        )
        pieces: List[str] = []
        logprobs: List[float] = []
        finish = None
        prompt_tokens = None
        completion_tokens = None
        loop_period = None
        chars_since_check = 0
        try:
            for chunk in stream:
                if chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta.content if choice.delta else None
                if delta:
                    pieces.append(delta)
                    chars_since_check += len(delta)
                lp = getattr(choice, "logprobs", None)
                if lp is not None and lp.content:
                    logprobs.extend(t.logprob for t in lp.content if t.logprob is not None)
                if choice.finish_reason is not None:
                    finish = choice.finish_reason
                if self._loop_detection and chars_since_check >= self._loop_check_every:
                    chars_since_check = 0
                    period = find_loop("".join(pieces))
                    if period is not None:
                        loop_period = period
                        finish = "loop_abort"
                        break
        finally:
            stream.close()  # ループ打ち切り時は接続を切り、vLLM 側の生成も中断させる
        text = "".join(pieces)
        n_out = completion_tokens if completion_tokens is not None else len(logprobs)
        mean_lp = sum(logprobs) / len(logprobs) if logprobs else None
        tail = logprobs[-64:]
        return {
            "text": text,
            "finish": finish,
            "n_out": n_out,
            "n_in": prompt_tokens,
            "mean_logprob": mean_lp,
            "tail_logprob": (sum(tail) / len(tail)) if tail else None,
            "loop_period": loop_period,
            "elapsed": round(time.time() - started, 3),
            "t_start": round(started, 3),
        }

    def list_models(self) -> List[str]:
        return [m.id for m in self._client.models.list().data]
