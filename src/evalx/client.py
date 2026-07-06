"""vLLM OpenAI 互換 API へのチャットクライアント。

vLLM Multi-LoRA では `--lora-modules persona_a=/path` で登録したアダプタ名を
model フィールドに指定して切り替える。ベースモデルはベースモデル名を指定。
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from openai import OpenAI


@dataclass
class GenerationConfig:
    """Qwen3-4B-Instruct-2507 の公式推奨値に準拠 (temp 0.7 / top_p 0.8 / top_k 20)。"""

    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_tokens: int = 4096
    seed: Optional[int] = None


@dataclass
class ChatResult:
    """生成テキストと logprob 由来の確信度。

    verbalized confidence は 4B 級で 80-100% に飽和し較正不能（arXiv:2502.11028）なため、
    トークン logprob から計算する（DeepConf arXiv:2508.15260 / CISC arXiv:2502.06233 系）。
    - mean_confidence: 全トークンの幾何平均確率 exp(mean(logprob))
    - tail_confidence: 末尾 window トークンの幾何平均確率（結論部の確信）
    logprobs が取得できなかった場合はどちらも None。
    """

    text: str
    mean_confidence: Optional[float] = None
    tail_confidence: Optional[float] = None


class _CallLogger:
    """全LLM呼び出しの入力コンテキストと出力を JSONL へ完全記録する。

    環境変数 EVALX_LOG_DIR が設定されている場合のみ有効。ファイルは
    プロセスごと（calls_<pid>_<起動秒>.jsonl）に分かれ、battery 実行では
    エントリ=サブプロセス単位で自然に分割される。追記方式は ProgressCache と
    同じで gcsfuse 上での動作実績あり。研究データの一次資産として、
    per_item の要約では失われる生のプロンプト・生成全文を保全する。
    """

    def __init__(self):
        self._path: Optional[Path] = None
        self._lock = threading.Lock()
        log_dir = os.environ.get("EVALX_LOG_DIR")
        if log_dir:
            directory = Path(log_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self._path = directory / f"calls_{os.getpid()}_{int(time.time())}.jsonl"

    def log(self, record: dict) -> None:
        if self._path is None:
            return
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


_CALL_LOGGER = _CallLogger()


class ChatClient:
    def __init__(self, base_url: str, api_key: str = "EMPTY", max_retries: int = 3, retry_wait: float = 5.0):
        # 高並列時は最難問（8192トークン級）の1リクエストが180秒を超えうる
        # （run001 v3で末尾数問がタイムアウト→エントリ失敗を実測）。既定を余裕側に取る。
        timeout = float(os.environ.get("EVALX_HTTP_TIMEOUT", "600"))
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._max_retries = max_retries
        self._retry_wait = retry_wait

    def chat(self, model: str, messages: List[dict], config: GenerationConfig) -> str:
        return self.chat_scored(model, messages, config, with_logprobs=False).text

    def chat_scored(
        self,
        model: str,
        messages: List[dict],
        config: GenerationConfig,
        with_logprobs: bool = True,
        tail_window: int = 64,
    ) -> ChatResult:
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                kwargs = {}
                if with_logprobs:
                    kwargs["logprobs"] = True
                started = time.time()
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_tokens=config.max_tokens,
                    seed=config.seed,
                    # top_k は OpenAI API 標準外のため vLLM 拡張として渡す
                    extra_body={"top_k": config.top_k},
                    **kwargs,
                )
                choice = response.choices[0]
                content = choice.message.content or ""
                mean_conf = tail_conf = None
                logprobs = getattr(choice, "logprobs", None)
                tokens = getattr(logprobs, "content", None) if logprobs else None
                if tokens:
                    lps = [t.logprob for t in tokens if t.logprob is not None]
                    if lps:
                        mean_conf = math.exp(sum(lps) / len(lps))
                        tail = lps[-tail_window:]
                        tail_conf = math.exp(sum(tail) / len(tail))
                _CALL_LOGGER.log(
                    {
                        "ts": round(started, 3),
                        "elapsed": round(time.time() - started, 3),
                        "model": model,
                        "messages": messages,
                        "response": content,
                        "mean_confidence": mean_conf,
                        "tail_confidence": tail_conf,
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens,
                        "seed": config.seed,
                    }
                )
                return ChatResult(text=content, mean_confidence=mean_conf, tail_confidence=tail_conf)
            except Exception as error:  # noqa: BLE001 - リトライ対象を広く取る
                last_error = error
                time.sleep(self._retry_wait * (attempt + 1))
        raise RuntimeError(f"Chat completion failed after {self._max_retries} retries: {last_error}")

    def list_models(self) -> List[str]:
        return [m.id for m in self._client.models.list().data]
