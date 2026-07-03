"""vLLM OpenAI 互換 API へのチャットクライアント。

vLLM Multi-LoRA では `--lora-modules persona_a=/path` で登録したアダプタ名を
model フィールドに指定して切り替える。ベースモデルはベースモデル名を指定。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    seed: Optional[int] = None


class ChatClient:
    def __init__(self, base_url: str, api_key: str = "EMPTY", max_retries: int = 3, retry_wait: float = 5.0):
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=180.0)
        self._max_retries = max_retries
        self._retry_wait = retry_wait

    def chat(self, model: str, messages: List[dict], config: GenerationConfig) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_tokens=config.max_tokens,
                    seed=config.seed,
                )
                content = response.choices[0].message.content
                return content or ""
            except Exception as error:  # noqa: BLE001 - リトライ対象を広く取る
                last_error = error
                time.sleep(self._retry_wait * (attempt + 1))
        raise RuntimeError(f"Chat completion failed after {self._max_retries} retries: {last_error}")

    def list_models(self) -> List[str]:
        return [m.id for m in self._client.models.list().data]
