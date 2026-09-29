"""evo4 の結合試験用の偽 vLLM（OpenAI 互換・ストリーミング対応）サーバ。

- model 名と seed から決定的に応答を作る。"loop" を含む質問には反復ループを返す。
- 応答の最終行は "ANSWER: <X>"。
"""

from __future__ import annotations

import hashlib
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()


def _answer(model: str, question: str, seed: int) -> str:
    if "same" in model:  # ゲーティング試験用: 全員が同じ答えを返す
        h = int(hashlib.sha256(question.encode()).hexdigest(), 16)
        return "ABCD"[h % 4]
    h = int(hashlib.sha256(f"{model}|{question}|{seed}".encode()).hexdigest(), 16)
    return "ABCD"[h % 4]


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    model = body["model"]
    messages = body["messages"]
    seed = body.get("seed", 0)
    question = messages[1]["content"] if len(messages) > 1 else ""
    if "loop" in question and len(messages) == 2:
        text = "Let me think.\n" + ("The same line repeats forever and ever. " * 400)
    else:
        text = f"Reasoning by {model} (seed {seed}).\nStep 1. Step 2.\nANSWER: {_answer(model, question, seed)}"
    tokens = [text[i:i + 8] for i in range(0, len(text), 8)]

    def gen():
        for i, tok in enumerate(tokens):
            chunk = {"id": "x", "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                     "choices": [{"index": 0, "delta": {"content": tok},
                                  "logprobs": {"content": [{"token": tok, "logprob": -0.1, "bytes": None, "top_logprobs": []}]},
                                  "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"
        final = {"id": "x", "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                 "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(final)}\n\n"
        usage = {"id": "x", "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                 "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": len(tokens), "total_tokens": 10 + len(tokens)}}
        yield f"data: {json.dumps(usage)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"ok": True}
