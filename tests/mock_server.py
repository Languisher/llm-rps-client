# =====================
# To use this mock server, run the following command in the terminal:
# pip install fastapi uvicorn
# Example usage:
# python tests/mock_server.py --port 8000 --base-delay 0.1
# Slightly more complex example with jitter and concurrency limit:
# python tests/mock_server.py --port 8000 --base-delay 0.1 --jitter 0.05 --jitter-mode uniform --seed 1
# python tests/mock_server.py --port 8000 --base-delay 0.1 --max-concurrency 16
# =====================

import argparse
import asyncio
import json
import random
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Response, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn


# ---------------------
# Request schema
# ---------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


# ---------------------
# App factory
# ---------------------
def create_app(
    base_delay: float,
    jitter: float,
    jitter_mode: str,
    max_concurrency: int,
) -> FastAPI:
    app = FastAPI()

    # concurrency gate
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None

    def sample_delay() -> float:
        if jitter <= 0:
            return max(0.0, base_delay)

        if jitter_mode == "uniform":
            d = base_delay + random.uniform(-jitter, jitter)
        elif jitter_mode == "normal":
            # normal with std=jitter, clamp to >=0
            d = base_delay + random.gauss(0.0, jitter)
        else:
            d = base_delay  # fallback
        return max(0.0, d)

    async def acquire_or_429() -> Optional[asyncio.Semaphore]:
        if semaphore is None:
            return None
        # try-acquire without waiting: if overloaded -> 429
        if semaphore.locked():
            # locked() isn't perfect for semaphore fullness, so we try with timeout=0
            pass
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=0.0)
            return semaphore
        except Exception:
            return None

    def build_non_stream_response(req: ChatCompletionRequest) -> Dict[str, Any]:
        return {
            "id": "mock-chatcmpl-001",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! I am a mock server 🙂",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    async def stream_sse(req: ChatCompletionRequest):
        """
        Minimal OpenAI-like streaming:
        yields a few chunks then [DONE]
        """
        # First chunk: role
        first = {
            "id": "mock-chatcmpl-001",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(first)}\n\n"

        # Some content chunks
        text = "Hello! I am a mock server 🙂"
        for ch in [text[:8], text[8:16], text[16:]]:
            chunk = {
                "id": "mock-chatcmpl-001",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {"content": ch}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0)  # allow scheduling

        # Final chunk
        final = {
            "id": "mock-chatcmpl-001",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        sem = await acquire_or_429()
        if semaphore is not None and sem is None:
            return JSONResponse(
                status_code=429,
                content={"error": {"message": "Too many in-flight requests", "type": "overloaded"}},
            )

        try:
            await asyncio.sleep(sample_delay())

            if req.stream:
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                }
                return StreamingResponse(stream_sse(req), headers=headers)

            return JSONResponse(content=build_non_stream_response(req))

        finally:
            if semaphore is not None and sem is not None:
                sem.release()

    return app


# ---------------------
# CLI
# ---------------------
def parse_args():
    p = argparse.ArgumentParser("Mock OpenAI-compatible chat server")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--base-delay", type=float, default=0.1, help="Base delay in seconds")
    p.add_argument("--jitter", type=float, default=0.0, help="Jitter amount in seconds")
    p.add_argument(
        "--jitter-mode",
        type=str,
        default="uniform",
        choices=["uniform", "normal"],
        help="How to sample jitter",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=0,
        help="Max in-flight requests. 0 = unlimited",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed (0 = no seed)")
    p.add_argument("--log-level", type=str, default="info")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed != 0:
        random.seed(args.seed)

    app = create_app(
        base_delay=args.base_delay,
        jitter=args.jitter,
        jitter_mode=args.jitter_mode,
        max_concurrency=args.max_concurrency,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)