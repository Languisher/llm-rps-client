# rps_chat_client.py
# Usage:
# Pre-Define LLM_SERVER_URL, e.g.
# http://127.0.0.1:8000
# uv run rps_chat_client.py \
#   --url $LLM_SERVER_URL \
#   --rps 1 \
#   --duration 10 \
#   --max-requests 20 \
#   --model Llama-2-7b-hf \
#   --model-prefix /models \
#   --prompt "What is your name?" \
#   --repeat 3

import asyncio
import aiohttp
import argparse
import time
import json
from typing import Dict, Any, Optional, Tuple


def normalize_model(model: str, prefix: Optional[str]) -> str:
    """
    Normalize model name to match servers that expect '/models/<name>'.
    If model is already absolute-like (starts with '/' or contains prefix), keep it.
    """
    if not prefix:
        return model
    if model.startswith("/"):
        return model
    # common: prefix="/models" -> "/models/<model>"
    prefix = prefix.rstrip("/")
    return f"{prefix}/{model}"


def build_user_prompt(prompt: str, repeat: int) -> str:
    return prompt * repeat


async def fetch_models(session: aiohttp.ClientSession, base_url: str, timeout: int) -> Optional[list]:
    try:
        async with session.get(
            f"{base_url}/v1/models",
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            txt = await resp.text()
            if resp.status // 100 != 2:
                print(f"[models] status={resp.status} body={txt[:500]}")
                return None
            data = json.loads(txt)
            # OpenAI format: {"data":[{"id":"..."}]}
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                return [m.get("id") for m in data["data"] if isinstance(m, dict)]
            return None
    except Exception as e:
        print(f"[models] ERROR: {e}")
        return None


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    idx: int,
    timeout: int,
) -> Tuple[bool, float]:
    start = time.perf_counter()
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            text = await resp.text()
            latency = time.perf_counter() - start

            if resp.status // 100 == 2:
                # Try parse JSON
                try:
                    data = json.loads(text)
                except Exception:
                    print(f"[{idx}] status={resp.status} latency={latency:.3f}s (non-json) body={text[:300]!r}")
                    return True, latency

                choice0 = (data.get("choices") or [{}])[0]
                msg = choice0.get("message") or {}
                content = msg.get("content")
                finish = choice0.get("finish_reason")
                usage = data.get("usage")

                content_repr = repr(content)

                is_empty = (content is None) or (isinstance(content, str) and len(content) == 0)
                if is_empty:
                    print(f"[{idx}] status=200 latency={latency:.3f}s EMPTY content={content_repr} finish={finish} usage={usage} msg={msg}")
                    print(f"[{idx}] raw={text[:800]}{' ...<truncated>' if len(text) > 800 else ''}")
                else:
                    print(f"[{idx}] status=200 latency={latency:.3f}s answer={content[:60]!r}")

                return True, latency
            else:
                preview = text if len(text) <= 800 else text[:800] + " ...<truncated>"
                print(f"[{idx}] status={resp.status} latency={latency:.3f}s body={preview}")
                return False, latency

    except Exception as e:
        latency = time.perf_counter() - start
        print(f"[{idx}] ERROR after {latency:.3f}s: {e}")
        return False, latency


async def run(args):
    if args.rps <= 0:
        raise ValueError("--rps must be > 0")

    interval = 1.0 / args.rps
    start_t = time.perf_counter()
    end_t = start_t + args.duration

    model = normalize_model(args.model, args.model_prefix)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": build_user_prompt(args.prompt, args.repeat)},
        ],
        # Make behavior explicit (many servers are picky)
        "stream": False,
    }

    # Optional generation params
    if args.max_tokens is not None:
        payload["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Optional: verify the model exists on server
        if args.verify_model:
            ids = await fetch_models(session, args.url, args.timeout)
            if ids is not None:
                ok = model in ids
                print(f"[verify] model={model} exists={ok}")
                if not ok:
                    # also show a few candidates
                    sample = ids[:20]
                    print(f"[verify] server models (first {len(sample)}): {sample}")

        tasks = []
        idx = 0
        max_requests: Optional[int] = args.max_requests

        while True:
            now = time.perf_counter()
            if now >= end_t:
                break
            if max_requests is not None and idx >= max_requests:
                break

            tasks.append(
                asyncio.create_task(
                    send_request(session, args.url, payload, idx, args.timeout)
                )
            )
            idx += 1
            await asyncio.sleep(interval)

        results = await asyncio.gather(*tasks)

    success = sum(1 for ok, _ in results if ok)
    total = len(results)
    avg_latency = sum(lat for _, lat in results) / total if total else 0.0

    print("\n===== SUMMARY =====")
    print(f"Model          : {model}")
    print(f"Total requests : {total}")
    print(f"Success        : {success}")
    print(f"Failure        : {total - success}")
    print(f"Avg latency    : {avg_latency:.3f}s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="RPS-controlled OpenAI-compatible LLM client"
    )
    parser.add_argument("--url", type=str, required=True,
                        help="Base URL of the LLM server (e.g. http://127.0.0.1:8000)")
    parser.add_argument("--rps", type=float, default=5,
                        help="Requests per second (must be > 0)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Total duration in seconds")
    parser.add_argument("--max-requests", type=int, default=None,
                        help="Maximum number of requests to send (default: unlimited)")
    parser.add_argument("--model", type=str, default="Llama-2-7b-hf",
                        help="Model name or full id. E.g. 'Llama-2-7b-hf' or '/models/Llama-2-7b-hf'")
    parser.add_argument("--model-prefix", type=str, default="/models",
                        help="If set, will normalize model to '<prefix>/<model>' when model doesn't start with '/'. "
                             "Set to '' to disable.")
    parser.add_argument("--prompt", type=str, default="What is your name?",
                        help="User prompt")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat count for user prompt content in the request body")
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.",
                        help="System prompt")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Request timeout (seconds)")

    # Optional generation controls
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="max_tokens for completion")
    parser.add_argument("--temperature", type=float, default=None,
                        help="temperature")
    parser.add_argument("--top-p", type=float, default=None,
                        help="top_p")

    parser.add_argument("--verify-model", action="store_true",
                        help="Call GET /v1/models once and check whether the provided model exists.")

    args = parser.parse_args()
    if args.max_requests is not None and args.max_requests <= 0:
        raise ValueError("--max-requests must be a positive integer, or omit it")
    if args.repeat <= 0:
        raise ValueError("--repeat must be a positive integer")
    if args.model_prefix == "":
        args.model_prefix = None
    return args


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))
