# =====================
# Usage
# python rps_chat_client.py \
#   --url http://127.0.0.1:8000 \
#   --rps 20 \
#   --duration 120 \
#   --max-requests 1000 \
#   --model facebook/opt-1.3b \
#   --prompt "What is your name?"
# =====================

import asyncio
import aiohttp
import argparse
import time
from typing import Dict, Any, Optional


# =====================
# Request
# =====================
async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    idx: int,
    timeout: int,
):
    start = time.perf_counter()
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            await resp.read()  # consume body (non-stream)
            latency = time.perf_counter() - start
            print(f"[{idx}] status={resp.status} latency={latency:.3f}s")
            return True, latency
    except Exception as e:
        latency = time.perf_counter() - start
        print(f"[{idx}] ERROR after {latency:.3f}s: {e}")
        return False, latency


# =====================
# Runner
# =====================
async def run(args):
    if args.rps <= 0:
        raise ValueError("--rps must be > 0")

    interval = 1.0 / args.rps
    start_t = time.perf_counter()
    end_t = start_t + args.duration

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        ],
    }

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        idx = 0

        # Stop condition: time OR max_requests
        max_requests: Optional[int] = args.max_requests
        while True:
            now = time.perf_counter()
            if now >= end_t:
                break
            if max_requests is not None and idx >= max_requests:
                break

            tasks.append(
                asyncio.create_task(
                    send_request(
                        session,
                        args.url,
                        payload,
                        idx,
                        args.timeout,
                    )
                )
            )
            idx += 1
            await asyncio.sleep(interval)

        results = await asyncio.gather(*tasks)

    success = sum(1 for ok, _ in results if ok)
    total = len(results)
    avg_latency = sum(lat for _, lat in results) / total if total else 0

    print("\n===== SUMMARY =====")
    print(f"Total requests : {total}")
    print(f"Success        : {success}")
    print(f"Failure        : {total - success}")
    print(f"Avg latency    : {avg_latency:.3f}s")


# =====================
# CLI
# =====================
def parse_args():
    parser = argparse.ArgumentParser(
        description="RPS-controlled OpenAI-compatible LLM client"
    )

    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Base URL of the LLM server (e.g. http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=5,
        help="Requests per second (must be > 0)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Total duration in seconds",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Maximum number of requests to send (default: unlimited)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-1.3b",
        help="Model name",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is your name?",
        help="User prompt",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout (seconds)",
    )

    args = parser.parse_args()
    if args.max_requests is not None and args.max_requests <= 0:
        raise ValueError("--max-requests must be a positive integer, or omit it")
    return args


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))