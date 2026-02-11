# rps_throughput_monitor.py
#
# Usage:
#   python rps_throughput_monitor.py \
#     --url http://127.0.0.1:8000 \
#     --rps 20 \
#     --duration 30 \
#     --max-requests 1000 \
#     --model Llama-2-7b-hf \
#     --model-prefix /models \
#     --verify-model \
#     --repeat 3 \
#     --csv outputs/throughput.csv
#
# Notes:
# - This file does NOT modify rps_chat_client.py
# - It reuses rps_chat_client.send_request()

import asyncio
import argparse
import time
import csv
from typing import Dict, Any, Optional, List

import aiohttp
import rps_chat_client  # reuse send_request from your original file


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


async def _monitor_loop(
    start_t: float,
    stop_evt: asyncio.Event,
    counters: Dict[str, int],
    latency_q: "asyncio.Queue[float]",
    csv_path: Optional[str],
):
    f = None
    writer = None
    if csv_path:
        f = open(csv_path, "w", newline="")
        writer = csv.writer(f)
        writer.writerow([
            "t_sec",
            "sent_total", "done_total", "ok_total", "err_total",
            "sent_rps", "done_rps",
            "inflight",
            "avg_latency", "p50_latency", "p95_latency",
        ])

    last_sent = counters["sent"]
    last_done = counters["done"]

    while not stop_evt.is_set():
        await asyncio.sleep(1.0)
        now = time.perf_counter()
        t_sec = now - start_t

        sent = counters["sent"]
        done = counters["done"]
        ok = counters["ok"]
        err = counters["err"]

        sent_rps = sent - last_sent
        done_rps = done - last_done
        inflight = sent - done

        # Drain latencies completed during this 1-second window
        lats: List[float] = []
        while True:
            try:
                lats.append(latency_q.get_nowait())
            except asyncio.QueueEmpty:
                break

        lats.sort()
        avg_lat = (sum(lats) / len(lats)) if lats else 0.0
        p50 = _percentile(lats, 0.50) if lats else 0.0
        p95 = _percentile(lats, 0.95) if lats else 0.0

        print(
            f"[t={t_sec:6.1f}s] sent_rps={sent_rps:4d} done_rps={done_rps:4d} "
            f"inflight={inflight:4d} ok={ok} err={err} "
            f"lat(avg/p50/p95)={avg_lat:.3f}/{p50:.3f}/{p95:.3f}"
        )

        if writer:
            writer.writerow([
                f"{t_sec:.1f}",
                sent, done, ok, err,
                sent_rps, done_rps,
                inflight,
                f"{avg_lat:.6f}", f"{p50:.6f}", f"{p95:.6f}",
            ])
            f.flush()

        last_sent, last_done = sent, done

    if f:
        f.close()


async def _run_with_monitor(args):
    if args.rps <= 0:
        raise ValueError("--rps must be > 0")

    interval = 1.0 / args.rps
    start_t = time.perf_counter()
    end_t = start_t + args.duration

    model = rps_chat_client.normalize_model(args.model, args.model_prefix)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": rps_chat_client.build_user_prompt(args.prompt, args.repeat)},
        ],
        "stream": False,
    }
    if args.max_tokens is not None:
        payload["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p

    counters = {"sent": 0, "done": 0, "ok": 0, "err": 0}
    latency_q: asyncio.Queue[float] = asyncio.Queue()

    stop_evt = asyncio.Event()
    monitor_task = asyncio.create_task(
        _monitor_loop(start_t, stop_evt, counters, latency_q, args.csv)
    )

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        if args.verify_model:
            ids = await rps_chat_client.fetch_models(session, args.url, args.timeout)
            if ids is not None:
                ok = model in ids
                print(f"[verify] model={model} exists={ok}")
                if not ok:
                    sample = ids[:20]
                    print(f"[verify] server models (first {len(sample)}): {sample}")

        tasks: List[asyncio.Task] = []
        idx = 0

        def _on_done(fut: "asyncio.Future"):
            # This callback runs when ONE request finishes -> update counters immediately
            try:
                ok, lat = fut.result()  # send_request returns (bool, latency)
            except Exception:
                counters["done"] += 1
                counters["err"] += 1
                return
            counters["done"] += 1
            if ok:
                counters["ok"] += 1
            else:
                counters["err"] += 1
            latency_q.put_nowait(lat)

        # schedule at RPS, and completions will be counted immediately via callback
        while True:
            now = time.perf_counter()
            if now >= end_t:
                break
            if args.max_requests is not None and idx >= args.max_requests:
                break

            counters["sent"] += 1
            t = asyncio.create_task(
                rps_chat_client.send_request(
                    session=session,
                    url=args.url,
                    payload=payload,
                    idx=idx,
                    timeout=args.timeout,
                )
            )
            t.add_done_callback(_on_done)
            tasks.append(t)

            idx += 1
            await asyncio.sleep(interval)

        # wait all inflight to finish so monitor can see final seconds correctly
        await asyncio.gather(*tasks, return_exceptions=True)

    # stop monitor
    stop_evt.set()
    await monitor_task

    print("\n===== FINAL =====")
    print(f"Model          : {model}")
    print(f"Sent           : {counters['sent']}")
    print(f"Done           : {counters['done']}")
    print(f"OK             : {counters['ok']}")
    print(f"Errors         : {counters['err']}")
    print(f"In-flight end  : {counters['sent'] - counters['done']}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Throughput monitor wrapper (reuses rps_chat_client.send_request)"
    )
    p.add_argument("--url", type=str, required=True)
    p.add_argument("--rps", type=float, default=5)
    p.add_argument("--duration", type=int, default=60)
    p.add_argument("--max-requests", type=int, default=None)
    p.add_argument("--model", type=str, default="Llama-2-7b-hf")
    p.add_argument(
        "--model-prefix",
        type=str,
        default="/models",
        help="If set, normalize model to '<prefix>/<model>' when model doesn't start with '/'. Set to '' to disable.",
    )
    p.add_argument("--prompt", type=str, default="What is your name?")
    p.add_argument("--repeat", type=int, default=1,
                   help="Repeat count for user prompt content in the request body")
    p.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="max_tokens for completion")
    p.add_argument("--temperature", type=float, default=None,
                   help="temperature")
    p.add_argument("--top-p", type=float, default=None,
                   help="top_p")
    p.add_argument("--verify-model", action="store_true",
                   help="Call GET /v1/models once and check whether the provided model exists.")
    p.add_argument("--csv", type=str, default="throughput.csv",
                   help="CSV output path (empty to disable)")
    args = p.parse_args()
    if args.csv == "":
        args.csv = None
    if args.model_prefix == "":
        args.model_prefix = None
    if args.repeat <= 0:
        raise ValueError("--repeat must be a positive integer")
    if args.max_requests is not None and args.max_requests <= 0:
        raise ValueError("--max-requests must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(_run_with_monitor(args))
