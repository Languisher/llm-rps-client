# llm-rps-client

A minimal, RPS-controlled client and throughput monitor for **OpenAI-compatible LLM servers**.

This repo is designed for **systems experiments**, not model evaluation:
- controllable arrival rate (RPS)
- precise completion-based throughput
- per-second monitoring (steady-state analysis)
- mock server for isolation testing

---

## Repository Structure

```
.
├── rps_chat_client.py          # Core RPS-controlled client (DO NOT modify)
├── rps_throughput_monitor.py   # Throughput / latency monitor (wrapper)
├── tests/
│   └── mock_server.py          # Mock OpenAI-compatible server
└── README.md
```

Design principle:

> Keep the core client simple and reusable; put all experimental logic in wrappers.

---

## Environment Setup (using `uv`)

This repo assumes you are using **`uv`** for Python environment management.

### 1. Create virtual environment

```
uv venv  
source .venv/bin/activate
```

### 2. Install dependencies

```
uv pip install aiohttp fastapi uvicorn
```

(Only minimal runtime dependencies are required.)

---

## Mock Server (for isolated testing)

Before testing a real LLM server, it is strongly recommended to validate the client
against a **mock server** with fixed latency.

### Start mock server

```
python tests/mock_server.py --port 8000 --base-delay 0.1
```

This starts an OpenAI-compatible endpoint:

```
POST /v1/chat/completions
```

Behavior:
- fixed ~100ms response time
- deterministic JSON output
- no GPU / model dependency

This allows you to test:
- RPS scheduling correctness
- client-side throughput accounting
- backlog / inflight behavior

---

## Core Client: `rps_chat_client.py`

This file implements a **pure RPS-controlled request generator**.

Important:  
You should treat this file as a *library*, not an experiment script.

### Example usage

```
python rps_chat_client.py \
  --url http://127.0.0.1:8000 \
  --rps 20 \
  --duration 30 \
  --max-requests 1000
```

What it does:
- sends requests at a fixed arrival rate
- waits for all requests to complete
- prints only global summary statistics

What it intentionally does NOT do:
- no per-second monitoring
- no throughput curve
- no steady-state detection

---

## Throughput Monitor (Recommended): `rps_throughput_monitor.py`

This is the **main experimental entrypoint**.

It reuses `send_request()` from `rps_chat_client.py` and adds:
- per-second throughput measurement
- inflight (backlog) tracking
- per-second latency statistics
- CSV export for plotting

### Example usage

```
python rps_throughput_monitor.py \
  --url http://127.0.0.1:8000 \
  --rps 20 \
  --duration 30 \
  --max-requests 1000 \
  --csv throughput.csv
```

---

## Output Metrics Explained

Each CSV row corresponds to **one second**:

```
t_sec,  
sent_total, done_total, ok_total, err_total,  
sent_rps, done_rps,  
inflight,  
avg_latency, p50_latency, p95_latency
```

Key definitions:

- sent_rps  
  Requests issued in this second (arrival rate)

- done_rps  
  Requests completed in this second  
  → This is the throughput

- inflight = sent_total - done_total  
  Current backlog / in-flight requests

- latency statistics  
  Computed only from requests completed within this second

---

## How to Measure Throughput Correctly

Do NOT compute throughput as:

total_requests / total_duration

This mixes:
- warmup phase
- drain phase
- scheduling artifacts

### Correct method

1. Identify a steady-state window where:
   - sent_rps ≈ done_rps
   - inflight is stable (not increasing)
   - latency does not trend upward

2. Compute:

throughput ≈ mean(done_rps over steady-state window)

Example (from real output):

```
t=27: sent_rps=20, done_rps=19, inflight=3  
t=28: sent_rps=19, done_rps=20, inflight=2  
t=29: sent_rps=20, done_rps=20, inflight=2  
t=30: sent_rps=19, done_rps=20, inflight=1  
```

→ Throughput ≈ 20 requests/sec

This indicates:
- stable system
- no queue explosion
- sustainable load

---

## Interpretation Guidelines (Systems View)

- sent_rps > done_rps persistently  
  → system overloaded, backlog growing

- done_rps ≈ sent_rps, stable inflight  
  → steady state reached

- rising p95_latency over time  
  → hidden queueing or admission pressure

- final second with sent_rps = 0  
  → drain phase (do not include in throughput)

---

## Why This Design

This repo explicitly separates concerns:

- Client correctness (RPS scheduling)
- System behavior observation (throughput, backlog, latency)

This mirrors real production systems:
- traffic generator ≠ metrics pipeline
- arrival rate ≠ service rate

---

## Next Extensions (Optional)

Natural next steps if you continue this work:

- --max-inflight admission control
- streaming (stream=true) + TTFT / tokens/sec
- Poisson / burst arrival models
- automatic steady-state detection
- plotting scripts (matplotlib)

