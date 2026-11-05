#!/usr/bin/env python3
"""Small streaming OpenAI-compatible load test for a local vLLM server."""
import argparse
import concurrent.futures
import datetime as dt
import json
import math
import os
import platform
import statistics
import hashlib
import glob
import subprocess
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit


SHORT = "Summarize why reliable measurements matter for local language model serving."
LONG = " ".join(
    f"Section {i}: A reliable service measures queueing, prompt processing, "
    "generation speed, resource use, and completion quality under repeatable conditions."
    for i in range(1, 41)
)


def stream_request(base_url, model, token, prompt, index):
    payload = json.dumps({
        "model": model, "messages": [{"role": "user", "content": f"{prompt}\nRequest {index}."}],
        "max_tokens": int(getattr(stream_request, "max_tokens", 32)), "temperature": 0, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", payload,
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    first = done = False
    usage = None
    try:
        host = urlsplit(base_url).hostname
        proxy_handler = (urllib.request.ProxyHandler({}) if host in {"127.0.0.1", "localhost", "::1"}
                         else urllib.request.ProxyHandler())
        opener = urllib.request.build_opener(proxy_handler)
        start = time.perf_counter()
        with opener.open(req, timeout=600) as response:
            status = response.status
            for raw in response:
                if not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                try:
                    event = json.loads(data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return {"ok": False, "status": status, "error": "malformed_sse"}
                if not isinstance(event, dict):
                    return {"ok": False, "status": status, "error": "malformed_sse"}
                if event.get("usage") is not None:
                    usage = event["usage"]
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content and not first:
                        first = time.perf_counter()
        end = time.perf_counter()
        if not done or usage is None or not first:
            return {"ok": False, "status": status, "error": "incomplete_stream"}
        if (not isinstance(usage, dict) or
                any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens"))):
            return {"ok": False, "status": status, "error": "invalid_usage"}
        return {"ok": True, "status": status, "ttft_seconds": first - start,
                "latency_seconds": end - start, "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"]}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "error": "http_error"}
    except Exception:
        return {"ok": False, "status": locals().get("status"), "error": "connection_or_protocol_error"}


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * p) - 1)]


def run_case(args, concurrency, workload):
    def call(i):
        return stream_request(args.base_url, args.model, args.token,
                              LONG if workload == "long" else SHORT, i)

    # Warm a full concurrency wave, then keep all warmup outside measurement.
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        warmups = list(pool.map(call, range(-concurrency, 0)))
    warmup_ok = all(r.get("ok") for r in warmups)
    if not warmup_ok:
        results = []
        elapsed = 0.0
    else:
        began = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(call, range(args.requests)))
        elapsed = time.perf_counter() - began
    good = [r for r in results if r["ok"]]
    return {
        "label": getattr(args, "label", "benchmark"), "repeat": getattr(args, "repeat", 1),
        "workload": workload, "concurrency": concurrency,
        "warmup": {"ok": warmup_ok, "count": len(warmups), "failed": sum(not x.get("ok") for x in warmups),
                   "errors": [{"status": x.get("status"), "error": x.get("error")} for x in warmups if not x.get("ok")]},
        "requests": args.requests, "successes": len(good),
        "errors": args.requests - len(good) if warmup_ok else args.requests, "wall_seconds": elapsed,
        "requests_per_second": len(good) / elapsed if elapsed else 0,
        "output_tokens_per_second": sum(r["output_tokens"] for r in good) / elapsed if elapsed else 0,
        "input_tokens": sum(r["input_tokens"] for r in good),
        "prompt_tokens_mean": statistics.mean(r["input_tokens"] for r in good) if good else None,
        "prompt_tokens_min": min((r["input_tokens"] for r in good), default=None),
        "prompt_tokens_max": max((r["input_tokens"] for r in good), default=None),
        "output_tokens": sum(r["output_tokens"] for r in good),
        "ttft_p50_seconds": percentile([r["ttft_seconds"] for r in good], .50),
        "ttft_p95_seconds": percentile([r["ttft_seconds"] for r in good], .95),
        "latency_p50_seconds": percentile([r["latency_seconds"] for r in good], .50),
        "latency_p95_seconds": percentile([r["latency_seconds"] for r in good], .95),
        "requests_detail": [{key: r.get(key) for key in (
            "ok", "status", "error", "ttft_seconds", "latency_seconds", "input_tokens", "output_tokens"
        )} for r in results] if warmup_ok else [],
    }


def markdown_table(results):
    lines = ["| Workload | Concurrency | Prompt tokens mean (range) | Success / errors | Req/s | Output tok/s | TTFT p50 / p95 (s) | Latency p50 / p95 (s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        fmt = lambda value: f"{value:.3f}" if value is not None else "n/a"
        mean = fmt(r['prompt_tokens_mean'])
        prompt_range = f"{r['prompt_tokens_min']}-{r['prompt_tokens_max']}" if r['prompt_tokens_min'] is not None else "n/a"
        lines.append(f"| {r['workload']} | {r['concurrency']} | {mean} ({prompt_range}) | {r['successes']} / {r['errors']} | {r['requests_per_second']:.3f} | "
                     f"{r['output_tokens_per_second']:.2f} | {fmt(r['ttft_p50_seconds'])} / {fmt(r['ttft_p95_seconds'])} | "
                     f"{fmt(r['latency_p50_seconds'])} / {fmt(r['latency_p95_seconds'])} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="qwen-cpu")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--label", default="benchmark")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--environment", help="JSON environment metadata file; only its SHA256 is recorded")
    parser.add_argument("--concurrency", default="1,2,4", help="comma-separated concurrency levels")
    parser.add_argument("--workload", choices=("short", "long", "both"), default="both")
    parser.add_argument("--output", default="bench/results.json")
    args = parser.parse_args()
    stream_request.max_tokens = args.max_tokens
    args.token = os.environ.get("API_TOKEN")
    if not args.token:
        parser.error("API_TOKEN environment variable is required")
    levels = [int(x) for x in args.concurrency.split(",")]
    if args.requests < 1 or args.repeats < 1 or args.max_tokens < 1 or any(x < 1 for x in levels):
        parser.error("requests, repeats, max-tokens, and concurrency levels must be positive")
    workloads = ("short", "long") if args.workload == "both" else (args.workload,)
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    files = ["app/*.py", "app/logging.json", "bench/run.py", "Dockerfile*", "requirements*"]
    hashes = {}
    for pattern in files:
        for path in glob.glob(pattern):
            if os.path.isfile(path):
                with open(path, "rb") as source:
                    hashes[path.replace("\\", "/")] = hashlib.sha256(source.read()).hexdigest()
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        git_head = None
    results = []
    def save_checkpoint():
        document = {"started_at": started_at, "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "base_url": args.base_url, "model": args.model, "requests_per_case": args.requests,
                    "workload": args.workload, "long_prompt_words": len(LONG.split()),
                    "max_tokens": args.max_tokens, "temperature": 0, "stream": True,
                    "repeats": args.repeats, "label": args.label,
                    "provenance": {"git_head": git_head, "sha256": hashes,
                                   "environment_sha256": hashlib.sha256(open(args.environment, "rb").read()).hexdigest() if args.environment else None},
                    "warmup_excluded": True, "client": {"python": platform.python_version(), "platform": platform.platform()},
                    "results": results, "markdown_table": markdown_table(results)}
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        temporary = args.output + ".tmp"
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2)
        os.replace(temporary, args.output)
        return document
    document = save_checkpoint()
    for workload in workloads:
        for c in levels:
            for repeat in range(1, args.repeats + 1):
                args.repeat = repeat
                case = run_case(args, c, workload)
                results.append(case)
                document = save_checkpoint()
                print(f"{args.label} {workload} c={c} repeat={repeat}: {case['successes']}/{case['requests']} ok, {case['wall_seconds']:.3f}s", flush=True)
    print(document["markdown_table"])
    if any(r["errors"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
