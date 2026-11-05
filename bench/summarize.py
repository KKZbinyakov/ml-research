#!/usr/bin/env python3
"""Pool benchmark cases by label, workload, and concurrency without averaging rates."""
import argparse
import json
import math


def percentile(values, p):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)]


def summarize(documents):
    groups = {}
    for document in documents:
        for case in document.get("results", []):
            key = (case.get("label", document.get("label", "benchmark")), case["workload"], case["concurrency"])
            groups.setdefault(key, []).append(case)
    output = []
    for (label, workload, concurrency), cases in sorted(groups.items()):
        requests = sum(c["requests"] for c in cases)
        successes = sum(c["successes"] for c in cases)
        wall = sum(c["wall_seconds"] for c in cases)
        good = [r for c in cases for r in c.get("requests_detail", []) if r.get("ok")]
        repeat_rates = [c["successes"] / c["wall_seconds"] if c["wall_seconds"] else 0 for c in cases]
        repeat_token_rates = [c.get("output_tokens", 0) / c["wall_seconds"] if c["wall_seconds"] else 0 for c in cases]
        output.append({"label": label, "workload": workload, "concurrency": concurrency,
                       "repeats": len(cases), "requests": requests, "successes": successes,
                       "errors": requests - successes, "wall_seconds": wall,
                       "requests_per_second": successes / wall if wall else 0,
                       "output_tokens": sum(c.get("output_tokens", 0) for c in cases),
                       "output_tokens_per_second": sum(c.get("output_tokens", 0) for c in cases) / wall if wall else 0,
                       "ttft_p50_seconds": percentile([r["ttft_seconds"] for r in good if r.get("ttft_seconds") is not None], .5),
                       "ttft_p95_seconds": percentile([r["ttft_seconds"] for r in good if r.get("ttft_seconds") is not None], .95),
                       "latency_p50_seconds": percentile([r["latency_seconds"] for r in good if r.get("latency_seconds") is not None], .5),
                       "latency_p95_seconds": percentile([r["latency_seconds"] for r in good if r.get("latency_seconds") is not None], .95),
                       "repeat_throughput_min": min(repeat_rates, default=None),
                       "repeat_throughput_max": max(repeat_rates, default=None),
                       "repeat_output_tokens_per_second_min": min(repeat_token_rates, default=None),
                       "repeat_output_tokens_per_second_max": max(repeat_token_rates, default=None)})
    return output


def markdown(rows):
    lines = ["| Label | Workload | Concurrency | Repeats | Success / total | Pooled req/s | Pooled output tok/s | Repeat req/s range | Repeat tok/s range | TTFT p50 / p95 | Latency p50 / p95 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        f = lambda v: f"{v:.3f}" if v is not None else "n/a"
        rate_range = f"{f(r['repeat_throughput_min'])}-{f(r['repeat_throughput_max'])}"
        token_range = f"{f(r['repeat_output_tokens_per_second_min'])}-{f(r['repeat_output_tokens_per_second_max'])}"
        lines.append(f"| {r['label']} | {r['workload']} | {r['concurrency']} | {r['repeats']} | {r['successes']} / {r['requests']} | {f(r['requests_per_second'])} | {f(r['output_tokens_per_second'])} | {rate_range} | {token_range} | {f(r['ttft_p50_seconds'])} / {f(r['ttft_p95_seconds'])} | {f(r['latency_p50_seconds'])} / {f(r['latency_p95_seconds'])} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--json", default="bench/summary.json")
    parser.add_argument("--markdown", default="bench/summary.md")
    args = parser.parse_args()
    documents = []
    for path in args.files:
        with open(path, encoding="utf-8") as f:
            documents.append(json.load(f))
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(summarize(documents), f, indent=2)
    with open(args.json, encoding="utf-8") as f:
        table = markdown(json.load(f))
    with open(args.markdown, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
