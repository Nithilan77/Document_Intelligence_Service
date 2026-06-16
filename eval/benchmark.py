"""Latency benchmark (Phase 8).

Measures two things the README reports as headline numbers:

1. Retrieval latency per mode (dense / sparse / hybrid) -- pure retrieval,
   no LLM. Run many times; report mean / p50 / p95.
2. Answer latency cold vs warm -- cache miss (retrieve + LLM) vs cache hit
   (Redis lookup). Demonstrates the cache payoff.

Retrieval is measured in-process (no HTTP) to isolate the retriever. Answer
latency is measured via the live API so it includes the real request path.
Usage:
  python benchmark.py retrieval         # mode latency, no API/LLM needed
  python benchmark.py answer            # cold vs warm, needs API + GEMINI key
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

QUERIES = [
    "What capital ratios must American Express maintain?",
    "How does the company generate revenue?",
    "What interchange regulation applies to debit transactions?",
    "What climate risks does the company describe?",
    "Who are the main competitors?",
    "What is the net write-off rate on card member loans?",
]


def _percentiles(samples_ms):
    s = sorted(samples_ms)
    return {
        "mean": statistics.mean(s),
        "p50": s[len(s) // 2],
        "p95": s[min(len(s) - 1, int(len(s) * 0.95))],
        "min": s[0], "max": s[-1],
    }


def bench_retrieval(runs: int = 20):
    from retriever import search
    # warm the model + indexes once (exclude load time from measurement)
    search(QUERIES[0], k=5, mode="hybrid")

    print(f"Retrieval latency over {runs} runs x {len(QUERIES)} queries\n")
    print(f"{'mode':8} {'mean':>8} {'p50':>8} {'p95':>8} {'min':>8} {'max':>8}  (ms)")
    print("-" * 60)
    for mode in ["dense", "sparse", "hybrid"]:
        samples = []
        for _ in range(runs):
            for q in QUERIES:
                t0 = time.perf_counter()
                search(q, k=5, mode=mode)
                samples.append((time.perf_counter() - t0) * 1000)
        p = _percentiles(samples)
        print(f"{mode:8} {p['mean']:8.1f} {p['p50']:8.1f} {p['p95']:8.1f} "
              f"{p['min']:8.1f} {p['max']:8.1f}")


def bench_answer(base="http://localhost:8000"):
    import urllib.parse
    import urllib.request
    import json

    def call(q, no_cache):
        params = urllib.parse.urlencode({"q": q, "mode": "hybrid", "k": 5,
                                         "no_cache": str(no_cache).lower()})
        t0 = time.perf_counter()
        with urllib.request.urlopen(f"{base}/ask?{params}", timeout=60) as r:
            body = json.loads(r.read())
        dt = (time.perf_counter() - t0) * 1000
        return dt, body.get("cached", False)

    print("Answer latency: cold (cache miss) vs warm (cache hit)\n")
    print(f"{'query':42} {'cold_ms':>9} {'warm_ms':>9} {'speedup':>8}")
    print("-" * 72)
    colds, warms = [], []
    for q in QUERIES:
        cold, _ = call(q, no_cache=True)     # force miss: retrieve + LLM
        warm, cached = call(q, no_cache=False)  # should hit (was cached by cold? no:
        # no_cache=True doesn't store; call once more to populate then hit:
        if not cached:
            call(q, no_cache=False)          # populate
            warm, cached = call(q, no_cache=False)
        colds.append(cold); warms.append(warm)
        print(f"{q[:42]:42} {cold:9.0f} {warm:9.0f} {cold/warm:7.1f}x")
    print("-" * 72)
    print(f"{'MEAN':42} {statistics.mean(colds):9.0f} "
          f"{statistics.mean(warms):9.0f} "
          f"{statistics.mean(colds)/statistics.mean(warms):7.1f}x")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    if what == "retrieval":
        bench_retrieval()
    elif what == "answer":
        bench_answer()
    else:
        print("usage: python benchmark.py [retrieval|answer]")