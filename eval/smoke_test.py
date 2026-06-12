"""Phase 2 smoke test: provisional retrieval sanity check.

This is NOT the real eval (that's Phase 5 with 20-50 carefully labeled pairs
and chunk-level gold). This is a quick gut-check: for a handful of questions,
does dense retrieval surface a chunk from the document we'd expect, and is the
top result topically right? It gives us a provisional recall@k and catches
gross retrieval failures before we build the full pipeline.

Gold here is at the DOCUMENT level (which filing should answer this), which is
easy to eyeball. Phase 5 upgrades to chunk-level gold for true recall@k.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from retriever import search  # noqa: E402

# (question, expected_doc_id_substring) -- doc-level gold for a quick check.
SMOKE = [
    ("What are the principal risk factors facing the company?", None),
    ("How does the company generate revenue from payment processing?", None),
    ("What credit risk does the company carry on cardmember loans?", "axp"),
    ("What is the company's exposure to interchange fee regulation?", None),
    ("Who are the company's main competitors?", None),
    ("What was total net revenue for the fiscal year?", None),
    ("How many employees does the company have?", None),
    ("What legal proceedings is the company involved in?", None),
]


def run(k: int = 5):
    hits = 0
    scored = 0
    for q, expect in SMOKE:
        results = search(q, k=k)
        top = results[0]
        line = f"\nQ: {q}\n   top: {top['doc_id']} | {top['section']} | {top['score']:.3f}"
        line += f"\n        {top['text'][:120].strip()}..."
        if expect is not None:
            scored += 1
            got = any(expect.lower() in r["doc_id"].lower() for r in results)
            hits += got
            line += f"\n   [expected doc containing '{expect}' in top-{k}: {'HIT' if got else 'MISS'}]"
        print(line)
    if scored:
        print(f"\nProvisional doc-level recall@{k}: {hits}/{scored} = {hits/scored:.0%}")
    print("\n(Smoke test only -- real chunk-level recall@k comes in Phase 5.)")


if __name__ == "__main__":
    run(k=int(sys.argv[1]) if len(sys.argv) > 1 else 5)