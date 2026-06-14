"""Retrieval eval harness (Phase 5): recall@k and MRR across modes.

Scores dense / sparse / hybrid against a human-verified, chunk-level gold set
(qa_set.json). This is the headline artifact: the dense-vs-hybrid delta is the
defensible result, and reporting it honestly -- including if hybrid does NOT
win -- is the point.

Metrics (chunk-level gold):
- recall@k: fraction of questions where >=1 gold chunk appears in top-k.
- MRR:      mean of 1/rank of the FIRST gold chunk (0 if none in top-k).

We evaluate at multiple k and across modes so the comparison is complete.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from retriever import search  # noqa: E402

HERE = Path(__file__).resolve().parent
KS = [1, 3, 5, 10]
MODES = ["dense", "sparse", "hybrid"]


def load_gold(path):
    qa = json.loads(Path(path).read_text())
    # Only score verified items with at least one gold chunk.
    usable = [q for q in qa if q.get("gold_chunk_ids")]
    unverified = [q for q in qa if not q.get("_verified", False)]
    if unverified:
        print(f"WARNING: {len(unverified)} item(s) not marked _verified=true. "
              f"Scoring anyway, but verify them for a trustworthy number.\n")
    return usable


def evaluate(qa, mode, max_k):
    """Return (recall_at_k dict, mrr) for one mode over the gold set."""
    recall_hits = {k: 0 for k in KS}
    rr_sum = 0.0
    for item in qa:
        gold = set(item["gold_chunk_ids"])
        results = search(item["question"], k=max_k, mode=mode)
        ranked_ids = [r["chunk_id"] for r in results]
        # recall@k
        for k in KS:
            if gold & set(ranked_ids[:k]):
                recall_hits[k] += 1
        # reciprocal rank of first gold
        rr = 0.0
        for rank, cid in enumerate(ranked_ids, 1):
            if cid in gold:
                rr = 1.0 / rank
                break
        rr_sum += rr
    n = len(qa)
    recall = {k: recall_hits[k] / n for k in KS}
    return recall, rr_sum / n


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else HERE / "qa_set.json"
    if not Path(path).exists():
        print(f"{path} not found. Build it first: run make_qa_set.py, verify, "
              f"save as qa_set.json.")
        sys.exit(1)
    qa = load_gold(path)
    print(f"Scoring {len(qa)} questions with gold labels.\n")

    max_k = max(KS)
    rows = {}
    for mode in MODES:
        rows[mode] = evaluate(qa, mode, max_k)

    # table
    header = f"{'mode':8} " + " ".join(f"R@{k:<4}" for k in KS) + "  MRR"
    print(header)
    print("-" * len(header))
    for mode in MODES:
        recall, mrr = rows[mode]
        cells = " ".join(f"{recall[k]:.3f}" for k in KS)
        print(f"{mode:8} {cells}  {mrr:.3f}")

    # headline delta
    dr5 = rows["hybrid"][0][5] - rows["dense"][0][5]
    print(f"\nHybrid vs dense, recall@5 delta: {dr5:+.3f}")
    print("(Positive => hybrid helps on this corpus. Report whatever it is.)")


if __name__ == "__main__":
    main()