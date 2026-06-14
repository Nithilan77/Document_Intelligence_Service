"""Generate CONCEPTUAL/paraphrased candidate questions (Phase 5, balance pass).

The first question set skewed toward exact-term queries (favoring BM25). These
questions are deliberately phrased WITHOUT the target chunk's key terms, so
answering them requires semantic matching -- the regime where dense retrieval
should contribute. Pooled across modes, same as before; you verify gold.

Appends to the existing pool by writing qa_candidates_conceptual.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from retriever import search  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "qa_candidates_conceptual.json"
POOL_MODES = ["dense", "sparse", "hybrid"]
POOL_DEPTH = 6

# Conceptual questions: paraphrased, no rare keyword shared with the answer.
QUESTIONS = [
    ("How does the business make most of its money?",
     None, "conceptual"),
    ("What could happen if a big technology firm decided to enter payments?",
     None, "conceptual"),
    ("Why might extreme weather hurt the company's operations?",
     None, "conceptual"),
    ("What happens to earnings when the dollar strengthens against other currencies?",
     None, "conceptual"),
    ("Why could a downturn in consumer spending hurt the lender specifically?",
     "axp", "conceptual"),
    ("What makes it hard for the company to keep large partners from leaving?",
     None, "conceptual"),
    ("How could new privacy or data rules raise the company's costs?",
     None, "conceptual"),
    ("Why does the company depend on banks to put its cards in customers' hands?",
     "ma", "conceptual"),
]


def main():
    out = []
    for q, hint, qtype in QUESTIONS:
        pool = {}
        for mode in POOL_MODES:
            for r in search(q, k=POOL_DEPTH, mode=mode):
                if hint and hint.lower() not in r["doc_id"].lower():
                    continue
                cid = r["chunk_id"]
                if cid not in pool:
                    pool[cid] = {
                        "chunk_id": cid, "doc_id": r["doc_id"],
                        "section": r["section"],
                        "text": " ".join(r["text"].split())[:320],
                        "found_by": [],
                    }
                pool[cid]["found_by"].append(mode)
        out.append({
            "question": q, "type": qtype, "doc_hint": hint,
            "candidates": list(pool.values()),
            "gold_chunk_ids": [], "_verified": False,
        })
        print(f"  pooled {len(pool):2d} candidates for: {q[:55]}")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(out)} conceptual questions -> {OUT.name}")
    print("NEXT: python review_qa.py qa_candidates_conceptual.json")


if __name__ == "__main__":
    main()