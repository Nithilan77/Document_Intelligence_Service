"""Generate a CANDIDATE QA set via retrieval pooling (Phase 5, v2).

Why v2: keyword-presence gold was noisy (matched boilerplate / tables of
contents, not chunks that ANSWER the question). This version instead:

  1. Uses SHARPER questions with specific, locatable answers (not "what are
     the risk factors" whose answer is an entire section).
  2. Pools candidate chunks by running each question through dense, sparse AND
     hybrid retrieval and taking the union of their top hits. Pooling across
     modes avoids any single retriever's bias deciding what you consider.

This still does NOT finalize gold. It writes qa_candidates.json where each
question has a list of POOLED candidate chunks with the text inline. You read
them, mark which actually answer the question (set them as gold), and save as
qa_set.json. Your reading is the gold; retrieval only proposes.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from retriever import search  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "qa_candidates.json"
POOL_MODES = ["dense", "sparse", "hybrid"]
POOL_DEPTH = 6   # take top-6 from each mode; union becomes the candidate pool

# Sharper questions: each targets a specific, locatable fact or passage.
# doc_hint restricts cross-doc questions to one filing.
QUESTIONS = [
    ("What net write-off rate does American Express report on Card Member loans?",
     "axp", "factual"),
    ("What interchange cap did the U.S. Federal Reserve set for large-bank debit transactions?",
     "v", "cross_doc"),
    ("Does Mastercard issue cards or extend credit to account holders?",
     "ma", "cross_doc"),
    ("What capital ratios is American Express required to maintain (CET1, Tier 1, Total)?",
     "axp", "factual"),
    ("What was Mastercard's GAAP net revenue and how much did it grow?",
     "ma", "factual"),
    ("What does the company say about competitors having greater scale or resources?",
     None, "conceptual"),
    ("What climate-related risk does the company say it may not be able to control?",
     None, "conceptual"),
    ("What honor-all-cards or merchant-contract litigation risk is disclosed?",
     "axp", "factual"),
    ("How does currency fluctuation affect revenue earned outside the United States?",
     None, "conceptual"),
    ("What debit routing requirement must be available for ecommerce transactions?",
     "v", "cross_doc"),
    ("What regulatory category (tailoring framework) is American Express assigned to?",
     "axp", "factual"),
    ("What restrictions exist on seller routing choice for debit and prepaid segments?",
     "v", "cross_doc"),
    ("What net interest yield does American Express report?",
     "axp", "factual"),
    ("What does the company disclose about no-surcharge rules?",
     "v", "factual"),
    ("What approval must the company get from the Federal Reserve for capital distributions?",
     "axp", "factual"),
]


def main():
    out = []
    for q, hint, qtype in QUESTIONS:
        pool = {}  # chunk_id -> {chunk fields, found_by:[modes]}
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
            "candidates": list(pool.values()),  # READ these, pick gold
            "gold_chunk_ids": [],               # YOU fill this from candidates
            "_verified": False,
        })
        print(f"  pooled {len(pool):2d} candidates for: {q[:55]}")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(out)} questions -> {OUT.name}")
    print("NEXT: python review_qa.py  (reads candidates), then mark gold in the JSON.")


if __name__ == "__main__":
    main()