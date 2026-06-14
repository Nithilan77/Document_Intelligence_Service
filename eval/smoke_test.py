"""Phase 3 smoke test: dense vs sparse vs hybrid, side by side.

Still NOT the real eval (Phase 5 has chunk-level gold + recall@k/MRR over a
labeled set). This shows, per question, what each retrieval mode returns at
rank 1, so we can eyeball whether hybrid combines dense's semantic strength
with BM25's keyword precision before we measure it rigorously.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from retriever import search  # noqa: E402

QUESTIONS = [
    "What are the principal risk factors facing the company?",
    "How does the company generate revenue from payment processing?",
    "What credit risk does the company carry on cardmember loans?",
    "What is the company's exposure to interchange fee regulation?",
    "Who are the company's main competitors?",
    "What was total net revenue for the fiscal year?",
    "How many employees does the company have?",
    "What legal proceedings is the company involved in?",
]

MODES = ["dense", "sparse", "hybrid"]


def run():
    for q in QUESTIONS:
        print(f"\nQ: {q}")
        for mode in MODES:
            top = search(q, k=1, mode=mode)[0]
            snippet = top["text"][:80].replace("\n", " ").strip()
            print(f"  {mode:6} -> {top['doc_id']:14} | {top['section'][:34]:34} | "
                  f"{top['score']:.4f}")
            print(f"            {snippet}")
    print("\n(Smoke test only -- rigorous recall@k / MRR comes in Phase 5.)")


if __name__ == "__main__":
    run()