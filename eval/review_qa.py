"""Print each question with its POOLED candidate chunks for gold selection.

For each question, lists candidate chunks (pooled across dense/sparse/hybrid)
with the modes that found each, the section, and a text snippet. You read
these and decide which chunk_ids actually ANSWER the question -- those become
gold_chunk_ids in qa_set.json.

Usage: python review_qa.py [qa_candidates.json]
"""
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "qa_candidates.json"
qa = json.loads(src.read_text())

for i, item in enumerate(qa, 1):
    print("=" * 80)
    print(f"Q{i} [{item['type']}]: {item['question']}")
    if item.get("doc_hint"):
        print(f"   (restricted to doc containing '{item['doc_hint']}')")
    print(f"   {len(item['candidates'])} pooled candidates:\n")
    for c in item["candidates"]:
        modes = "+".join(m[0] for m in c["found_by"])  # d/s/h
        print(f"   [{modes:5}] {c['chunk_id']}")
        print(f"           {c['doc_id']} | {c['section'][:46]}")
        print(f"           {c['text'][:230]}")
        print()
    print("   -> put the chunk_id(s) that ANSWER this into gold_chunk_ids\n")