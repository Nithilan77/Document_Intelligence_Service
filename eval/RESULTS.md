# Retrieval Evaluation Results

## Setup
- **Corpus:** 3 SEC 10-K filings (American Express, Mastercard, Visa), 1,911 chunks
  (~1000 chars each, 150-char overlap), parsed from EDGAR HTML.
- **Eval set:** 22 questions, hand-labeled with chunk-level gold, built by pooling
  candidates across dense/sparse/hybrid and verifying each answer against source text.
  Spans two query types: exact-term/factual and conceptual/paraphrased.
- **Retrievers:**
  - Dense: `all-MiniLM-L6-v2` embeddings, FAISS `IndexFlatIP` (exact cosine).
  - Sparse: BM25 (Okapi) with English stopword filtering.
  - Hybrid: Reciprocal Rank Fusion (RRF, k=60) over dense + sparse.

## Results (22 questions)

| mode   | R@1   | R@3   | R@5   | R@10  | MRR   |
|--------|-------|-------|-------|-------|-------|
| dense  | 0.273 | 0.500 | 0.682 | 0.727 | 0.410 |
| sparse | 0.273 | 0.591 | 0.864 | 0.955 | 0.495 |
| hybrid | 0.409 | 0.636 | 0.727 | 0.909 | 0.560 |

## Findings
1. **Hybrid has the best ranking quality** — highest R@1 (0.41) and MRR (0.56),
   ahead of both dense and sparse. For RAG this matters most: the top-ranked chunk
   is what feeds the generator.
2. **Sparse has the highest deep recall** (R@10 0.955) — 10-K language is
   terminology-dense, so exact keyword matching reliably finds the answer chunk
   *somewhere* in the top-k. But sparse ranks it poorly (R@1 only 0.27).
3. **Fusion corrects BM25's weak ranking** — hybrid lifts the right chunk toward
   rank 1 (R@1 0.27 -> 0.41 vs sparse) while retaining most of sparse's recall.

## Design decision
Ship **hybrid**. It is the most robust across query types and optimizes the metric
that matters for grounded generation (top-rank precision). RRF k was left at the
canonical 60 and deliberately not tuned against this small eval set, to avoid
overfitting.

## Limitations
- 22 questions is a small sample; metrics are coarse (~0.045/question).
- Gold is chunk-level; an answer spanning multiple chunks counts a hit if any
  gold chunk is retrieved.
- Corpus is a single fixed snapshot of three payments-industry filings.