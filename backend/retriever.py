"""Retriever with dense / sparse / hybrid modes (Phase 3).

- dense:  FAISS IndexFlatIP over MiniLM embeddings (cosine).
- sparse: BM25 keyword retrieval.
- hybrid: Reciprocal Rank Fusion (RRF) over dense + sparse rankings.

RRF combines two ranked lists without needing to normalize their scores onto
a common scale -- it uses only ranks: score(d) = sum 1/(k + rank_r(d)) over
retrievers r. k (default 60, the canonical value) is tunable so the eval can
confirm retrieval is insensitive to it rather than trusting a magic number.

All three modes resolve to the same chunk objects, so the Phase 5 eval can
score each mode through one code path -- the mode switch IS the ablation.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from embed import INDEX_PATH, MAP_PATH, MODEL_NAME
from sparse import BM25_PATH, tokenize

_model: SentenceTransformer | None = None
_index = None
_chunks: list[dict] | None = None
_by_id: dict[str, dict] | None = None
_bm25 = None
_bm25_ids: list[str] | None = None
_bm25_use_stopwords: bool = True


def _ensure_dense() -> None:
    global _model, _index, _chunks, _by_id
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    if _index is None:
        if not Path(INDEX_PATH).exists():
            raise FileNotFoundError(f"{INDEX_PATH} not found -- run embed.py first")
        _index = faiss.read_index(str(INDEX_PATH))
    if _chunks is None:
        with open(MAP_PATH, "rb") as f:
            _chunks = pickle.load(f)
        _by_id = {c["chunk_id"]: c for c in _chunks}


def _ensure_sparse() -> None:
    global _bm25, _bm25_ids, _bm25_use_stopwords
    if _bm25 is None:
        if not Path(BM25_PATH).exists():
            raise FileNotFoundError(f"{BM25_PATH} not found -- run sparse.py first")
        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)
        _bm25 = data["bm25"]
        _bm25_ids = data["chunk_ids"]
        _bm25_use_stopwords = data.get("use_stopwords", True)


def _format(chunk: dict, score: float) -> dict:
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "section": chunk.get("section", ""),
        "score": float(score),
        "text": chunk["text"],
    }


def _dense_ranking(query: str, depth: int) -> list[tuple[str, float]]:
    """Return [(chunk_id, score)] for the top `depth` dense hits."""
    _ensure_dense()
    q = _model.encode([query], normalize_embeddings=True,
                      convert_to_numpy=True).astype("float32")
    scores, idxs = _index.search(q, depth)
    out = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        out.append((_chunks[idx]["chunk_id"], float(score)))
    return out


def _sparse_ranking(query: str, depth: int) -> list[tuple[str, float]]:
    """Return [(chunk_id, score)] for the top `depth` BM25 hits."""
    _ensure_sparse()
    scores = _bm25.get_scores(tokenize(query, _bm25_use_stopwords))
    top = np.argsort(scores)[::-1][:depth]
    return [(_bm25_ids[i], float(scores[i])) for i in top]


def _rrf(rankings: list[list[tuple[str, float]]], k: int) -> dict[str, float]:
    """Reciprocal Rank Fusion over multiple ranked lists of chunk_ids."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (cid, _score) in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return fused


def search(query: str, k: int = 5, mode: str = "hybrid",
           rrf_k: int = 60, depth: int = 50) -> list[dict]:
    """Top-k retrieval.

    mode: 'dense' | 'sparse' | 'hybrid'
    depth: how deep each retriever goes before fusion (hybrid only).
    """
    _ensure_dense()  # always needed to resolve chunk objects

    if mode == "dense":
        ranked = _dense_ranking(query, k)
        return [_format(_by_id[cid], s) for cid, s in ranked]

    if mode == "sparse":
        ranked = _sparse_ranking(query, k)
        return [_format(_by_id[cid], s) for cid, s in ranked]

    if mode == "hybrid":
        dense = _dense_ranking(query, depth)
        sparse = _sparse_ranking(query, depth)
        fused = _rrf([dense, sparse], rrf_k)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return [_format(_by_id[cid], s) for cid, s in ranked]

    raise ValueError(f"unknown mode: {mode!r} (use dense|sparse|hybrid)")


if __name__ == "__main__":
    import sys
    mode = "hybrid"
    args = sys.argv[1:]
    if args and args[0] in ("dense", "sparse", "hybrid"):
        mode = args.pop(0)
    q = " ".join(args) or "What are the main risk factors?"
    print(f"[mode={mode}] {q}\n")
    for i, r in enumerate(search(q, k=5, mode=mode), 1):
        print(f"[{i}] {r['doc_id']} | {r['section']} | {r['score']:.4f}")
        print("    " + r["text"][:150].replace("\n", " "))