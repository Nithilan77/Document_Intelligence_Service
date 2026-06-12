"""Dense retriever (Phase 2).

Loads the persisted FAISS index + chunk map once, embeds incoming queries
with the same MiniLM model, and returns top-k chunks by cosine similarity.
The model and index are loaded lazily on first use and cached on the module,
so the API pays the load cost once.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from embed import INDEX_PATH, MAP_PATH, MODEL_NAME

_model: SentenceTransformer | None = None
_index = None
_chunks: list[dict] | None = None


def _ensure_loaded() -> None:
    global _model, _index, _chunks
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    if _index is None:
        if not Path(INDEX_PATH).exists():
            raise FileNotFoundError(
                f"{INDEX_PATH} not found -- run embed.py first"
            )
        _index = faiss.read_index(str(INDEX_PATH))
    if _chunks is None:
        with open(MAP_PATH, "rb") as f:
            _chunks = pickle.load(f)


def search(query: str, k: int = 5) -> list[dict]:
    """Return top-k chunks for a query, each with a similarity score."""
    _ensure_loaded()
    q = _model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    scores, idxs = _index.search(q, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        c = _chunks[idx]
        results.append({
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "section": c.get("section", ""),
            "score": float(score),
            "text": c["text"],
        })
    return results


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What are the main risk factors?"
    for i, r in enumerate(search(q, k=5), 1):
        print(f"\n[{i}] {r['doc_id']} | {r['section']} | score={r['score']:.3f}")
        print("    " + r["text"][:160].replace("\n", " "))