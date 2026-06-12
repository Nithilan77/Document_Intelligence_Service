"""Embedding + FAISS index build (Phase 2).

Loads chunks.json, embeds every chunk with all-MiniLM-L6-v2, L2-normalizes
the vectors, and builds a FAISS IndexFlatIP. Inner product on normalized
vectors == cosine similarity, and IndexFlatIP is *exact* (brute-force), so
at ~2k chunks we get correct top-k with no approximation to defend.

The index and a parallel chunk map are persisted to disk so the API can load
them on startup instead of re-embedding every boot. Rebuild is an explicit
offline step, mirroring how this is done in production.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_PATH = DATA_DIR / "index.faiss"
MAP_PATH = DATA_DIR / "chunk_map.pkl"


def load_chunks(data_dir: Path = DATA_DIR) -> list[dict]:
    path = data_dir / "chunks.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run ingest.py first")
    return json.loads(path.read_text())


def build_index(data_dir: Path = DATA_DIR) -> None:
    chunks = load_chunks(data_dir)
    print(f"Loaded {len(chunks)} chunks")

    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]
    print(f"Embedding with {MODEL_NAME} ...")
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2 normalize -> inner product = cosine
        convert_to_numpy=True,
    ).astype("float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    print(f"Built IndexFlatIP: {index.ntotal} vectors, dim={dim}")

    faiss.write_index(index, str(INDEX_PATH))
    with open(MAP_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Persisted index -> {INDEX_PATH.name}, chunk map -> {MAP_PATH.name}")


if __name__ == "__main__":
    build_index()