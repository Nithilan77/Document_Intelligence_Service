"""Document Intelligence Service — FastAPI entrypoint.

Phase 0: skeleton + health check.
Phase 2: dense retrieval query endpoint.
"""
import os

import redis
from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI(title="Document Intelligence Service", version="0.2.0")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_redis = redis.from_url(REDIS_URL, socket_connect_timeout=2)


@app.get("/health")
def health():
    """Liveness + Redis reachability."""
    try:
        _redis.ping()
        redis_ok = True
    except redis.exceptions.RedisError:
        redis_ok = False
    return {"status": "ok", "redis": redis_ok}


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    section: str
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    k: int
    results: list[RetrievedChunk]


@app.get("/search", response_model=SearchResponse)
def search_endpoint(
    q: str = Query(..., description="Query text"),
    k: int = Query(5, ge=1, le=20),
):
    """Dense retrieval: top-k chunks for a query."""
    # Imported lazily so the app can boot (and /health works) even before
    # the FAISS index has been built.
    from retriever import search

    results = search(q, k=k)
    return {"query": q, "k": k, "results": results}


class AskResponse(BaseModel):
    question: str
    answer: str
    mode: str
    sources: list


@app.get("/ask", response_model=AskResponse)
def ask_endpoint(
    q: str = Query(..., description="Question"),
    k: int = Query(5, ge=1, le=10),
    mode: str = Query("hybrid", pattern="^(dense|sparse|hybrid)$"),
):
    """Grounded answer: retrieve top-k chunks and generate a cited answer."""
    from generate import answer

    result = answer(q, k=k, mode=mode)
    return result.to_dict()