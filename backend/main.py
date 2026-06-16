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
    cached: bool = False


@app.get("/ask", response_model=AskResponse)
def ask_endpoint(
    q: str = Query(..., description="Question"),
    k: int = Query(5, ge=1, le=10),
    mode: str = Query("hybrid", pattern="^(dense|sparse|hybrid)$"),
    no_cache: bool = Query(False, description="Bypass the answer cache"),
):
    """Grounded answer: retrieve top-k chunks and generate a cited answer.

    Checks the answer cache first; a hit skips retrieval and the LLM call.
    """
    import cache as _cache
    from generate import answer

    if not no_cache:
        hit = _cache.get_answer(q, mode, k)
        if hit is not None:
            hit["cached"] = True
            return hit

    result = answer(q, k=k, mode=mode).to_dict()
    result["cached"] = False
    _cache.set_answer(q, mode, k, result)
    return result


@app.post("/documents")
async def upload_document(filename: str = Query(..., description="HTML filing in data/")):
    """Enqueue async ingestion of an HTML filing already placed in data/.

    Returns a job id immediately; embedding runs in the arq worker so the API
    stays responsive during the ~30-60s embed. Poll /jobs/{id} for status.
    """
    from pathlib import Path

    from arq import create_pool
    from arq.connections import RedisSettings

    data_dir = Path(__file__).resolve().parent.parent / "data"
    path = data_dir / filename
    if not path.exists():
        return {"status": "error", "reason": f"{filename} not found in data/"}

    redis = await create_pool(RedisSettings.from_dsn(REDIS_URL))
    job = await redis.enqueue_job("ingest_document", str(path))
    return {"status": "enqueued", "job_id": job.job_id, "filename": filename}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    """Poll an ingestion job's status/result."""
    from arq import create_pool
    from arq.connections import RedisSettings
    from arq.jobs import Job

    redis = await create_pool(RedisSettings.from_dsn(REDIS_URL))
    job = Job(job_id, redis)
    status = await job.status()
    result = None
    try:
        if str(status) == "JobStatus.complete":
            result = await job.result(timeout=1)
    except Exception:
        result = None
    return {"job_id": job_id, "status": str(status), "result": result}


@app.get("/cache/stats")
def cache_stats_endpoint():
    """Cache introspection for the benchmark/README."""
    import cache as _cache
    return _cache.cache_stats()