"""Async ingestion worker (Phase 6): arq + Redis.

Reuses the queue pattern from the job-processing project. A filing placed in
data/ and registered via POST /documents is ingested in the background so the
API never blocks on the ~30-60s parse+embed+index. arq tracks job state and
result; the API polls via /jobs/{id}.

Run the worker with:  arq worker.WorkerSettings
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from arq.connections import RedisSettings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


async def ingest_document(ctx, file_path: str) -> dict:
    """Background task: parse + chunk a filing, merge into the corpus, and
    rebuild dense + sparse indexes over the updated corpus.

    Returns a small dict result that arq stores and the API surfaces via
    /jobs/{id}. Raising on error lets arq mark the job failed.
    """
    from ingest import chunk_document
    from embed import build_index
    from sparse import build_bm25

    path = Path(file_path)
    doc_id = path.stem
    chunks = chunk_document(path)
    if not chunks:
        return {"ok": False, "doc_id": doc_id, "reason": "no text extracted"}

    corpus_path = DATA_DIR / "chunks.json"
    existing = json.loads(corpus_path.read_text()) if corpus_path.exists() else []
    # Replace any prior chunks for this doc so re-ingesting is idempotent.
    existing = [c for c in existing if c["doc_id"] != doc_id]
    existing.extend(c.to_dict() for c in chunks)
    corpus_path.write_text(json.dumps(existing, indent=2))

    # Rebuild indexes over the merged corpus.
    build_index(DATA_DIR)
    build_bm25(DATA_DIR)

    return {"ok": True, "doc_id": doc_id,
            "added_chunks": len(chunks), "total_chunks": len(existing)}


class WorkerSettings:
    functions = [ingest_document]
    redis_settings = RedisSettings.from_dsn(REDIS_URL)