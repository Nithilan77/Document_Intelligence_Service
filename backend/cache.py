"""Redis-backed caching for retrieval + generation (Phase 6).

Two caches, both keyed by a hash of the inputs so identical requests are
served from Redis instead of recomputing:

- query cache: full /ask result (retrieval + LLM answer) keyed by
  (question, mode, k). The LLM call dominates latency, so caching the whole
  answer turns a repeat question from ~seconds into ~milliseconds.
- embedding cache: query-text -> vector, so repeated query strings skip the
  MiniLM forward pass.

Caches are best-effort: any Redis error degrades to "no cache" rather than
failing the request. TTLs keep entries from going stale forever.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import numpy as np
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_ANSWER_TTL = int(os.getenv("ANSWER_CACHE_TTL", "86400"))   # 1 day
_EMBED_TTL = int(os.getenv("EMBED_CACHE_TTL", "604800"))    # 1 week

_r: Optional[redis.Redis] = None


def _client() -> Optional[redis.Redis]:
    global _r
    if _r is None:
        try:
            _r = redis.from_url(REDIS_URL, socket_connect_timeout=2)
            _r.ping()
        except redis.exceptions.RedisError:
            _r = None
    return _r


def _key(prefix: str, *parts) -> str:
    raw = "|".join(str(p) for p in parts)
    return f"{prefix}:{hashlib.sha1(raw.encode()).hexdigest()}"


# ---- answer cache ----------------------------------------------------------
def get_answer(question: str, mode: str, k: int) -> Optional[dict]:
    r = _client()
    if not r:
        return None
    try:
        val = r.get(_key("ans", question, mode, k))
        return json.loads(val) if val else None
    except redis.exceptions.RedisError:
        return None


def set_answer(question: str, mode: str, k: int, payload: dict) -> None:
    r = _client()
    if not r:
        return
    try:
        r.setex(_key("ans", question, mode, k), _ANSWER_TTL, json.dumps(payload))
    except redis.exceptions.RedisError:
        pass


# ---- embedding cache -------------------------------------------------------
def get_embedding(text: str) -> Optional[np.ndarray]:
    r = _client()
    if not r:
        return None
    try:
        val = r.get(_key("emb", text))
        if not val:
            return None
        return np.frombuffer(val, dtype="float32").copy()
    except redis.exceptions.RedisError:
        return None


def set_embedding(text: str, vec) -> None:
    r = _client()
    if not r:
        return
    try:
        arr = np.asarray(vec, dtype="float32")
        r.setex(_key("emb", text), _EMBED_TTL, arr.tobytes())
    except redis.exceptions.RedisError:
        pass


# ---- introspection ---------------------------------------------------------
def cache_stats() -> dict:
    """Report cache occupancy for the benchmark/README. Counts live keys by
    prefix; returns availability=False if Redis is unreachable.
    """
    r = _client()
    if not r:
        return {"available": False, "answers": 0, "embeddings": 0}
    try:
        answers = sum(1 for _ in r.scan_iter(match="ans:*", count=500))
        embeddings = sum(1 for _ in r.scan_iter(match="emb:*", count=500))
        return {"available": True, "answers": answers, "embeddings": embeddings}
    except redis.exceptions.RedisError:
        return {"available": False, "answers": 0, "embeddings": 0}