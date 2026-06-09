"""Document Intelligence Service — FastAPI entrypoint.

Phase 0: skeleton + health check. The /health endpoint verifies the
service is up and that Redis is reachable, so `docker-compose up` gives
an immediate, honest signal that the whole stack is wired correctly.
"""
import os

import redis
from fastapi import FastAPI

app = FastAPI(title="Document Intelligence Service", version="0.1.0")

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