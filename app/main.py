from __future__ import annotations

from fastapi import FastAPI

from app.api.dependencies import get_dependencies
from app.api.routes.health import router as health_router
from app.api.routes.query import router as query_router

app = FastAPI(title="Engineering Standards RAG Assistant", version="0.1.0")
app.include_router(health_router)
app.include_router(query_router)


@app.on_event("startup")
def warm_dependencies() -> None:
    # Fail fast if index/runtime embedder settings are incompatible.
    get_dependencies()
