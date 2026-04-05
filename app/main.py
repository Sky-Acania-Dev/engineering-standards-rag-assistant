from __future__ import annotations

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.query import router as query_router

app = FastAPI(title="Engineering Standards RAG Assistant", version="0.1.0")
app.include_router(health_router)
app.include_router(query_router)
print("API routes registered: ", [route.path for route in app.routes])