from __future__ import annotations

from functools import lru_cache
import os
from typing import Literal

from app.services.query_service import GenerationConfig, QueryService, QueryServiceConfig


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_query_service() -> QueryService:
    index_dir = os.getenv("INDEX_DIR", "app/data/index")
    embedder_provider = os.getenv("EMBEDDER_PROVIDER", "hash")
    embedding_model = os.getenv("EMBEDDING_MODEL")

    default_mode_raw = os.getenv("QUERY_DEFAULT_MODE", "extractive").strip().lower()
    default_mode: Literal["extractive", "generative"]
    if default_mode_raw == "generative":
        default_mode = "generative"
    else:
        default_mode = "extractive"

    config = QueryServiceConfig(
        default_mode=default_mode,
        min_score=float(os.getenv("QUERY_MIN_SCORE", "0.2")),
        min_chunks=max(1, int(os.getenv("QUERY_MIN_CHUNKS", "1"))),
        retrieval_top_k=max(1, int(os.getenv("QUERY_TOP_K", "5"))),
        generation=GenerationConfig(
            enabled=_read_bool("QUERY_GENERATION_ENABLED", False),
            model=os.getenv("QUERY_GENERATION_MODEL"),
            endpoint=os.getenv("QUERY_GENERATION_ENDPOINT", "http://localhost:11434/api/generate"),
            timeout_seconds=float(os.getenv("QUERY_GENERATION_TIMEOUT_SECONDS", "15.0")),
            temperature=float(os.getenv("QUERY_GENERATION_TEMPERATURE", "0.0")),
        ),
    )

    return QueryService.from_index_dir(
        index_dir=index_dir,
        embedder_provider=embedder_provider,
        embedding_model=embedding_model,
        config=config,
    )
