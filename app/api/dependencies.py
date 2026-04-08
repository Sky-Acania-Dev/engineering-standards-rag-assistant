from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Literal

from app.llm.generation import GeneratorConfig
from app.services.query_service import QueryService, QueryServiceConfig


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Dependencies:
    query_service: QueryService


@lru_cache(maxsize=1)
def get_dependencies() -> Dependencies:
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
        min_score_thresholds={
            "default": float(os.getenv("QUERY_MIN_SCORE", "0.2")),
            "sentence_transformer": float(os.getenv("QUERY_MIN_SCORE_SENTENCE_TRANSFORMER", "0.4")),
        },
        min_chunks=max(1, int(os.getenv("QUERY_MIN_CHUNKS", "1"))),
        retrieval_top_k=max(1, int(os.getenv("QUERY_TOP_K", "5"))),
        generation=GeneratorConfig(
            provider=os.getenv("QUERY_GENERATION_PROVIDER", "extractive"),
            enabled=_read_bool("QUERY_GENERATION_ENABLED", False),
            model=os.getenv("QUERY_GENERATION_MODEL"),
            endpoint=os.getenv("QUERY_GENERATION_ENDPOINT", "http://localhost:11434/api/generate"),
            timeout_seconds=float(os.getenv("QUERY_GENERATION_TIMEOUT_SECONDS", "15.0")),
            temperature=float(os.getenv("QUERY_GENERATION_TEMPERATURE", "0.0")),
        ),
    )

    return Dependencies(
        query_service=QueryService.from_index_dir(
            index_dir=index_dir,
            embedder_provider=embedder_provider,
            embedding_model=embedding_model,
            config=config,
        )
    )


@lru_cache(maxsize=1)
def get_query_service() -> QueryService:
    return get_dependencies().query_service
