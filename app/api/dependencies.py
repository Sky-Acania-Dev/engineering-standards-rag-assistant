from __future__ import annotations

from functools import lru_cache
import os

from app.services.query_service import QueryService


@lru_cache(maxsize=1)
def get_query_service() -> QueryService:
    index_dir = os.getenv("INDEX_DIR", "app/data/index")
    embedder_provider = os.getenv("EMBEDDER_PROVIDER", "hash")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    return QueryService.from_index_dir(
        index_dir=index_dir,
        embedder_provider=embedder_provider,
        embedding_model=embedding_model,
    )
