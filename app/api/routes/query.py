from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import get_query_service
from app.api.schemas.query import QueryRequest
from app.services.query_service import QueryService

router = APIRouter(tags=["query"])


class QueryBody(BaseModel):
    question: str = Field(min_length=1, description="User question")
    top_k: int | None = Field(default=None, ge=1, le=20, description="Optional retrieval override")
    mode: Literal["extractive", "generative"] = Field(
        default="extractive",
        description="Answering mode. extractive preserves existing behavior.",
    )
    debug: bool = Field(default=False, description="Include retrieval diagnostics in response")


@router.post("/query")
def query(body: QueryBody, service: QueryService = Depends(get_query_service)) -> dict[str, object]:
    response = service.query(
        QueryRequest(
            question=body.question,
            top_k=body.top_k,
            mode=body.mode,
            debug=body.debug,
        )
    )
    return response.to_dict()
