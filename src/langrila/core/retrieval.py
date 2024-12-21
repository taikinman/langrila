from typing import Any

from .pydantic import BaseModel
from .usage import Usage


class RetrievalResults(BaseModel):
    ids: list[int | str]
    documents: list[str]
    metadatas: list[dict[str, Any]] | list[None] | None
    scores: list[float]
    collections: list[str]
    usage: Usage = Usage(prompt_tokens=0, completion_tokens=0)
