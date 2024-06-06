from typing import Any, Optional

from pydantic import BaseModel

from .usage import Usage


class CompletionResults(BaseModel):
    message: dict[str, str | list[str]]
    usage: Usage
    prompt: Any = None


class ToolOutput(BaseModel):
    call_id: str | None
    funcname: str | None
    args: str | None
    output: Any


class FunctionCallingResults(BaseModel):
    usage: Usage
    results: list[ToolOutput]
    prompt: Optional[str | dict[str, str | list[str]] | list[dict[str, str | list[str]]]] = None


class EmbeddingResults(BaseModel):
    text: list[str]
    embeddings: list[list[float]]
    usage: Usage


class RetrievalResult(BaseModel):
    ids: list[int | str]
    documents: list[str]
    metadatas: Optional[list[dict[str, Any]] | list[None]]
    scores: list[float]
    collections: list[str]
    usage: Usage = Usage(prompt_tokens=0, completion_tokens=0)
