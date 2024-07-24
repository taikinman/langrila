from typing import Any, Optional

from pydantic import BaseModel

from .usage import Usage


class CompletionResults(BaseModel):
    message: Any
    usage: Usage = Usage()
    prompt: Any = None


class ToolCallResponse(BaseModel):
    name: str
    args: Any
    call_id: str | None = None


class ToolOutput(BaseModel):
    call_id: str | None
    funcname: str | None
    args: Any | None
    output: Any


class FunctionCallingResults(BaseModel):
    usage: Usage
    results: list[Any]
    calls: Any | None = None
    prompt: Any | None = None


class EmbeddingResults(BaseModel):
    text: list[str]
    embeddings: list[list[float]]
    usage: Usage


class RetrievalResults(BaseModel):
    ids: list[int | str]
    documents: list[str]
    metadatas: Optional[list[dict[str, Any]] | list[None]]
    scores: list[float]
    collections: list[str]
    usage: Usage = Usage(prompt_tokens=0, completion_tokens=0)
