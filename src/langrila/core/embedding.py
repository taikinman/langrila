from abc import ABC, abstractmethod

from .pydantic import BaseModel
from .usage import Usage


class EmbeddingResults(BaseModel):
    text: list[str]
    embeddings: list[list[float]]
    usage: Usage


class BaseEmbeddingModule(ABC):
    @abstractmethod
    def run(self, text: str | list[str]) -> EmbeddingResults:
        raise NotImplementedError

    async def arun(self, text: str | list[str]) -> EmbeddingResults:
        raise NotImplementedError
