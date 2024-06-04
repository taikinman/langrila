from abc import ABC, abstractmethod

from ..result import EmbeddingResults


class BaseEmbeddingModule(ABC):
    @abstractmethod
    def run(self, text: str | list[str]) -> EmbeddingResults:
        raise NotImplementedError

    async def arun(self, text: str | list[str]) -> EmbeddingResults:
        raise NotImplementedError
