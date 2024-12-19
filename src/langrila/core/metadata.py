from abc import ABC, abstractmethod
from typing import Any


class BaseMetadataStore(ABC):
    @abstractmethod
    def store(self, ids: list[int], metadatas: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def fetch(self, ids: list[int]) -> list[dict[str, Any]]:
        raise NotImplementedError


class BaseMetadataFilter(ABC):
    @abstractmethod
    def run(self, metadata: dict[str, str]) -> bool:
        raise NotImplementedError
