from abc import ABC, abstractmethod


class BaseMetadataFilter(ABC):
    @abstractmethod
    def run(self, metadata: dict[str, str]) -> bool:
        raise NotImplementedError
