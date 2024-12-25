from abc import ABC, abstractmethod
from typing import Any


class BaseConversationMemory(ABC):
    @abstractmethod
    def store(self, conversation_history: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> list[dict[str, Any]]:
        raise NotImplementedError
