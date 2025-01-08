import copy
from typing import Any

from ..core.memory import BaseConversationMemory


class InMemoryConversationMemory(BaseConversationMemory):
    def __init__(self) -> None:
        self.history: list[list[dict[str, Any]]] = []

    def store(self, conversation_history: list[list[dict[str, Any]]]) -> None:
        self.history = copy.deepcopy(conversation_history)

    def load(self) -> list[list[dict[str, Any]]]:
        return self.history
