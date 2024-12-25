import copy

from ..core.memory import BaseConversationMemory


class InMemoryConversationMemory(BaseConversationMemory):
    def __init__(self):
        self.history = []

    def store(self, conversation_history: list[dict[str, str]]):
        self.history = copy.deepcopy(conversation_history)

    def load(self) -> list[dict[str, str]]:
        return self.history
