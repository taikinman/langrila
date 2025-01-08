import os
import pickle
from typing import Any, cast

from ..core.memory import BaseConversationMemory


class PickleConversationMemory(BaseConversationMemory):
    def __init__(self, path: str = "conversation_memory.pkl", exist_ok: bool = False) -> None:
        self.path = path

        if not exist_ok and os.path.exists(self.path):
            os.remove(self.path)

    def store(self, conversation_history: list[list[dict[str, Any]]]) -> None:
        with open(self.path, "wb") as f:
            pickle.dump(conversation_history, f)

    def load(self) -> list[list[dict[str, Any]]]:
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                conversation_history = pickle.load(f)
            return cast(list[list[dict[str, Any]]], conversation_history)
        else:
            return []
