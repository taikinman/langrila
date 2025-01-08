import json
import os
from typing import Any, cast

from ..core.memory import BaseConversationMemory


class JSONConversationMemory(BaseConversationMemory):
    def __init__(self, path: str = "conversation_memory.json", exist_ok: bool = False) -> None:
        self.path = path

        if not exist_ok and os.path.exists(self.path):
            os.remove(self.path)

    def store(self, conversation_history: list[list[dict[str, Any]]]) -> None:
        with open(self.path, "w") as f:
            json.dump(conversation_history, f, ensure_ascii=False, indent=2)

    def load(self) -> list[list[dict[str, Any]]]:
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                conversation_history = json.load(f)
            return cast(list[list[dict[str, Any]]], conversation_history)
        else:
            return []
