import json
import os
from typing import Any, cast

from ..core.memory import BaseConversationMemory


class JSONConversationMemory(BaseConversationMemory):
    """
    A conversation memory that stores the conversation history in a local JSON file.

    Parameters
    ----------
    path : str, optional
        The path to the JSON file to store the conversation history, by default "conversation_memory.json"
    exist_ok : bool, optional
        If True, do not raise an error if the file already exists and reuses it.
        If False, delete the file if it exists and create a new one, by default False.
    """

    def __init__(self, path: str = "conversation_memory.json", exist_ok: bool = False) -> None:
        self.path = path

        if not exist_ok and os.path.exists(self.path):
            os.remove(self.path)

    def store(self, conversation_history: list[list[dict[str, Any]]]) -> None:
        """
        Store the conversation history in a local JSON file.

        Parameters
        ----------
        conversation_history : list[list[dict[str, Any]]]
            The conversation history to store. The outer list represents the conversation turns,
            and the inner list represents the messages in each turn.
        """
        with open(self.path, "w") as f:
            json.dump(conversation_history, f, ensure_ascii=False, indent=2)

    def load(self) -> list[list[dict[str, Any]]]:
        """
        Load the conversation history from a local JSON file. If no history is found, return an empty list.
        The outer list represents the conversation turns, and the inner list represents the messages
        in each turn.

        Returns
        -------
        list[list[dict[str, Any]]]
            The conversation history. If no history is found, return an empty list.
        """
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                conversation_history = json.load(f)
            return cast(list[list[dict[str, Any]]], conversation_history)
        else:
            return []
