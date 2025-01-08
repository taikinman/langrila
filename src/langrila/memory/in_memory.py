import copy
from typing import Any

from ..core.memory import BaseConversationMemory


class InMemoryConversationMemory(BaseConversationMemory):
    """
    A conversation memory that stores the conversation history in memory.
    """

    def __init__(self) -> None:
        self.history: list[list[dict[str, Any]]] = []

    def store(self, conversation_history: list[list[dict[str, Any]]]) -> None:
        """
        Store the conversation history in memory.

        Parameters
        ----------
        conversation_history : list[list[dict[str, Any]]]
            The conversation history to store. The outer list represents the conversation turns,
            and the inner list represents the messages in each turn.
        """
        self.history = copy.deepcopy(conversation_history)

    def load(self) -> list[list[dict[str, Any]]]:
        """
        Load the conversation history from memory. If no history is found, return an empty list.
        The outer list represents the conversation turns, and the inner list represents the messages
        in each turn.

        Returns
        -------
        list[list[dict[str, Any]]]
            The conversation history. If no history is found, return an empty list.
        """
        return self.history
