from abc import ABC, abstractmethod
from typing import Any


class BaseConversationMemory(ABC):
    @abstractmethod
    def store(self, conversation_history: list[list[dict[str, Any]]]) -> None:
        """
        Store the conversation history.

        Parameters
        ----------
        conversation_history : list[list[dict[str, Any]]]
            The conversation history to store. The outer list represents the conversation turns,
            and the inner list represents the messages in each turn.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self) -> list[list[dict[str, Any]]]:
        """
        Load the conversation history. If no history is found, return an empty list.
        The outer list represents the conversation turns, and the inner list represents the messages
        in each turn.

        Returns
        -------
        list[list[dict[str, Any]]]
            The conversation history. If no history is found, return an empty list.
        """
        raise NotImplementedError
