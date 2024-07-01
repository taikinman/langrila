from typing import Any

import numpy as np


class ConversationMixin:
    def load_conversation(self) -> list[dict[str, str]]:
        if self.conversation_memory is not None:
            messages: list[dict[str, str]] = self.conversation_memory.load()
        else:
            messages = []

        return messages

    def save_conversation(self, messages: list[dict[str, str]]) -> None:
        self.conversation_memory.store(messages)

    def _init_conversation_memory(self, init_conversation: list[dict[str, Any]] | None) -> None:
        if (
            self.conversation_memory
            and init_conversation
            and (
                not hasattr(self, "_INIT_STATUS")
                or (hasattr(self, "_INIT_STATUS") and not self._INIT_STATUS)
            )
        ):
            memory: list[dict[str, Any]] = self.conversation_memory.load()

            # check if init_conversation is already stored in conversation_memory
            # if not, add it to the conversation_memory
            if not np.isin(init_conversation, memory).all():
                memory.extend(init_conversation)
                self.conversation_memory.store(memory)
                self._INIT_STATUS = True


class FilterMixin:
    def apply_content_filter(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return self.content_filter.apply(messages)

    def restore_content_filter(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return self.content_filter.restore(messages)
