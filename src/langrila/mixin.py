class ConversationMixin:
    def load_conversation(self) -> list[dict[str, str]]:
        if self.conversation_memory is not None:
            messages: list[dict[str, str]] = self.conversation_memory.load()
        else:
            messages = []

        return messages

    def save_conversation(self, messages: list[dict[str, str]]) -> None:
        self.conversation_memory.store(messages)


class FilterMixin:
    def apply_content_filter(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return self.content_filter.apply(messages)

    def restore_content_filter(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return self.content_filter.restore(messages)
