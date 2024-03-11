import asyncio
from abc import ABC, abstractmethod


class BaseModule(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    async def arun(self, *args, **kwargs):
        raise NotImplementedError

    def stream(self, *args, **kwargs):
        raise NotImplementedError

    async def astream(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        _async = kwargs.pop("arun", False)
        _stream = kwargs.pop("stream", False)
        if _async:
            if _stream:
                return self.astream(*args, **kwargs)
            else:
                return asyncio.create_task(self.arun(*args, **kwargs))
        else:
            if _stream:
                return self.stream(*args, **kwargs)
            else:
                return self.run(*args, **kwargs)

    def load_conversation(self) -> list[dict[str, str]]:
        if self.conversation_memory is not None:
            messages: list[dict[str, str]] = self.conversation_memory.load()
        else:
            messages = []

        return messages

    def save_conversation(self, messages: list[dict[str, str]]) -> None:
        self.conversation_memory.store(messages)

    def apply_content_filter(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return self.content_filter.apply(messages)

    def restore_content_filter(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return self.content_filter.restore(messages)


class BaseConversationLengthAdjuster(ABC):
    @abstractmethod
    def run(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        raise NotImplementedError

    def __call__(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return self.run(messages)


class BaseFilter(ABC):
    @abstractmethod
    def apply(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        raise NotImplementedError

    def aapply(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def restore(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        raise NotImplementedError

class BaseConversationMemory(ABC):
    @abstractmethod
    def store(self, conversation_history: list[dict[str, str]]):
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str):
        raise NotImplementedError
