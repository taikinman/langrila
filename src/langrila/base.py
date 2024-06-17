from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator

from .result import CompletionResults, EmbeddingResults, FunctionCallingResults


class BaseChatModule(ABC):
    @abstractmethod
    def run(self, messages: list[dict[str, str]]) -> CompletionResults:
        raise NotImplementedError

    async def arun(self, messages: list[dict[str, str]]) -> CompletionResults:
        raise NotImplementedError

    def stream(self, messages: list[dict[str, str]]) -> Generator[CompletionResults, None, None]:
        raise NotImplementedError

    async def astream(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[CompletionResults, None]:
        raise NotImplementedError


class BaseFunctionCallingModule(ABC):
    @abstractmethod
    def run(self, messages: list[dict[str, str]]) -> FunctionCallingResults:
        raise NotImplementedError

    async def arun(self, messages: list[dict[str, str]]) -> FunctionCallingResults:
        raise NotImplementedError


class BaseEmbeddingModule(ABC):
    @abstractmethod
    def run(self, text: str | list[str]) -> EmbeddingResults:
        raise NotImplementedError

    async def arun(self, text: str | list[str]) -> EmbeddingResults:
        raise NotImplementedError


class BaseConversationLengthAdjuster(ABC):
    @abstractmethod
    def run(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        raise NotImplementedError


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
    def store(self, conversation_history: list[dict[str, str]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> list[dict[str, str]]:
        raise NotImplementedError


class BaseMessage(ABC):
    def __init__(
        self, content: str, images: Any | list[Any] | None = None, name: str | None = None
    ):
        self.content = content
        self.images = images
        self.name = name

    @property
    @abstractmethod
    def as_system(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def as_user(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def as_assistant(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def as_function(self):
        raise NotImplementedError

    # @property
    # def as_tool(self):
    #     return {"role": "tool", "content": self.content}
