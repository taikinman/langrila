from abc import ABC, abstractmethod
from typing import AsyncGenerator, Generator

from ..result import CompletionResults, FunctionCallingResults


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
