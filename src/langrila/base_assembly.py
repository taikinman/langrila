from abc import ABC, abstractmethod
from typing import AsyncGenerator, Generator, Literal

from .base import BaseConversationMemory
from .memory.in_memory import InMemoryConversationMemory
from .message_content import ConversationType, InputType
from .result import CompletionResults


class BaseAssembly(ABC):
    def _setup_memory(self, conversation_memory: BaseConversationMemory | None = None):
        if conversation_memory:
            self.__HASMEMORY = True
            return conversation_memory
        else:
            # conversation memory is useful to bridge chat and function calling
            # so if it is not provided, we will use InMemoryConversationMemory internally,
            # and clear the memory after each run
            self.__HASMEMORY = False
            return InMemoryConversationMemory()

    @abstractmethod
    def run(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "required", "any"] | str | None = None,
        tool_only: bool = False,
    ) -> CompletionResults:
        raise NotImplementedError

    async def arun(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "required", "any"] | str | None = None,
        tool_only: bool = False,
    ) -> CompletionResults:
        raise NotImplementedError

    def stream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "required", "any"] | str | None = None,
    ) -> Generator[CompletionResults, None, None]:
        raise NotImplementedError

    async def astream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "required", "any"] | str | None = None,
    ) -> AsyncGenerator[CompletionResults, None]:
        raise NotImplementedError

    def _clear_memory(self, conversation_memory: BaseConversationMemory) -> None:
        if not self.__HASMEMORY:
            # Clear conversation memory if conversation memory is not provided
            conversation_memory.store([])
