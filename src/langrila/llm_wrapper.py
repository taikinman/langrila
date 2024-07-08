from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator, Optional

from PIL import Image

from .base import (
    BaseChatModule,
    BaseConversationLengthAdjuster,
    BaseConversationMemory,
    BaseFilter,
    BaseFunctionCallingModule,
    BaseMessage,
)
from .mixin import ConversationMixin, FilterMixin
from .result import CompletionResults, FunctionCallingResults
from .usage import TokenCounter


class ChatWrapperModule(ABC, ConversationMixin, FilterMixin):
    def __init__(
        self,
        chat_model: BaseChatModule,
        conversation_memory: Optional[BaseConversationMemory] = None,
        content_filter: Optional[BaseFilter] = None,
        token_counter: Optional[TokenCounter] = None,
    ):
        self.chat_model = chat_model
        self.conversation_memory = conversation_memory
        self.content_filter = content_filter
        self.token_counter = token_counter
        self._INIT_STATUS = False

    @abstractmethod
    def _get_client_message_type(self) -> type[BaseMessage]:
        raise NotImplementedError

    def run(
        self,
        prompt: str,
        images: Image.Image | bytes | list[Image.Image | bytes] | None = None,
        init_conversation: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> CompletionResults:
        Message = self._get_client_message_type()
        self._init_conversation_memory(init_conversation=init_conversation)

        messages = self.load_conversation()

        if isinstance(init_conversation, list) and len(messages) == 0:
            messages.extend(init_conversation)

        messages.append(Message(content=prompt, images=images, **kwargs).as_user)

        if self.content_filter is not None:
            messages = self.content_filter.apply(messages)

        response = self.chat_model.run(messages)

        if self.token_counter is not None:
            self.token_counter += response.usage

        if self.content_filter is not None:
            response.message = self.restore_content_filter([response.message])[0]

        messages.append(response.message)

        if self.conversation_memory is not None:
            self.save_conversation(messages)

        return response

    async def arun(
        self,
        prompt: str,
        images: Image.Image | bytes | list[Image.Image | bytes] | None = None,
        init_conversation: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> CompletionResults:
        Message = self._get_client_message_type()
        self._init_conversation_memory(init_conversation=init_conversation)

        messages = self.load_conversation()

        if isinstance(init_conversation, list) and len(messages) == 0:
            messages.extend(init_conversation)

        messages.append(Message(content=prompt, images=images, **kwargs).as_user)

        if self.content_filter is not None:
            messages = self.content_filter.apply(messages)

        response = await self.chat_model.arun(messages)

        if self.token_counter is not None:
            self.token_counter += response.usage

        if self.content_filter is not None:
            response.message = self.restore_content_filter([response.message])[0]

        messages.append(response.message)

        if self.conversation_memory is not None:
            self.save_conversation(messages)

        return response

    def stream(
        self,
        prompt: str,
        images: Image.Image | bytes | list[Image.Image | bytes] | None = None,
        init_conversation: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> Generator[CompletionResults, None, None]:
        Message = self._get_client_message_type()
        self._init_conversation_memory(init_conversation=init_conversation)

        messages = self.load_conversation()

        if isinstance(init_conversation, list) and len(messages) == 0:
            messages.extend(init_conversation)

        messages.append(Message(content=prompt, images=images, **kwargs).as_user)

        if self.content_filter is not None:
            messages = self.content_filter.apply(messages)

        response = self.chat_model.stream(messages)

        for chunk in response:
            if isinstance(chunk, CompletionResults):
                if self.content_filter is not None:
                    chunk.message = self.restore_content_filter([chunk.message])[0]
                yield chunk

            else:
                raise AssertionError

        if self.token_counter is not None:
            self.token_counter += chunk.usage

        messages.append(chunk.message)

        if self.conversation_memory is not None:
            self.save_conversation(messages)

    async def astream(
        self,
        prompt: str,
        images: Image.Image | bytes | list[Image.Image | bytes] | None = None,
        init_conversation: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncGenerator[CompletionResults, None]:
        Message = self._get_client_message_type()
        self._init_conversation_memory(init_conversation=init_conversation)

        messages = self.load_conversation()

        if isinstance(init_conversation, list) and len(messages) == 0:
            messages.extend(init_conversation)

        messages.append(Message(content=prompt, images=images, **kwargs).as_user)

        if self.content_filter is not None:
            messages = self.content_filter.apply(messages)

        response = self.chat_model.astream(messages)

        async for chunk in response:
            if isinstance(chunk, CompletionResults):
                if self.content_filter is not None:
                    chunk.message = self.restore_content_filter([chunk.message])[0]
                yield chunk
            else:
                raise AssertionError

        if self.token_counter is not None:
            self.token_counter += chunk.usage

        messages.append(chunk.message)

        if self.conversation_memory is not None:
            self.save_conversation(messages)


class FunctionCallingWrapperModule(ABC, ConversationMixin, FilterMixin):
    def __init__(
        self,
        function_calling_model: BaseFunctionCallingModule,
        conversation_memory: Optional[BaseConversationMemory] = None,
        content_filter: Optional[BaseFilter] = None,
        token_counter: Optional[TokenCounter] = None,
    ):
        self.function_calling_model = function_calling_model
        self.conversation_memory = conversation_memory
        self.content_filter = content_filter
        self.token_counter = token_counter
        self._INIT_STATUS = False

    @abstractmethod
    def _get_client_message_type(self) -> type[BaseMessage]:
        raise NotImplementedError

    def run(
        self,
        prompt: str,
        init_conversation: Optional[list[dict[str, str]]] = None,
        **kwargs,
    ) -> FunctionCallingResults:
        Message = self._get_client_message_type()
        self._init_conversation_memory(init_conversation=init_conversation)

        messages = self.load_conversation()

        if isinstance(init_conversation, list) and len(messages) == 0:
            messages.extend(init_conversation)

        messages.append(Message(content=prompt).as_user)

        if self.content_filter is not None:
            messages = self.apply_content_filter(messages)

        response = self.function_calling_model.run(messages, **kwargs)

        if self.token_counter is not None:
            self.token_counter += response.usage

        if self.content_filter is not None:
            for i, _ in enumerate(response.results):
                response.results[i].args = self.restore_content_filter([response.results[i].args])[
                    0
                ]

        if self.conversation_memory is not None:
            for result in response.results:
                messages.append(
                    Message(content=str(result.output), name=result.funcname).as_function
                )

            self.save_conversation(messages)

        return response

    async def arun(
        self,
        prompt: str,
        init_conversation: Optional[list[dict[str, str]]] = None,
        **kwargs,
    ) -> FunctionCallingResults:
        Message = self._get_client_message_type()
        self._init_conversation_memory(init_conversation=init_conversation)

        messages = self.load_conversation()

        if isinstance(init_conversation, list) and len(messages) == 0:
            messages.extend(init_conversation)

        messages.append(Message(content=prompt).as_user)

        if self.content_filter is not None:
            messages = self.apply_content_filter(messages)

        response = await self.function_calling_model.arun(messages, **kwargs)

        if self.token_counter is not None:
            self.token_counter += response.usage

        if self.content_filter is not None:
            for i, _ in enumerate(response.results):
                response.results[i].args = self.restore_content_filter([response.results[i].args])[
                    0
                ]

        if self.conversation_memory is not None:
            for result in response.results:
                messages.append(
                    Message(content=str(result.output), name=result.funcname).as_function
                )

            self.save_conversation(messages)

        return response
