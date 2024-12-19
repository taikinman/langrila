from abc import ABC, abstractmethod
from inspect import isfunction, ismethod
from typing import Any, AsyncGenerator, Callable, Generator, Optional

from pydantic import BaseModel

from ..utils import model2func
from .base import (
    BaseChatModule,
    BaseConversationLengthAdjuster,
    BaseConversationMemory,
    BaseFilter,
    BaseFunctionCallingModule,
    BaseMessage,
)
from .message_content import ConversationType, InputType, Message, TextContent
from .mixin import ConversationMixin, FilterMixin
from .result import CompletionResults, FunctionCallingResults
from .usage import TokenCounter


class ChatWrapperModule(ABC, ConversationMixin, FilterMixin):
    def __init__(
        self,
        chat_model: BaseChatModule,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        token_counter: TokenCounter | None = None,
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
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        **kwargs,
    ) -> CompletionResults:
        _conversation_memory = conversation_memory or self.conversation_memory
        _content_filter = content_filter or self.content_filter

        ClientMessage = self._get_client_message_type()

        # Load conversation memory
        if _conversation_memory is not None:
            messages = _conversation_memory.load()
        else:
            messages = []

        # Convert init_conversation to ClientMessage
        if init_conversation:
            if not isinstance(init_conversation, list):
                init_conversation = [init_conversation]

            messages.extend(init_conversation)

        # Convert messages to UniversalMessage
        messages = [ClientMessage.to_universal_message(message=message) for message in messages]

        # Convert prompt to UniversalMessage
        messages.append(ClientMessage.to_universal_message(message=prompt, role="user"))

        # Apply content filter if available
        if _content_filter is not None:
            for i, m in enumerate(messages):
                for j, content in enumerate(m.content):
                    if isinstance(content, TextContent):
                        messages[i].content[j].text = _content_filter.apply(content.text)

        # Convert UniversalMessage to ClientMessage
        _messages = [
            ClientMessage.to_client_message(message=message)
            for message in ClientMessage._preprocess_message(messages)
        ]

        response: CompletionResults = self.chat_model.run(_messages, **kwargs)

        # Update total tokens
        if self.token_counter is not None:
            self.token_counter += response.usage

        # Append the response to the conversation with the UniversalMessage format
        response.message = ClientMessage.to_universal_message_from_completion_response(
            response=response
        )

        # Restore content filter if available
        if _content_filter is not None:
            for i, content in enumerate(response.message.content):
                if isinstance(content, TextContent):
                    response.message.content[i].text = _content_filter.restore(content.text)

            for m in messages:
                for i, content in enumerate(m.content):
                    if isinstance(content, TextContent):
                        m.content[i].text = _content_filter.restore(content.text)
            # messages = [_content_filter.restore(m) for m in messages]

        messages.append(response.message)

        # Save conversation memory
        if _conversation_memory is not None:
            serializable = [m.model_dump() for m in messages]
            _conversation_memory.store(serializable)

        return response

    async def arun(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        **kwargs,
    ) -> CompletionResults:
        _conversation_memory = conversation_memory or self.conversation_memory
        _content_filter = content_filter or self.content_filter

        ClientMessage = self._get_client_message_type()

        # Load conversation memory
        if _conversation_memory is not None:
            messages = _conversation_memory.load()
        else:
            messages = []

        # Convert init_conversation to ClientMessage
        if init_conversation:
            if not isinstance(init_conversation, list):
                init_conversation = [init_conversation]

            messages.extend(init_conversation)

        # Convert messages to UniversalMessage
        messages = [ClientMessage.to_universal_message(message=message) for message in messages]

        # Convert prompt to UniversalMessage
        messages.append(ClientMessage.to_universal_message(message=prompt, role="user"))

        # Apply content filter if available
        if _content_filter is not None:
            for i, m in enumerate(messages):
                for j, content in enumerate(m.content):
                    if isinstance(content, TextContent):
                        messages[i].content[j].text = _content_filter.apply(content.text)

        # Convert UniversalMessage to ClientMessage
        _messages = [
            ClientMessage.to_client_message(message=message)
            for message in ClientMessage._preprocess_message(messages)
        ]

        response: CompletionResults = await self.chat_model.arun(_messages, **kwargs)

        # Update total tokens
        if self.token_counter is not None:
            self.token_counter += response.usage

        # Append the response to the conversation with the UniversalMessage format
        response.message = ClientMessage.to_universal_message_from_completion_response(
            response=response
        )

        # Restore content filter if available
        if _content_filter is not None:
            for i, content in enumerate(response.message.content):
                if isinstance(content, TextContent):
                    response.message.content[i].text = _content_filter.restore(content.text)

            for m in messages:
                for i, content in enumerate(m.content):
                    if isinstance(content, TextContent):
                        m.content[i].text = _content_filter.restore(content.text)
            # messages = [_content_filter.restore(m) for m in messages]

        messages.append(response.message)

        # Save conversation memory
        if _conversation_memory is not None:
            serializable = [m.model_dump() for m in messages]
            _conversation_memory.store(serializable)

        return response

    def stream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        **kwargs,
    ) -> Generator[CompletionResults, None, None]:
        _conversation_memory = conversation_memory or self.conversation_memory
        _content_filter = content_filter or self.content_filter

        ClientMessage = self._get_client_message_type()

        # Load conversation memory
        if _conversation_memory is not None:
            messages = _conversation_memory.load()
        else:
            messages = []

        # Convert init_conversation to ClientMessage
        if init_conversation:
            if not isinstance(init_conversation, list):
                init_conversation = [init_conversation]

            messages.extend(init_conversation)

        # Convert messages to UniversalMessage
        messages = [ClientMessage.to_universal_message(message=message) for message in messages]

        # Convert prompt to UniversalMessage
        messages.append(ClientMessage.to_universal_message(message=prompt, role="user"))

        # Apply content filter if available
        if _content_filter is not None:
            for i, m in enumerate(messages):
                for j, content in enumerate(m.content):
                    if isinstance(content, TextContent):
                        messages[i].content[j].text = _content_filter.apply(content.text)

        # Convert UniversalMessage to ClientMessage
        _messages = [
            ClientMessage.to_client_message(message=message)
            for message in ClientMessage._preprocess_message(messages)
        ]

        # Run the model
        response = self.chat_model.stream(_messages, **kwargs)

        for chunk in response:
            if isinstance(chunk, CompletionResults):
                # Convert to UniversalMessage
                chunk.message = ClientMessage.to_universal_message_from_completion_response(
                    response=chunk
                )

                if _content_filter is not None:
                    for i, content in enumerate(chunk.message.content):
                        if isinstance(content, TextContent):
                            chunk.message.content[i].text = _content_filter.restore(content.text)

                yield chunk
            else:
                raise AssertionError

        # Update total tokens
        if self.token_counter is not None:
            self.token_counter += chunk.usage

        messages.append(chunk.message)

        # Save conversation memory
        if _conversation_memory is not None:
            serializable = [m.model_dump() for m in messages]
            _conversation_memory.store(serializable)

    async def astream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        **kwargs,
    ) -> AsyncGenerator[CompletionResults, None]:
        _conversation_memory = conversation_memory or self.conversation_memory
        _content_filter = content_filter or self.content_filter

        ClientMessage = self._get_client_message_type()

        # Load conversation memory
        if _conversation_memory is not None:
            messages = _conversation_memory.load()
        else:
            messages = []

        # Convert init_conversation to ClientMessage
        if init_conversation:
            if not isinstance(init_conversation, list):
                init_conversation = [init_conversation]

            messages.extend(init_conversation)

        # Convert messages to UniversalMessage
        messages = [ClientMessage.to_universal_message(message=message) for message in messages]

        # Convert prompt to UniversalMessage
        messages.append(ClientMessage.to_universal_message(message=prompt, role="user"))

        # Apply content filter if available
        if _content_filter is not None:
            for i, m in enumerate(messages):
                for j, content in enumerate(m.content):
                    if isinstance(content, TextContent):
                        messages[i].content[j].text = _content_filter.apply(content.text)

        # Convert UniversalMessage to ClientMessage
        _messages = [
            ClientMessage.to_client_message(message=message)
            for message in ClientMessage._preprocess_message(messages)
        ]

        # Run the model
        response = self.chat_model.astream(_messages, **kwargs)

        async for chunk in response:
            if isinstance(chunk, CompletionResults):
                # Convert to UniversalMessage
                chunk.message = ClientMessage.to_universal_message_from_completion_response(
                    response=chunk
                )

                if _content_filter is not None:
                    for i, content in enumerate(chunk.message.content):
                        if isinstance(content, TextContent):
                            chunk.message.content[i].text = _content_filter.restore(content.text)

                yield chunk
            else:
                raise AssertionError

        # Update total tokens
        if self.token_counter is not None:
            self.token_counter += chunk.usage

        messages.append(chunk.message)

        # Save conversation memory
        if _conversation_memory is not None:
            serializable = [m.model_dump() for m in messages]
            _conversation_memory.store(serializable)


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

    def _set_runnable_tools_dict(self, tools: list[Callable | BaseModel]) -> dict[str, callable]:
        return {f.__name__: f if (isfunction(f) or ismethod(f)) else model2func(f) for f in tools}

    @abstractmethod
    def _get_client_message_type(self) -> type[BaseMessage]:
        raise NotImplementedError

    def run(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        **kwargs,
    ) -> FunctionCallingResults:
        _conversation_memory = conversation_memory or self.conversation_memory
        _content_filter = content_filter or self.content_filter

        ClientMessage = self._get_client_message_type()

        # Load conversation memory
        if _conversation_memory is not None:
            messages = _conversation_memory.load()
        else:
            messages = []

        # Convert init_conversation to ClientMessage
        if init_conversation:
            if not isinstance(init_conversation, list):
                init_conversation = [init_conversation]

            messages.extend(init_conversation)

        # Convert messages to UniversalMessage
        messages = [ClientMessage.to_universal_message(message=message) for message in messages]

        # Convert prompt to UniversalMessage
        messages.append(ClientMessage.to_universal_message(message=prompt, role="user"))

        # Apply content filter if available
        if _content_filter is not None:
            for i, m in enumerate(messages):
                for j, content in enumerate(m.content):
                    if isinstance(content, TextContent):
                        messages[i].content[j].text = _content_filter.apply(content.text)

        # Convert UniversalMessage to ClientMessage
        _messages = [
            ClientMessage.to_client_message(message=message)
            for message in ClientMessage._preprocess_message(messages)
        ]

        # Run the model
        response = self.function_calling_model.run(_messages, **kwargs)

        # Update total tokens
        if self.token_counter is not None:
            self.token_counter += response.usage

        # Append the response to the conversation
        response.results = ClientMessage.to_universal_message_from_function_response(
            response=response
        )

        # Restore content filter if available
        if _content_filter is not None:
            for i, result in enumerate(response.results):
                for j, content in enumerate(result.content):
                    if isinstance(content.args, str):
                        response.results[i].content[j].args = _content_filter.restore(content.args)

                    if isinstance(content.output, str):
                        response.results[i].content[j].output = _content_filter.restore(
                            content.output
                        )

        if response.calls:
            response.calls = ClientMessage.to_universal_message_from_function_call(
                response=response
            )

            if _content_filter is not None:
                for i, content in enumerate(response.calls.content):
                    if isinstance(content, TextContent):
                        response.calls.content[i].text = _content_filter.restore(content.text)

            messages.append(response.calls)
        else:
            messages.extend(response.results)

        # Save conversation memory
        if _conversation_memory is not None and (response.calls or response.results):
            serializable = [m.model_dump() for m in messages]
            _conversation_memory.store(serializable)

        return response

    async def arun(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        **kwargs,
    ) -> FunctionCallingResults:
        _conversation_memory = conversation_memory or self.conversation_memory
        _content_filter = content_filter or self.content_filter

        ClientMessage = self._get_client_message_type()

        # Load conversation memory
        if _conversation_memory is not None:
            messages = _conversation_memory.load()
        else:
            messages = []

        # Convert init_conversation to ClientMessage
        if init_conversation:
            if not isinstance(init_conversation, list):
                init_conversation = [init_conversation]

            messages.extend(init_conversation)

        # Convert messages to UniversalMessage
        messages = [ClientMessage.to_universal_message(message=message) for message in messages]

        # Convert prompt to UniversalMessage
        messages.append(ClientMessage.to_universal_message(message=prompt, role="user"))

        # Apply content filter if available
        if _content_filter is not None:
            for i, m in enumerate(messages):
                for j, content in enumerate(m.content):
                    if isinstance(content, TextContent):
                        messages[i].content[j].text = _content_filter.apply(content.text)

        # Convert UniversalMessage to ClientMessage
        _messages = [
            ClientMessage.to_client_message(message=message)
            for message in ClientMessage._preprocess_message(messages)
        ]

        # Run the model
        response = await self.function_calling_model.arun(_messages, **kwargs)

        # Update total tokens
        if self.token_counter is not None:
            self.token_counter += response.usage

        # Append the response to the conversation
        response.results = ClientMessage.to_universal_message_from_function_response(
            response=response
        )

        # Restore content filter if available
        if _content_filter is not None:
            for i, result in enumerate(response.results):
                for j, content in enumerate(result.content):
                    if isinstance(content.args, str):
                        response.results[i].content[j].args = _content_filter.restore(content.args)

                    if isinstance(content.output, str):
                        response.results[i].content[j].output = _content_filter.restore(
                            content.output
                        )

        if response.calls:
            response.calls = ClientMessage.to_universal_message_from_function_call(
                response=response
            )

            if _content_filter is not None:
                for i, content in enumerate(response.calls.content):
                    if isinstance(content, TextContent):
                        response.calls.content[i].text = _content_filter.restore(content.text)
                # response.calls.content = [
                #     _content_filter.restore(_content) for _content in response.calls.content
                # ]

            messages.append(response.calls)
        else:
            messages.extend(response.results)

        # Save conversation memory
        if _conversation_memory is not None and (response.calls or response.results):
            serializable = [m.model_dump() for m in messages]
            _conversation_memory.store(serializable)

        return response
