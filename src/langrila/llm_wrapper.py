from abc import ABC, abstractmethod
from typing import AsyncGenerator, Generator, Optional

from .base import (
    BaseChatModule,
    BaseConversationMemory,
    BaseFilter,
    BaseFunctionCallingModule,
    BaseMessage,
)
from .message_content import ConversationType, InputType, Message
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
        **kwargs,
    ) -> CompletionResults:
        ClientMessage = self._get_client_message_type()

        # Load conversation memory
        messages = self.load_conversation()

        # Convert init_conversation to ClientMessage
        if init_conversation:
            if not isinstance(init_conversation, list):
                init_conversation = [init_conversation]

            messages.extend(init_conversation)

        # Convert messages to UniversalMessage
        messages = [ClientMessage.to_universal_message(message=message) for message in messages]

        # Convert prompt to UniversalMessage
        messages.append(ClientMessage.to_universal_message(message=prompt, role="user"))

        # Convert UniversalMessage to ClientMessage
        _messages = [
            ClientMessage.to_client_message(message=message)
            for message in ClientMessage._preprocess_message(messages)
        ]

        # Apply content filter if available
        if self.content_filter is not None:
            _messages = self.content_filter.apply(_messages)

        response: CompletionResults = self.chat_model.run(_messages, **kwargs)

        # Update total tokens
        if self.token_counter is not None:
            self.token_counter += response.usage

        # Restore content filter if available
        if self.content_filter is not None:
            response.message = self.restore_content_filter([response.message])[0]

        # Append the response to the conversation with the UniversalMessage format
        response.message = ClientMessage.to_universal_message_from_completion_response(
            response=response
        )
        messages.append(response.message)

        # Save conversation memory
        if self.conversation_memory is not None:
            serializable = [m.model_dump() for m in messages]
            self.save_conversation(serializable)

        return response

    async def arun(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        **kwargs,
    ) -> CompletionResults:
        ClientMessage = self._get_client_message_type()

        # Load conversation memory
        messages = self.load_conversation()

        # Convert init_conversation to ClientMessage
        if init_conversation:
            if not isinstance(init_conversation, list):
                init_conversation = [init_conversation]

            messages.extend(init_conversation)

        # Convert messages to UniversalMessage
        messages = [ClientMessage.to_universal_message(message=message) for message in messages]

        # Convert prompt to UniversalMessage
        messages.append(ClientMessage.to_universal_message(message=prompt, role="user"))

        # Convert UniversalMessage to ClientMessage
        _messages = [
            ClientMessage.to_client_message(message=message)
            for message in ClientMessage._preprocess_message(messages)
        ]

        # Apply content filter if available
        if self.content_filter is not None:
            _messages = self.content_filter.apply(_messages)

        response: CompletionResults = await self.chat_model.arun(_messages, **kwargs)

        # Update total tokens
        if self.token_counter is not None:
            self.token_counter += response.usage

        # Restore content filter if available
        if self.content_filter is not None:
            response.message = self.restore_content_filter([response.message])[0]

        # Append the response to the conversation with the UniversalMessage format
        response.message = ClientMessage.to_universal_message_from_completion_response(
            response=response
        )
        messages.append(response.message)

        # Save conversation memory
        if self.conversation_memory is not None:
            serializable = [m.model_dump() for m in messages]
            self.save_conversation(serializable)

        return response

    def stream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
    ) -> Generator[CompletionResults, None, None]:
        ClientMessage = self._get_client_message_type()

        # Load conversation memory
        messages = self.load_conversation()

        # Convert init_conversation to ClientMessage
        if init_conversation:
            if not isinstance(init_conversation, list):
                init_conversation = [init_conversation]

            messages.extend(init_conversation)

        # Convert messages to UniversalMessage
        messages = [ClientMessage.to_universal_message(message=message) for message in messages]

        # Convert prompt to UniversalMessage
        messages.append(ClientMessage.to_universal_message(message=prompt, role="user"))

        # Convert UniversalMessage to ClientMessage
        _messages = [
            ClientMessage.to_client_message(message=message)
            for message in ClientMessage._preprocess_message(messages)
        ]

        # Apply content filter if available
        if self.content_filter is not None:
            _messages = self.content_filter.apply(_messages)

        # Run the model
        response = self.chat_model.stream(_messages)

        for chunk in response:
            if isinstance(chunk, CompletionResults):
                if self.content_filter is not None:
                    chunk.message = self.restore_content_filter([chunk.message])[0]

                # Convert to UniversalMessage
                chunk.message = ClientMessage.to_universal_message_from_completion_response(
                    response=chunk
                )

                yield chunk
            else:
                raise AssertionError

        # Update total tokens
        if self.token_counter is not None:
            self.token_counter += chunk.usage

        messages.append(chunk.message)

        # Save conversation memory
        if self.conversation_memory is not None:
            serializable = [m.model_dump() for m in messages]
            self.save_conversation(serializable)

    async def astream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
    ) -> AsyncGenerator[CompletionResults, None]:
        ClientMessage = self._get_client_message_type()

        # Load conversation memory
        messages = self.load_conversation()

        # Convert init_conversation to ClientMessage
        if init_conversation:
            if not isinstance(init_conversation, list):
                init_conversation = [init_conversation]

            messages.extend(init_conversation)

        # Convert messages to UniversalMessage
        messages = [ClientMessage.to_universal_message(message=message) for message in messages]

        # Convert prompt to UniversalMessage
        messages.append(ClientMessage.to_universal_message(message=prompt, role="user"))

        # Convert UniversalMessage to ClientMessage
        _messages = [
            ClientMessage.to_client_message(message=message)
            for message in ClientMessage._preprocess_message(messages)
        ]

        # Apply content filter if available
        if self.content_filter is not None:
            _messages = self.content_filter.apply(_messages)

        # Run the model
        response = self.chat_model.astream(_messages)

        async for chunk in response:
            if isinstance(chunk, CompletionResults):
                if self.content_filter is not None:
                    chunk.message = self.restore_content_filter([chunk.message])[0]

                # Convert to UniversalMessage
                chunk.message = ClientMessage.to_universal_message_from_completion_response(
                    response=chunk
                )

                yield chunk
            else:
                raise AssertionError

        # Update total tokens
        if self.token_counter is not None:
            self.token_counter += chunk.usage

        messages.append(chunk.message)

        # Save conversation memory
        if self.conversation_memory is not None:
            serializable = [m.model_dump() for m in messages]
            self.save_conversation(serializable)


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
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        **kwargs,
    ) -> FunctionCallingResults:
        ClientMessage = self._get_client_message_type()

        # Load conversation memory
        messages = self.load_conversation()

        # Convert init_conversation to ClientMessage
        if init_conversation:
            if not isinstance(init_conversation, list):
                init_conversation = [init_conversation]

            messages.extend(init_conversation)

        # Convert messages to UniversalMessage
        messages = [ClientMessage.to_universal_message(message=message) for message in messages]

        # Convert prompt to UniversalMessage
        messages.append(ClientMessage.to_universal_message(message=prompt, role="user"))

        # Convert UniversalMessage to ClientMessage
        _messages = [
            ClientMessage.to_client_message(message=message)
            for message in ClientMessage._preprocess_message(messages)
        ]

        # Apply content filter if available
        if self.content_filter is not None:
            _messages = self.content_filter.apply(_messages)

        # Run the model
        response = self.function_calling_model.run(_messages, **kwargs)

        # Update total tokens
        if self.token_counter is not None:
            self.token_counter += response.usage

        # Restore content filter if available
        if self.content_filter is not None:
            for i, _ in enumerate(response.results):
                response.results[i].args = self.restore_content_filter([response.results[i].args])[
                    0
                ]

        # Append the response to the conversation
        response.results = ClientMessage.to_universal_message_from_function_response(
            response=response
        )

        if response.calls:
            response.calls = ClientMessage.to_universal_message_from_function_call(
                response=response
            )
            messages.append(response.calls)
        else:
            messages.extend(response.results)

        # Save conversation memory
        if self.conversation_memory is not None and (response.calls or response.results):
            serializable = [m.model_dump() for m in messages]
            self.save_conversation(serializable)

        return response

    async def arun(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        **kwargs,
    ) -> FunctionCallingResults:
        ClientMessage = self._get_client_message_type()

        # Load conversation memory
        messages = self.load_conversation()

        # Convert init_conversation to ClientMessage
        if init_conversation:
            if not isinstance(init_conversation, list):
                init_conversation = [init_conversation]

            messages.extend(init_conversation)

        # Convert messages to UniversalMessage
        messages = [ClientMessage.to_universal_message(message=message) for message in messages]

        # Convert prompt to UniversalMessage
        messages.append(ClientMessage.to_universal_message(message=prompt, role="user"))

        # Convert UniversalMessage to ClientMessage
        _messages = [
            ClientMessage.to_client_message(message=message)
            for message in ClientMessage._preprocess_message(messages)
        ]

        # Apply content filter if available
        if self.content_filter is not None:
            _messages = self.content_filter.apply(_messages)

        # Run the model
        response = await self.function_calling_model.arun(_messages, **kwargs)

        # Update total tokens
        if self.token_counter is not None:
            self.token_counter += response.usage

        # Restore content filter if available
        if self.content_filter is not None:
            for i, _ in enumerate(response.results):
                response.results[i].args = self.restore_content_filter([response.results[i].args])[
                    0
                ]

        # Append the response to the conversation
        response.results = ClientMessage.to_universal_message_from_function_response(
            response=response
        )

        if response.calls:
            response.calls = ClientMessage.to_universal_message_from_function_call(
                response=response
            )
            messages.append(response.calls)
        else:
            messages.extend(response.results)

        # Save conversation memory
        if self.conversation_memory is not None and (response.calls or response.results):
            serializable = [m.model_dump() for m in messages]
            self.save_conversation(serializable)

        return response
