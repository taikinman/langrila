from typing import Any, AsyncGenerator, Callable, Generator, Literal, Sequence

from google.auth import credentials as auth_credentials

from ..base import (
    BaseConversationMemory,
    BaseFilter,
)
from ..base_assembly import BaseAssembly
from ..message_content import ConversationType, InputType, Message, ToolContent
from ..result import CompletionResults, FunctionCallingResults
from ..tools import ToolConfig
from ..usage import TokenCounter, Usage
from .llm.chat import GeminiChatModule
from .llm.function_calling import GeminiFunctionCallingModule


class GeminiFunctionalChat(BaseAssembly):
    def __init__(
        self,
        model_name: str,
        api_key_env_name: str | None = None,
        max_tokens: int = 2048,
        json_mode: bool = False,
        timeout: int = 60,
        content_filter: BaseFilter | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        system_instruction: str | None = None,
        token_counter: TokenCounter | None = None,
        api_type: str = "genai",
        project_id_env_name: str | None = None,
        location_env_name: str | None = None,
        experiment: str | None = None,
        experiment_description: str | None = None,
        experiment_tensorboard: str | bool | None = None,
        staging_bucket: str | None = None,
        credentials: auth_credentials.Credentials | None = None,
        encryption_spec_key_name: str | None = None,
        network: str | None = None,
        service_account: str | None = None,
        endpoint_env_name: str | None = None,
        request_metadata: Sequence[tuple[str, str]] | None = None,
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        response_schema: dict[str, Any] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ):
        super().__init__(conversation_memory=conversation_memory)

        self.chat = GeminiChatModule(
            model_name=model_name,
            api_key_env_name=api_key_env_name,
            max_tokens=max_tokens,
            json_mode=json_mode,
            timeout=timeout,
            content_filter=content_filter,
            conversation_memory=self.conversation_memory,
            system_instruction=system_instruction,
            token_counter=token_counter,
            api_type=api_type,
            project_id_env_name=project_id_env_name,
            location_env_name=location_env_name,
            experiment=experiment,
            experiment_description=experiment_description,
            experiment_tensorboard=experiment_tensorboard,
            staging_bucket=staging_bucket,
            credentials=credentials,
            encryption_spec_key_name=encryption_spec_key_name,
            network=network,
            service_account=service_account,
            endpoint_env_name=endpoint_env_name,
            request_metadata=request_metadata,
            response_schema=response_schema,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        if tools:
            self.function_calling = GeminiFunctionCallingModule(
                model_name=model_name,
                api_key_env_name=api_key_env_name,
                max_tokens=max_tokens,
                json_mode=json_mode,
                timeout=timeout,
                content_filter=content_filter,
                conversation_memory=self.conversation_memory,
                system_instruction=system_instruction,
                token_counter=token_counter,
                api_type=api_type,
                project_id_env_name=project_id_env_name,
                location_env_name=location_env_name,
                experiment=experiment,
                experiment_description=experiment_description,
                experiment_tensorboard=experiment_tensorboard,
                staging_bucket=staging_bucket,
                credentials=credentials,
                encryption_spec_key_name=encryption_spec_key_name,
                network=network,
                service_account=service_account,
                endpoint_env_name=endpoint_env_name,
                request_metadata=request_metadata,
                tools=tools,
                tool_configs=tool_configs,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        else:
            self.function_calling = None

    def run(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "any"] | str | None = None,
        tool_only: bool = False,
        n_results: int | None = None,
    ) -> CompletionResults | FunctionCallingResults:
        if self.function_calling and tool_choice is None:
            tool_choice = "auto"

        if tool_only:
            assert tool_choice is not None, "tool_choice must be provided when tool_only is True"

        total_usage = Usage(model_name=self.chat.chat_model.model_name)

        if self.function_calling and tool_choice:
            response_function_calling: FunctionCallingResults = self.function_calling.run(
                prompt,
                init_conversation=init_conversation,
                tool_choice=tool_choice,
            )

            if tool_only:
                self._clear_memory()

                return response_function_calling

            if response_function_calling.results:
                prompt = Message(
                    role="function",
                    content=[
                        ToolContent(**content.model_dump())
                        for result in response_function_calling.results
                        for content in result.content
                    ],
                )
                init_conversation = (
                    None  # if tool is used, init_conversation is stored in the memory
                )

            total_usage += response_function_calling.usage

        response_chat: CompletionResults = self.chat.run(
            prompt, init_conversation=init_conversation, n_results=n_results
        )

        total_usage += response_chat.usage
        response_chat.usage = total_usage

        self._clear_memory()

        return response_chat

    async def arun(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "any"] | str | None = None,
        tool_only: bool = False,
        n_results: int | None = None,
    ) -> CompletionResults | FunctionCallingResults:
        if self.function_calling and tool_choice is None:
            tool_choice = "auto"

        if tool_only:
            assert tool_choice is not None, "tool_choice must be provided when tool_only is True"

        total_usage = Usage(model_name=self.chat.chat_model.model_name)

        if self.function_calling and tool_choice:
            response_function_calling: FunctionCallingResults = await self.function_calling.arun(
                prompt,
                init_conversation=init_conversation,
                tool_choice=tool_choice,
            )

            if tool_only:
                self._clear_memory()

                return response_function_calling

            if response_function_calling.results:
                prompt = Message(
                    role="function",
                    content=[
                        ToolContent(**content.model_dump())
                        for result in response_function_calling.results
                        for content in result.content
                    ],
                )
                init_conversation = (
                    None  # if tool is used, init_conversation is stored in the memory
                )

            total_usage += response_function_calling.usage

        response_chat: CompletionResults = await self.chat.arun(
            prompt, init_conversation=init_conversation, n_results=n_results
        )

        total_usage += response_chat.usage
        response_chat.usage = total_usage

        self._clear_memory()

        return response_chat

    def stream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "any"] | str | None = None,
    ) -> Generator[CompletionResults, None, None]:
        if self.function_calling and tool_choice is None:
            tool_choice = "auto"

        total_usage = Usage(model_name=self.chat.chat_model.model_name)

        if self.function_calling and tool_choice:
            response_function_calling: FunctionCallingResults = self.function_calling.run(
                prompt,
                init_conversation=init_conversation,
                tool_choice=tool_choice,
            )

            if response_function_calling.results:
                prompt = Message(
                    role="function",
                    content=[
                        ToolContent(**content.model_dump())
                        for result in response_function_calling.results
                        for content in result.content
                    ],
                )
                init_conversation = (
                    None  # if tool is used, init_conversation is stored in the memory
                )

            total_usage += response_function_calling.usage

        response_chat = self.chat.stream(prompt, init_conversation=init_conversation)

        for result in response_chat:
            if result.usage.total_tokens > 0:
                total_usage += result.usage
                result.usage = total_usage
            yield result

        self._clear_memory()

    async def astream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "any"] | str | None = None,
    ) -> AsyncGenerator[CompletionResults, None]:
        if self.function_calling and tool_choice is None:
            tool_choice = "auto"

        total_usage = Usage(model_name=self.chat.chat_model.model_name)

        if self.function_calling and tool_choice:
            response_function_calling: FunctionCallingResults = await self.function_calling.arun(
                prompt,
                init_conversation=init_conversation,
                tool_choice=tool_choice,
            )

            if response_function_calling.results:
                prompt = Message(
                    role="function",
                    content=[
                        ToolContent(**content.model_dump())
                        for result in response_function_calling.results
                        for content in result.content
                    ],
                )
                init_conversation = (
                    None  # if tool is used, init_conversation is stored in the memory
                )

            total_usage += response_function_calling.usage

        response_chat = self.chat.astream(prompt, init_conversation=init_conversation)

        async for result in response_chat:
            if result.usage.total_tokens > 0:
                total_usage += result.usage
                result.usage = total_usage
            yield result

        self._clear_memory()
