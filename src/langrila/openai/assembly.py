from typing import AsyncGenerator, Callable, Generator, Literal

from openai._types import NOT_GIVEN, NotGiven
from pydantic import BaseModel

from ..base import BaseConversationLengthAdjuster, BaseConversationMemory, BaseFilter
from ..base_assembly import BaseAssembly
from ..message_content import ConversationType, InputType, Message, ToolContent
from ..result import CompletionResults, FunctionCallingResults
from ..tools import ToolConfig
from ..usage import TokenCounter, Usage
from .llm.chat import OpenAIChatModule
from .llm.function_calling import OpenAIFunctionCallingModule


class OpenAIFunctionalChat(BaseAssembly):
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str,
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        organization_id_env_name: str | None = None,
        api_type: str = "openai",
        api_version: str | None = None,
        endpoint_env_name: str | None = None,
        deployment_id_env_name: str | None = None,
        max_tokens: int = 2048,
        timeout: int = 60,
        max_retries: int = 2,
        seed: int | NotGiven = NOT_GIVEN,
        context_length: int | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        system_instruction: str | None = None,
        conversation_length_adjuster: BaseConversationLengthAdjuster | None = None,
        token_counter: TokenCounter | None = None,
        json_mode: bool = False,
        top_p: float | None | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        presence_penalty: float | None | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        response_schema: BaseModel | None = None,
    ) -> None:
        super().__init__(conversation_memory=conversation_memory)

        self.chat = OpenAIChatModule(
            api_key_env_name=api_key_env_name,
            model_name=model_name,
            organization_id_env_name=organization_id_env_name,
            api_type=api_type,
            api_version=api_version,
            endpoint_env_name=endpoint_env_name,
            deployment_id_env_name=deployment_id_env_name,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            seed=seed,
            context_length=context_length,
            conversation_memory=self.conversation_memory,
            content_filter=content_filter,
            system_instruction=system_instruction,
            conversation_length_adjuster=conversation_length_adjuster,
            token_counter=token_counter,
            json_mode=json_mode,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            user=user,
            response_schema=response_schema,
        )

        if tools:
            self.function_calling = OpenAIFunctionCallingModule(
                api_key_env_name=api_key_env_name,
                model_name=model_name,
                tools=tools,
                tool_configs=tool_configs,
                organization_id_env_name=organization_id_env_name,
                api_type=api_type,
                api_version=api_version,
                endpoint_env_name=endpoint_env_name,
                deployment_id_env_name=deployment_id_env_name,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
                seed=seed,
                context_length=context_length,
                conversation_memory=self.conversation_memory,
                content_filter=content_filter,
                system_instruction=system_instruction,
                conversation_length_adjuster=conversation_length_adjuster,
                token_counter=token_counter,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                temperature=temperature,
                user=user,
            )
        else:
            self.function_calling = None

    def run(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "required"] | str | None = None,
        tool_only: bool = False,
        n_results: int | NotGiven = NOT_GIVEN,
    ) -> CompletionResults | FunctionCallingResults:
        if self.function_calling and tool_choice is None:
            tool_choice = "auto"

        if tool_only:
            assert tool_choice is not None, "tool_choice must be provided when tool_only is True"

        total_usage = Usage(model_name=self.chat.chat_model.model_name)

        if self.function_calling and tool_choice:
            response_function_calling: FunctionCallingResults = self.function_calling.run(
                prompt=prompt,
                init_conversation=init_conversation,
                tool_choice=tool_choice,
            )

            if tool_only:
                self._clear_memory()

                return response_function_calling

            if response_function_calling.results:
                prompt = Message(
                    role="function",
                    name=None,
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
            prompt=prompt,
            n_results=n_results,
            init_conversation=init_conversation,
        )

        total_usage += response_chat.usage
        response_chat.usage = total_usage

        self._clear_memory()

        return response_chat

    async def arun(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "required"] | str | None = None,
        tool_only: bool = False,
        n_results: int | NotGiven = NOT_GIVEN,
    ) -> CompletionResults | FunctionCallingResults:
        if self.function_calling and tool_choice is None:
            tool_choice = "auto"

        if tool_only:
            assert tool_choice is not None, "tool_choice must be provided when tool_only is True"

        total_usage = Usage(model_name=self.chat.chat_model.model_name)

        if self.function_calling and tool_choice:
            response_function_calling: FunctionCallingResults = await self.function_calling.arun(
                prompt=prompt,
                init_conversation=init_conversation,
                tool_choice=tool_choice,
            )

            if tool_only:
                self._clear_memory()

                return response_function_calling

            if response_function_calling.results:
                prompt = Message(
                    role="function",
                    name=None,
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
            prompt=prompt,
            n_results=n_results,
            init_conversation=init_conversation,
        )

        total_usage += response_chat.usage
        response_chat.usage = total_usage

        self._clear_memory()

        return response_chat

    def stream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "required"] | str | None = None,
    ) -> Generator[CompletionResults, None, None]:
        if self.function_calling and tool_choice is None:
            tool_choice = "auto"

        total_usage = Usage(model_name=self.chat.chat_model.model_name)

        if self.function_calling and tool_choice:
            response_function_calling: FunctionCallingResults = self.function_calling.run(
                prompt=prompt,
                init_conversation=init_conversation,
                tool_choice=tool_choice,
            )

            if response_function_calling.results:
                prompt = Message(
                    role="function",
                    name=None,
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

        response_chat = self.chat.stream(
            prompt=prompt,
            init_conversation=init_conversation,
        )

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
        tool_choice: Literal["auto", "required"] | str | None = None,
    ) -> AsyncGenerator[CompletionResults, None]:
        if self.function_calling and tool_choice is None:
            tool_choice = "auto"

        total_usage = Usage(model_name=self.chat.chat_model.model_name)

        if self.function_calling and tool_choice:
            response_function_calling: FunctionCallingResults = await self.function_calling.arun(
                prompt=prompt,
                init_conversation=init_conversation,
                tool_choice=tool_choice,
            )

            if response_function_calling.results:
                prompt = Message(
                    role="function",
                    name=None,
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

        response_chat = self.chat.astream(
            prompt=prompt,
            init_conversation=init_conversation,
        )

        async for result in response_chat:
            if result.usage.total_tokens > 0:
                total_usage += result.usage
                result.usage = total_usage
            yield result

        self._clear_memory()
