from typing import AsyncGenerator, Callable, Generator, Literal

from ..base import BaseConversationLengthAdjuster, BaseConversationMemory, BaseFilter
from ..message_content import ConversationType, InputType, Message
from ..result import CompletionResults, FunctionCallingResults
from ..usage import TokenCounter
from .llm.chat import OpenAIChatModule
from .llm.function_calling import OpenAIFunctionCallingModule, ToolConfig


class ChatGPT:
    def __init__(
        self,
        api_key_env_name: str,
        model_name: str,
        tools: list[Callable],
        tool_configs: list[ToolConfig],
        organization_id_env_name: str | None = None,
        api_type: str = "openai",
        api_version: str | None = None,
        endpoint_env_name: str | None = None,
        deployment_id_env_name: str | None = None,
        max_tokens: int = 2048,
        timeout: int = 60,
        max_retries: int = 2,
        seed: int | None = None,
        context_length: int | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        system_instruction: str | None = None,
        conversation_length_adjuster: BaseConversationLengthAdjuster | None = None,
        token_counter: TokenCounter | None = None,
        response_format: dict[str, str] | None = None,
    ) -> None:
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
            conversation_memory=conversation_memory,
            content_filter=content_filter,
            system_instruction=system_instruction,
            conversation_length_adjuster=conversation_length_adjuster,
            token_counter=token_counter,
            response_format=response_format,
        )

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
            conversation_memory=conversation_memory,
            content_filter=content_filter,
            system_instruction=system_instruction,
            conversation_length_adjuster=conversation_length_adjuster,
            token_counter=token_counter,
        )

        self.conversation_memory = conversation_memory

    def run(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "required"] | str | None = None,
        tool_only: bool = False,
    ) -> CompletionResults | FunctionCallingResults:
        if tool_only:
            assert tool_choice is not None, "tool_choice must be provided when tool_only is True"

        if tool_choice is not None:
            response_function_calling: FunctionCallingResults = self.function_calling.run(
                prompt=prompt,
                init_conversation=init_conversation,
                tool_choice=tool_choice,
            )

            if tool_only:
                return response_function_calling

            prompt = [
                Message(role="function", content=result.content)
                for result in response_function_calling.results
            ]

        response_chat: CompletionResults = self.chat.run(
            prompt=prompt,
            gather_prompts=False if tool_choice is not None else True,
        )

        return response_chat

    async def arun(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "required"] | str | None = None,
        tool_only: bool = False,
    ) -> CompletionResults | FunctionCallingResults:
        if tool_only:
            assert tool_choice is not None, "tool_choice must be provided when tool_only is True"

        if tool_choice is not None:
            response_function_calling: FunctionCallingResults = await self.function_calling.arun(
                prompt=prompt,
                init_conversation=init_conversation,
                tool_choice=tool_choice,
            )

            if tool_only:
                return response_function_calling

            prompt = [
                Message(role="function", content=result.content)
                for result in response_function_calling.results
            ]

        response_chat: CompletionResults = await self.chat.arun(
            prompt=prompt,
            gather_prompts=False if tool_choice is not None else True,
        )

        return response_chat

    def stream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "required"] | str | None = None,
    ) -> Generator[CompletionResults, None, None]:
        if tool_choice is not None:
            response_function_calling: FunctionCallingResults = self.function_calling.run(
                prompt=prompt,
                init_conversation=init_conversation,
                tool_choice=tool_choice,
            )

            prompt = [
                Message(role="function", content=result.content)
                for result in response_function_calling.results
            ]

        response_chat: CompletionResults = self.chat.stream(
            prompt=prompt,
            gather_prompts=False if tool_choice is not None else True,
        )

        return response_chat

    async def astream(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "required"] | str | None = None,
    ) -> Generator[CompletionResults, None, None]:
        if tool_choice is not None:
            response_function_calling: FunctionCallingResults = await self.function_calling.arun(
                prompt=prompt,
                init_conversation=init_conversation,
                tool_choice=tool_choice,
            )

            prompt = [
                Message(role="function", content=result.content)
                for result in response_function_calling.results
            ]

        response_chat: CompletionResults = self.chat.astream(
            prompt=prompt,
            gather_prompts=False if tool_choice is not None else True,
        )

        return response_chat
