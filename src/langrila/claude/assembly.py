from typing import Any, Callable, Iterable, Literal, Mapping, Union

import httpx
from anthropic._base_client import DEFAULT_MAX_RETRIES
from anthropic._types import (
    NOT_GIVEN,
    NotGiven,
    ProxiesTypes,
    Timeout,
    Transport,
)
from anthropic.types.text_block_param import TextBlockParam

from ..base import (
    BaseConversationLengthAdjuster,
    BaseConversationMemory,
    BaseFilter,
)
from ..base_assembly import BaseAssembly
from ..message_content import ConversationType, InputType, Message, TextContent, ToolContent
from ..result import CompletionResults, FunctionCallingResults
from ..tools import ToolConfig
from ..usage import TokenCounter, Usage
from .llm.chat import AnthropicChatModule
from .llm.function_calling import AnthropicFunctionCallingModule


class ClaudeFunctionalChat(BaseAssembly):
    """
    FIXME : This module might not work for multi-turn conversation using tools.
    """

    def __init__(
        self,
        model_name: str,
        api_type: str = "anthropic",
        api_key_env_name: str | None = None,
        auth_token_env_name: str | None = None,
        endpoint_env_name: str | httpx.URL | None = None,
        aws_secret_key_env_name: str | None = None,
        aws_access_key_env_name: str | None = None,
        aws_region_env_name: str | None = None,
        aws_session_token_env_name: str | None = None,
        gc_region_env_name: str | NotGiven = NOT_GIVEN,
        gc_project_id_env_name: str | NotGiven = NOT_GIVEN,
        gc_access_token_env_name: str | None = None,
        credentials: Any | None = None,
        timeout: Union[float, Timeout, None, NotGiven] = 600,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.Client | None = None,
        transport: Transport | None = None,
        proxies: ProxiesTypes | None = None,
        connection_pool_limits: httpx.Limits | None = None,
        _strict_response_validation: bool = False,
        conversation_length_adjuster: BaseConversationLengthAdjuster | None = None,
        system_instruction: Union[str, Iterable[TextBlockParam]] | NotGiven = NOT_GIVEN,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        token_counter: TokenCounter | None = None,
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        temperature: float | NotGiven = NOT_GIVEN,
        top_k: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
    ):
        super().__init__(conversation_memory=conversation_memory)

        self.chat = AnthropicChatModule(
            model_name=model_name,
            api_type=api_type,
            api_key_env_name=api_key_env_name,
            auth_token_env_name=auth_token_env_name,
            endpoint_env_name=endpoint_env_name,
            aws_secret_key_env_name=aws_secret_key_env_name,
            aws_access_key_env_name=aws_access_key_env_name,
            aws_region_env_name=aws_region_env_name,
            aws_session_token_env_name=aws_session_token_env_name,
            gc_region_env_name=gc_region_env_name,
            gc_project_id_env_name=gc_project_id_env_name,
            gc_access_token_env_name=gc_access_token_env_name,
            credentials=credentials,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            transport=transport,
            proxies=proxies,
            connection_pool_limits=connection_pool_limits,
            _strict_response_validation=_strict_response_validation,
            conversation_length_adjuster=conversation_length_adjuster,
            system_instruction=system_instruction,
            conversation_memory=self.conversation_memory,
            content_filter=content_filter,
            token_counter=token_counter,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        if tools:
            self.function_calling = AnthropicFunctionCallingModule(
                model_name=model_name,
                api_type=api_type,
                api_key_env_name=api_key_env_name,
                auth_token_env_name=auth_token_env_name,
                endpoint_env_name=endpoint_env_name,
                aws_secret_key_env_name=aws_secret_key_env_name,
                aws_access_key_env_name=aws_access_key_env_name,
                aws_region_env_name=aws_region_env_name,
                aws_session_token_env_name=aws_session_token_env_name,
                gc_region_env_name=gc_region_env_name,
                gc_project_id_env_name=gc_project_id_env_name,
                gc_access_token_env_name=gc_access_token_env_name,
                credentials=credentials,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
                default_query=default_query,
                http_client=http_client,
                transport=transport,
                proxies=proxies,
                connection_pool_limits=connection_pool_limits,
                _strict_response_validation=_strict_response_validation,
                conversation_length_adjuster=conversation_length_adjuster,
                system_instruction=system_instruction,
                conversation_memory=self.conversation_memory,
                content_filter=content_filter,
                token_counter=token_counter,
                tools=tools,
                tool_configs=tool_configs,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        else:
            self.function_calling = None

    def run(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "any"] | str | None = None,
        tool_only: bool = False,
    ) -> CompletionResults | FunctionCallingResults:
        if self.function_calling and tool_choice is None:
            tool_choice = "auto"

        if tool_only:
            assert tool_choice is not None, "tool_choice must be provided when tool_only is True"

        if self.function_calling:
            if tool_choice is not None:
                total_usage = Usage(model_name=self.chat.chat_model.model_name)

                response_function_calling: FunctionCallingResults = self.function_calling.run(
                    prompt,
                    init_conversation=init_conversation,
                    tool_choice=tool_choice,
                )

                if tool_only:
                    self._clear_memory()

                    return response_function_calling

                total_usage += response_function_calling.usage

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

                    response_function_calling: FunctionCallingResults = self.function_calling.run(
                        prompt, init_conversation=init_conversation
                    )

                    total_usage += response_function_calling.usage

                self._clear_memory()

                return CompletionResults(
                    usage=total_usage,
                    prompt=response_function_calling.prompt,
                    message=Message(
                        role="assistant",
                        content=[
                            c
                            for c in response_function_calling.calls.content
                            if isinstance(c, TextContent)
                        ],
                    ),
                )

            else:
                raise ValueError("tool_choice must be provided when the model has tools available")
        else:
            response_chat: CompletionResults = self.chat.run(
                prompt, init_conversation=init_conversation
            )

            self._clear_memory()

            return response_chat

    async def arun(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        tool_choice: Literal["auto", "any"] | str | None = None,
        tool_only: bool = False,
    ) -> CompletionResults | FunctionCallingResults:
        if self.function_calling and tool_choice is None:
            tool_choice = "auto"

        if tool_only:
            assert tool_choice is not None, "tool_choice must be provided when tool_only is True"

        if self.function_calling:
            if tool_choice is not None:
                total_usage = Usage(model_name=self.chat.chat_model.model_name)

                response_function_calling: FunctionCallingResults = (
                    await self.function_calling.arun(
                        prompt,
                        init_conversation=init_conversation,
                        tool_choice=tool_choice,
                    )
                )

                if tool_only:
                    self._clear_memory()

                    return response_function_calling

                total_usage += response_function_calling.usage

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

                    response_function_calling: FunctionCallingResults = (
                        await self.function_calling.arun(
                            prompt, init_conversation=init_conversation
                        )
                    )

                    total_usage += response_function_calling.usage

                self._clear_memory()

                return CompletionResults(
                    usage=total_usage,
                    prompt=response_function_calling.prompt,
                    message=Message(
                        role="assistant",
                        content=[
                            c
                            for c in response_function_calling.calls.content
                            if isinstance(c, TextContent)
                        ],
                    ),
                )
            else:
                raise ValueError("tool_choice must be provided when the model has tools available")
        else:
            response_chat: CompletionResults = await self.chat.arun(
                prompt, init_conversation=init_conversation
            )

            self._clear_memory()

            return response_chat
