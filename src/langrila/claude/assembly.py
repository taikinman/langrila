from typing import Any, Callable, Iterable, Literal, Mapping, Union

import httpx
from anthropic._base_client import DEFAULT_MAX_RETRIES
from anthropic._types import (
    NOT_GIVEN,
    Body,
    Headers,
    NotGiven,
    ProxiesTypes,
    Query,
    Timeout,
    Transport,
)
from anthropic.types import (
    TextBlockParam,
    message_create_params,
)

# from anthropic.types.text_block_param import TextBlockParam
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
        model_name: str | None = None,
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
        timeout: Union[float, Timeout, None, NotGiven] = 60,
        max_tokens: int | NotGiven = 2048,
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
        temperature: float | NotGiven = NOT_GIVEN,
        top_k: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        metadata: message_create_params.Metadata | NotGiven = NOT_GIVEN,
        stop_sequences: list[str] | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        tools: list[Callable] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        tool_choice: Literal["auto", "any"] | str = "auto",
        tool_only: bool = False,
    ):
        self.model_name = model_name
        self.api_type = api_type
        self.api_key_env_name = api_key_env_name
        self.auth_token_env_name = auth_token_env_name
        self.endpoint_env_name = endpoint_env_name
        self.aws_secret_key_env_name = aws_secret_key_env_name
        self.aws_access_key_env_name = aws_access_key_env_name
        self.aws_region_env_name = aws_region_env_name
        self.aws_session_token_env_name = aws_session_token_env_name
        self.gc_region_env_name = gc_region_env_name
        self.gc_project_id_env_name = gc_project_id_env_name
        self.gc_access_token_env_name = gc_access_token_env_name
        self.credentials = credentials
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.default_headers = default_headers
        self.default_query = default_query
        self.http_client = http_client
        self.transport = transport
        self.proxies = proxies
        self.connection_pool_limits = connection_pool_limits
        self._strict_response_validation = _strict_response_validation
        self.conversation_length_adjuster = conversation_length_adjuster
        self.system_instruction = system_instruction
        self.conversation_memory = conversation_memory
        self.content_filter = content_filter
        self.token_counter = token_counter
        self.tools = tools
        self.tool_configs = tool_configs
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.metadata = metadata
        self.stop_sequences = stop_sequences
        self.extra_headers = extra_headers
        self.extra_query = extra_query
        self.extra_body = extra_body
        self.tool_choice = tool_choice
        self.tool_only = tool_only

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
            max_tokens=max_tokens,
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
            conversation_memory=conversation_memory,
            content_filter=content_filter,
            token_counter=token_counter,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            metadata=metadata,
            stop_sequences=stop_sequences,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
        )

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
            max_tokens=max_tokens,
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
            conversation_memory=conversation_memory,
            content_filter=content_filter,
            token_counter=token_counter,
            tools=tools,
            tool_configs=tool_configs,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            metadata=metadata,
            stop_sequences=stop_sequences,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            tool_choice=tool_choice,
        )

    def _get_generation_kwargs(self, **kwargs) -> dict[str, Any]:
        _kwargs = {}
        _kwargs["model_name"] = kwargs.get("model_name") or self.model_name
        _kwargs["max_tokens"] = kwargs.get("max_tokens") or self.max_tokens
        _kwargs["metadata"] = kwargs.get("metadata") or self.metadata
        _kwargs["stop_sequences"] = kwargs.get("stop_sequences") or self.stop_sequences
        _kwargs["system_instruction"] = kwargs.get("system_instruction") or self.system_instruction
        _kwargs["temperature"] = kwargs.get("temperature") or self.temperature
        _kwargs["top_k"] = kwargs.get("top_k") or self.top_k
        _kwargs["top_p"] = kwargs.get("top_p") or self.top_p
        _kwargs["extra_headers"] = kwargs.get("extra_headers") or self.extra_headers
        _kwargs["extra_query"] = kwargs.get("extra_query") or self.extra_query
        _kwargs["extra_body"] = kwargs.get("extra_body") or self.extra_body
        _kwargs["timeout"] = kwargs.get("timeout") or self.timeout
        _kwargs["tool_choice"] = kwargs.get("tool_choice") or self.tool_choice
        _kwargs["tools"] = kwargs.get("tools") or self.tools
        _kwargs["tool_configs"] = kwargs.get("tool_configs") or self.tool_configs

        return _kwargs

    def run(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        model_name: str | None = None,
        max_tokens: int | NotGiven = 2048,
        metadata: message_create_params.Metadata | NotGiven = NOT_GIVEN,
        stop_sequences: list[str] | NotGiven = NOT_GIVEN,
        system_instruction: Union[str, Iterable[TextBlockParam]] | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = 60,
        temperature: float | NotGiven = NOT_GIVEN,
        top_k: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        tools: list[Callable] | NotGiven = NOT_GIVEN,
        tool_configs: list[ToolConfig] | NotGiven = NOT_GIVEN,
        tool_choice: Literal["auto", "any"] | str = "auto",
        tool_only: bool = False,
    ) -> CompletionResults | FunctionCallingResults:
        _conversation_memory = self._setup_memory(conversation_memory or self.conversation_memory)

        generation_kwargs = self._get_generation_kwargs(
            model_name=model_name,
            max_tokens=max_tokens,
            metadata=metadata,
            stop_sequences=stop_sequences,
            system_instruction=system_instruction,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            tools=tools,
            tool_configs=tool_configs,
            tool_choice=tool_choice,
        )

        if generation_kwargs.get("tools"):
            if tool_choice is not None:
                total_usage = Usage(model_name=generation_kwargs.get("model_name"))

                response_function_calling: FunctionCallingResults = self.function_calling.run(
                    prompt,
                    init_conversation=init_conversation,
                    conversation_memory=_conversation_memory,
                    content_filter=content_filter,
                    **generation_kwargs,
                )

                if tool_only or self.tool_only:
                    self._clear_memory(_conversation_memory)

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
                        prompt,
                        init_conversation=init_conversation,
                        conversation_memory=_conversation_memory,
                        content_filter=content_filter,
                        **generation_kwargs,
                    )

                    total_usage += response_function_calling.usage

                self._clear_memory(_conversation_memory)

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
                prompt,
                init_conversation=init_conversation,
                conversation_memory=_conversation_memory,
                content_filter=content_filter,
                **generation_kwargs,
            )

            self._clear_memory(_conversation_memory)

            return response_chat

    async def arun(
        self,
        prompt: InputType,
        init_conversation: ConversationType | None = None,
        conversation_memory: BaseConversationMemory | None = None,
        content_filter: BaseFilter | None = None,
        model_name: str | None = None,
        max_tokens: int | NotGiven = 2048,
        metadata: message_create_params.Metadata | NotGiven = NOT_GIVEN,
        stop_sequences: list[str] | NotGiven = NOT_GIVEN,
        system_instruction: Union[str, Iterable[TextBlockParam]] | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = 60,
        temperature: float | NotGiven = NOT_GIVEN,
        top_k: int | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        tools: list[Callable] | NotGiven = NOT_GIVEN,
        tool_configs: list[ToolConfig] | NotGiven = NOT_GIVEN,
        tool_choice: Literal["auto", "any"] | str = "auto",
        tool_only: bool = False,
    ) -> CompletionResults | FunctionCallingResults:
        _conversation_memory = self._setup_memory(conversation_memory or self.conversation_memory)

        generation_kwargs = self._get_generation_kwargs(
            model_name=model_name,
            max_tokens=max_tokens,
            metadata=metadata,
            stop_sequences=stop_sequences,
            system_instruction=system_instruction,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            tools=tools,
            tool_configs=tool_configs,
            tool_choice=tool_choice,
        )

        if generation_kwargs.get("tools"):
            if tool_choice is not None:
                total_usage = Usage(model_name=generation_kwargs.get("model_name"))

                response_function_calling: FunctionCallingResults = (
                    await self.function_calling.arun(
                        prompt,
                        init_conversation=init_conversation,
                        conversation_memory=_conversation_memory,
                        content_filter=content_filter,
                        **generation_kwargs,
                    )
                )

                if tool_only or self.tool_only:
                    self._clear_memory(_conversation_memory)

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
                            prompt,
                            init_conversation=init_conversation,
                            conversation_memory=_conversation_memory,
                            content_filter=content_filter,
                            **generation_kwargs,
                        )
                    )

                    total_usage += response_function_calling.usage

                self._clear_memory(_conversation_memory)

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
                prompt,
                init_conversation=init_conversation,
                conversation_memory=_conversation_memory,
                content_filter=content_filter,
                **generation_kwargs,
            )

            self._clear_memory(_conversation_memory)

            return response_chat
