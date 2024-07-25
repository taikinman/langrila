from typing import Any, Iterable, Mapping, Union

import httpx
from anthropic import Anthropic, AnthropicBedrock, AsyncAnthropic, AsyncAnthropicBedrock
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
from anthropic.types import message_create_params
from anthropic.types.message_param import MessageParam
from anthropic.types.text_block_param import TextBlockParam
from anthropic.types.tool_param import ToolParam
from typing_extensions import Literal


def get_client(
    api_type: str,
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
    timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
    max_retries: int = DEFAULT_MAX_RETRIES,
    default_headers: Mapping[str, str] | None = None,
    default_query: Mapping[str, object] | None = None,
    http_client: httpx.Client | None = None,
    transport: Transport | None = None,
    proxies: ProxiesTypes | None = None,
    connection_pool_limits: httpx.Limits | None = None,
    _strict_response_validation: bool = False,
):
    if api_type == "anthropic":
        from .anthropic.client import get_anthropic_client

        return get_anthropic_client(
            api_key_env_name=api_key_env_name,
            auth_token_env_name=auth_token_env_name,
            endpoint_env_name=endpoint_env_name,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            transport=transport,
            proxies=proxies,
            connection_pool_limits=connection_pool_limits,
            _strict_response_validation=_strict_response_validation,
        )
    elif api_type == "bedrock":
        from .bedrock.client import get_bedrock_client

        return get_bedrock_client(
            aws_secret_key_env_name=aws_secret_key_env_name,
            aws_access_key_env_name=aws_access_key_env_name,
            aws_region_env_name=aws_region_env_name,
            aws_session_token_env_name=aws_session_token_env_name,
            endpoint_env_name=endpoint_env_name,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
        )
    elif api_type == "vertexai":
        from .vertexai.client import get_vertexai_client

        return get_vertexai_client(
            gc_region_env_name=gc_region_env_name,
            gc_project_id_env_name=gc_project_id_env_name,
            gc_access_token_env_name=gc_access_token_env_name,
            credentials=credentials,
            endpoint_env_name=endpoint_env_name,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            transport=transport,
            proxies=proxies,
            connection_pool_limits=connection_pool_limits,
            _strict_response_validation=_strict_response_validation,
        )

    else:
        raise ValueError(f"Unknown API type: {api_type}")


def get_async_client(
    api_type: str,
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
    timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
    max_retries: int = DEFAULT_MAX_RETRIES,
    default_headers: Mapping[str, str] | None = None,
    default_query: Mapping[str, object] | None = None,
    http_client: httpx.Client | None = None,
    transport: Transport | None = None,
    proxies: ProxiesTypes | None = None,
    connection_pool_limits: httpx.Limits | None = None,
    _strict_response_validation: bool = False,
):
    if api_type == "anthropic":
        from .anthropic.client import get_async_anthropic_client

        return get_async_anthropic_client(
            api_key_env_name=api_key_env_name,
            auth_token_env_name=auth_token_env_name,
            endpoint_env_name=endpoint_env_name,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            transport=transport,
            proxies=proxies,
            connection_pool_limits=connection_pool_limits,
            _strict_response_validation=_strict_response_validation,
        )
    elif api_type == "bedrock":
        from .bedrock.client import get_async_bedrock_client

        return get_async_bedrock_client(
            aws_secret_key_env_name=aws_secret_key_env_name,
            aws_access_key_env_name=aws_access_key_env_name,
            aws_region_env_name=aws_region_env_name,
            aws_session_token_env_name=aws_session_token_env_name,
            endpoint_env_name=endpoint_env_name,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
        )
    elif api_type == "vertexai":
        from .vertexai.client import get_async_vertexai_client

        return get_async_vertexai_client(
            gc_region_env_name=gc_region_env_name,
            gc_project_id_env_name=gc_project_id_env_name,
            gc_access_token_env_name=gc_access_token_env_name,
            credentials=credentials,
            endpoint_env_name=endpoint_env_name,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            transport=transport,
            proxies=proxies,
            connection_pool_limits=connection_pool_limits,
            _strict_response_validation=_strict_response_validation,
        )
    else:
        raise ValueError(f"Unknown API type: {api_type}")


def completion(
    client: Anthropic | AnthropicBedrock,
    model_name: Union[
        str,
        Literal[
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ],
    ],
    messages: Iterable[MessageParam],
    max_tokens: int = 2048,
    metadata: message_create_params.Metadata | NotGiven = NOT_GIVEN,
    stop_sequences: list[str] | NotGiven = NOT_GIVEN,
    stream: Literal[False] | Literal[True] | NotGiven = NOT_GIVEN,
    system_instruction: Union[str, Iterable[TextBlockParam]] | NotGiven = NOT_GIVEN,
    tool_choice: message_create_params.ToolChoice | NotGiven = NOT_GIVEN,
    tools: Iterable[ToolParam] | NotGiven = NOT_GIVEN,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = 600,
    temperature: float | NotGiven = NOT_GIVEN,
    top_k: int | NotGiven = NOT_GIVEN,
    top_p: float | NotGiven = NOT_GIVEN,
):
    return client.messages.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        metadata=metadata,
        stop_sequences=stop_sequences,
        stream=stream,
        system=system_instruction,
        temperature=temperature,
        tool_choice=tool_choice,
        tools=tools,
        top_p=top_p,
        top_k=top_k,
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body,
        timeout=timeout,
    )


async def acompletion(
    client: AsyncAnthropic | AsyncAnthropicBedrock,
    model_name: Union[
        str,
        Literal[
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ],
    ],
    messages: Iterable[MessageParam],
    max_tokens: int = 2048,
    metadata: message_create_params.Metadata | NotGiven = NOT_GIVEN,
    stop_sequences: list[str] | NotGiven = NOT_GIVEN,
    stream: Literal[False] | Literal[True] | NotGiven = NOT_GIVEN,
    system_instruction: Union[str, Iterable[TextBlockParam]] | NotGiven = NOT_GIVEN,
    tool_choice: message_create_params.ToolChoice | NotGiven = NOT_GIVEN,
    tools: Iterable[ToolParam] | NotGiven = NOT_GIVEN,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = 600,
    temperature: float | NotGiven = NOT_GIVEN,
    top_k: int | NotGiven = NOT_GIVEN,
    top_p: float | NotGiven = NOT_GIVEN,
):
    return await client.messages.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        metadata=metadata,
        stop_sequences=stop_sequences,
        stream=stream,
        system=system_instruction,
        temperature=temperature,
        tool_choice=tool_choice,
        tools=tools,
        top_p=top_p,
        top_k=top_k,
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body,
        timeout=timeout,
    )
