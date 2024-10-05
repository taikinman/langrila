import os
from typing import Mapping, Union

import httpx
from anthropic import Anthropic, AsyncAnthropic
from anthropic._base_client import DEFAULT_MAX_RETRIES
from anthropic._streaming import AsyncStream, Stream
from anthropic._types import (
    NOT_GIVEN,
    NotGiven,
    ProxiesTypes,
    Timeout,
    Transport,
)
from anthropic.types import Message, RawMessageStreamEvent
from typing_extensions import override

from ...base import BaseClient
from ...utils import create_parameters


def get_anthropic_client(
    api_key_env_name: str | None = None,
    auth_token_env_name: str | None = None,
    endpoint_env_name: str | httpx.URL | None = None,
    timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
    max_retries: int = DEFAULT_MAX_RETRIES,
    default_headers: Mapping[str, str] | None = None,
    default_query: Mapping[str, object] | None = None,
    http_client: httpx.Client | None = None,
    transport: Transport | None = None,
    proxies: ProxiesTypes | None = None,
    connection_pool_limits: httpx.Limits | None = None,
    _strict_response_validation: bool = False,
) -> Anthropic:
    return Anthropic(
        api_key=os.getenv(api_key_env_name) if api_key_env_name else None,
        auth_token=os.getenv(auth_token_env_name) if auth_token_env_name else None,
        base_url=os.getenv(endpoint_env_name) if endpoint_env_name else None,
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


def get_async_anthropic_client(
    api_key_env_name: str | None = None,
    auth_token_env_name: str | None = None,
    endpoint_env_name: str | httpx.URL | None = None,
    timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
    max_retries: int = DEFAULT_MAX_RETRIES,
    default_headers: Mapping[str, str] | None = None,
    default_query: Mapping[str, object] | None = None,
    http_client: httpx.Client | None = None,
    transport: Transport | None = None,
    proxies: ProxiesTypes | None = None,
    connection_pool_limits: httpx.Limits | None = None,
    _strict_response_validation: bool = False,
) -> AsyncAnthropic:
    return AsyncAnthropic(
        api_key=os.getenv(api_key_env_name) if api_key_env_name else None,
        auth_token=os.getenv(auth_token_env_name) if auth_token_env_name else None,
        base_url=os.getenv(endpoint_env_name) if endpoint_env_name else None,
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


class ClaudeAnthropicChat(BaseClient):
    def __init__(self, **kwargs):
        self._client = Anthropic(**create_parameters(Anthropic, **kwargs))
        self._async_client = AsyncAnthropic(**create_parameters(AsyncAnthropic, **kwargs))

    @override
    def generate_content(self, **kwargs) -> Message | Stream[RawMessageStreamEvent]:
        completion_params = create_parameters(self._client.messages.create, **kwargs)
        return self._client.messages.create(**completion_params)

    @override
    async def generate_content_async(
        self, **kwargs
    ) -> Message | AsyncStream[RawMessageStreamEvent]:
        completion_params = create_parameters(self._async_client.messages.create, **kwargs)
        return await self._async_client.messages.create(**completion_params)
