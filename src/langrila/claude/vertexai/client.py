import os
from typing import Mapping

import httpx
from anthropic import AnthropicVertex, AsyncAnthropicVertex
from anthropic._base_client import DEFAULT_MAX_RETRIES
from anthropic._types import NOT_GIVEN, NotGiven, ProxiesTypes, Transport
from google.auth.credentials import Credentials as GoogleCredentials


def get_vertexai_client(
    gc_region_env_name: str | NotGiven = NOT_GIVEN,
    gc_project_id_env_name: str | NotGiven = NOT_GIVEN,
    gc_access_token_env_name: str | None = None,
    credentials: GoogleCredentials | None = None,
    endpoint_env_name: str | httpx.URL | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    max_retries: int = DEFAULT_MAX_RETRIES,
    default_headers: Mapping[str, str] | None = None,
    default_query: Mapping[str, object] | None = None,
    http_client: httpx.Client | None = None,
    transport: Transport | None = None,
    proxies: ProxiesTypes | None = None,
    connection_pool_limits: httpx.Limits | None = None,
    _strict_response_validation: bool = False,
) -> AnthropicVertex:
    return AnthropicVertex(
        region=os.getenv(gc_region_env_name) if gc_region_env_name else None,
        project_id=os.getenv(gc_project_id_env_name) if gc_project_id_env_name else None,
        access_token=os.getenv(gc_access_token_env_name) if gc_access_token_env_name else None,
        credentials=credentials,
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


def get_async_vertexai_client(
    gc_region_env_name: str | NotGiven = NOT_GIVEN,
    gc_project_id_env_name: str | NotGiven = NOT_GIVEN,
    gc_access_token_env_name: str | None = None,
    credentials: GoogleCredentials | None = None,
    endpoint_env_name: str | httpx.URL | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    max_retries: int = DEFAULT_MAX_RETRIES,
    default_headers: Mapping[str, str] | None = None,
    default_query: Mapping[str, object] | None = None,
    http_client: httpx.Client | None = None,
    transport: Transport | None = None,
    proxies: ProxiesTypes | None = None,
    connection_pool_limits: httpx.Limits | None = None,
    _strict_response_validation: bool = False,
) -> AsyncAnthropicVertex:
    return AsyncAnthropicVertex(
        region=os.getenv(gc_region_env_name) if gc_region_env_name else None,
        project_id=os.getenv(gc_project_id_env_name) if gc_project_id_env_name else None,
        access_token=os.getenv(gc_access_token_env_name) if gc_access_token_env_name else None,
        credentials=credentials,
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
