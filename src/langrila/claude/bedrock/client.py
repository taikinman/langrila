import os
from typing import Mapping

import httpx
from anthropic import AnthropicBedrock, AsyncAnthropicBedrock
from anthropic._base_client import DEFAULT_MAX_RETRIES
from anthropic._types import NOT_GIVEN, NotGiven


def get_bedrock_client(
    aws_secret_key_env_name: str | None = None,
    aws_access_key_env_name: str | None = None,
    aws_region_env_name: str | None = None,
    aws_session_token_env_name: str | None = None,
    endpoint_env_name: str | httpx.URL | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    max_retries: int = DEFAULT_MAX_RETRIES,
    default_headers: Mapping[str, str] | None = None,
    default_query: Mapping[str, object] | None = None,
    http_client: httpx.Client | None = None,
    _strict_response_validation: bool = False,
) -> AnthropicBedrock:
    return AnthropicBedrock(
        aws_secret_key=os.getenv(aws_secret_key_env_name) if aws_secret_key_env_name else None,
        aws_access_key=os.getenv(aws_access_key_env_name) if aws_access_key_env_name else None,
        aws_region=os.getenv(aws_region_env_name) if aws_region_env_name else None,
        aws_session_token=os.getenv(aws_session_token_env_name)
        if aws_session_token_env_name
        else None,
        base_url=os.getenv(endpoint_env_name) if endpoint_env_name else None,
        timeout=timeout,
        max_retries=max_retries,
        default_headers=default_headers,
        default_query=default_query,
        http_client=http_client,
        _strict_response_validation=_strict_response_validation,
    )


def get_async_bedrock_client(
    aws_secret_key_env_name: str | None = None,
    aws_access_key_env_name: str | None = None,
    aws_region_env_name: str | None = None,
    aws_session_token_env_name: str | None = None,
    endpoint_env_name: str | httpx.URL | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    max_retries: int = DEFAULT_MAX_RETRIES,
    default_headers: Mapping[str, str] | None = None,
    default_query: Mapping[str, object] | None = None,
    http_client: httpx.Client | None = None,
    _strict_response_validation: bool = False,
) -> AsyncAnthropicBedrock:
    return AsyncAnthropicBedrock(
        aws_secret_key=os.getenv(aws_secret_key_env_name) if aws_secret_key_env_name else None,
        aws_access_key=os.getenv(aws_access_key_env_name) if aws_access_key_env_name else None,
        aws_region=os.getenv(aws_region_env_name) if aws_region_env_name else None,
        aws_session_token=os.getenv(aws_session_token_env_name)
        if aws_session_token_env_name
        else None,
        base_url=os.getenv(endpoint_env_name) if endpoint_env_name else None,
        timeout=timeout,
        max_retries=max_retries,
        default_headers=default_headers,
        default_query=default_query,
        http_client=http_client,
        _strict_response_validation=_strict_response_validation,
    )
