import math
import os
import re
from typing import Literal, Mapping, overload

import httpx
import tiktoken
from openai.lib.azure import AzureADTokenProvider

from ..utils import decode_image
from .azure.client import AzureOpenAIChat
from .model_config import (
    _TILE_SIZE,
    _TOKENS_PER_TILE,
    EMBEDDING_CONFIG,
    MODEL_CONFIG,
)
from .origin.client import OpenAIChat

MODEL_ZOO = set(MODEL_CONFIG.keys()) | set(EMBEDDING_CONFIG.keys())


def get_encoding(model_name: str) -> tiktoken.Encoding:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")

    return encoding


@overload
def get_client(
    api_type: Literal["openai"],
    api_key_env_name: str,
    organization_id_env_name: str | None = None,
    timeout: int = 60,
    max_retries: int = 5,
    project: str | None = None,
    base_url: str | httpx.URL | None = None,
    default_headers: Mapping[str, str] | None = None,
    default_query: Mapping[str, object] | None = None,
    http_client: httpx.Client | None = None,
    _strict_response_validation: bool = False,
) -> OpenAIChat: ...


@overload
def get_client(
    api_type: Literal["azure"],
    api_key_env_name: str,
    api_version: str | None = None,
    endpoint_env_name: str | None = None,
    organization_id_env_name: str | None = None,
    deployment_id_env_name: str | None = None,
    timeout: int = 60,
    max_retries: int = 5,
    project: str | None = None,
    base_url: str | httpx.URL | None = None,
    azure_ad_token: str | None = None,
    azure_ad_token_provider: AzureADTokenProvider | None = None,
    default_headers: Mapping[str, str] | None = None,
    default_query: Mapping[str, object] | None = None,
    http_client: httpx.Client | None = None,
    _strict_response_validation: bool = False,
) -> AzureOpenAIChat: ...


def get_client(
    api_key_env_name: str,
    api_version: str | None = None,
    endpoint_env_name: str | None = None,
    organization_id_env_name: str | None = None,
    deployment_id_env_name: str | None = None,
    api_type: Literal["openai", "azure"] | None = "openai",
    timeout: int = 60,
    max_retries: int = 5,
    project: str | None = None,
    base_url: str | httpx.URL | None = None,
    azure_ad_token: str | None = None,
    azure_ad_token_provider: AzureADTokenProvider | None = None,
    default_headers: Mapping[str, str] | None = None,
    default_query: Mapping[str, object] | None = None,
    http_client: httpx.Client | None = None,
    _strict_response_validation: bool = False,
) -> OpenAIChat | AzureOpenAIChat:
    if api_type == "azure":
        assert (
            api_version and endpoint_env_name and deployment_id_env_name
        ), "api_version, endpoint_env_name, and deployment_id_env_name must be specified when api_type is 'azure'."
        return AzureOpenAIChat(
            api_key=os.getenv(api_key_env_name),
            organization=os.getenv(organization_id_env_name),
            api_version=api_version,
            azure_endpoint=os.getenv(endpoint_env_name),
            azure_deployment=os.getenv(deployment_id_env_name),
            max_retries=max_retries,
            timeout=timeout,
            project=project,
            base_url=base_url,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
        )
    elif api_type == "openai":
        return OpenAIChat(
            api_key=os.getenv(api_key_env_name),
            organization=os.getenv(organization_id_env_name),
            max_retries=max_retries,
            timeout=timeout,
            project=project,
            base_url=base_url,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
        )
    else:
        raise ValueError(f"api_type must be 'azure' or 'openai'. Got {api_type}.")


# def get_async_client(
#     api_key_env_name: str,
#     api_version: str | None = None,
#     endpoint_env_name: str | None = None,
#     organization_id_env_name: str | None = None,
#     deployment_id_env_name: str | None = None,
#     api_type: str | None = "openai",
#     timeout: int = 60,
#     max_retries: int = 5,
# ) -> AsyncOpenAI | AsyncAzureOpenAI:
#     if api_type == "azure":
#         return AsyncAzureOpenAI(
#             **get_openai_client_settings(
#                 api_key_env_name=api_key_env_name,
#                 organization_id_env_name=organization_id_env_name,
#                 api_version=api_version,
#                 endpoint_env_name=endpoint_env_name,
#                 deployment_id_env_name=deployment_id_env_name,
#                 max_retries=max_retries,
#                 timeout=timeout,
#             )
#         )
#     elif api_type == "openai":
#         return AsyncOpenAI(
#             **get_openai_client_settings(
#                 api_key_env_name=api_key_env_name,
#                 organization_id_env_name=organization_id_env_name,
#                 max_retries=max_retries,
#                 timeout=timeout,
#             )
#         )
#     else:
#         raise ValueError(f"api_type must be 'azure' or 'openai'. Got {api_type}.")


# def get_openai_client_settings(
#     api_key_env_name: str,
#     api_version: str | None = None,
#     endpoint_env_name: str | None = None,
#     organization_id_env_name: str | None = None,
#     deployment_id_env_name: str | None = None,
#     timeout: int = 60,
#     max_retries: int = 5,
# ) -> dict[str, Any]:
#     outputs = {}
#     outputs["api_key"] = os.getenv(api_key_env_name)

#     if isinstance(api_version, str):
#         outputs["api_version"] = api_version

#     if isinstance(endpoint_env_name, str):
#         outputs["azure_endpoint"] = os.getenv(endpoint_env_name)

#     if isinstance(organization_id_env_name, str):
#         outputs["organization"] = os.getenv(organization_id_env_name)

#     if isinstance(deployment_id_env_name, str):
#         outputs["azure_deployment"] = os.getenv(deployment_id_env_name)

#     outputs["timeout"] = timeout
#     outputs["max_retries"] = max_retries

#     return outputs


# def set_openai_envs(
#     api_key_env_name: str,
#     api_version: str | None = None,
#     api_type: str | None = None,
#     endpoint_env_name: str | None = None,
#     organization_id_env_name: str | None = None,
# ) -> None:
#     openai.api_key = os.getenv(api_key_env_name)

#     if isinstance(api_version, str):
#         openai.api_version = api_version

#     if isinstance(api_type, str):
#         openai.api_type = api_type

#     if isinstance(endpoint_env_name, str):
#         openai.api_base = os.getenv(endpoint_env_name)

#     if isinstance(organization_id_env_name, str):
#         openai.organization = os.getenv(organization_id_env_name)


def get_n_tokens(
    message: dict[str, dict[str, str]], model_name: str, depth: int = 0
) -> dict[str, int]:
    """
    Return the number of tokens used by a list of messages.
    Forked and edited from : https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    FIXME: This function is not perfect. It might not be accurate while it works enough in many cases.
    """
    encoding = get_encoding(model_name)

    tokens_per_message = 3
    tokens_per_name = 1

    n_content_tokens = 0
    n_other_tokens = 0
    # num_tokens += tokens_per_message
    for key, value in message.items():
        if key == "tool_calls":
            continue
        if value is not None:
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        tokens = get_n_tokens(item, model_name, depth + 1)
                        n_content_tokens += tokens["content"]
                        n_other_tokens += tokens["other"]
                        # n_other_tokens += len(encoding.encode(key))
            elif key == "name":
                n_other_tokens += tokens_per_name
            elif isinstance(value, str):
                n_content_tokens += len(encoding.encode(value))
            elif key == "text":
                n_content_tokens += len(encoding.encode(item["text"]))
            elif key == "image_url":
                n_content_tokens += 85  # Base tokens
                if value["detail"] in ["high", "auto"]:
                    if value["url"].startswith("data:image/"):
                        img_encoded = re.sub("^(data:image/.+;base64,)", "", value["url"])
                        n_content_tokens += calculate_high_resolution_image_tokens(
                            decode_image(img_encoded).size
                        )
                    elif value["url"].startswith("https://"):
                        raise NotImplementedError(
                            "Image URL is not acceptable. Please use base64 encoded image."
                        )
            elif isinstance(value, dict):
                tokens = get_n_tokens(value, model_name, depth + 1)
                n_content_tokens += tokens["content"]
                n_other_tokens += tokens["other"]
                # n_other_tokens += len(encoding.encode(key))
            else:
                n_content_tokens += len(encoding.encode(value))
        else:
            n_other_tokens += len(encoding.encode(key))

    if depth == 0:
        n_other_tokens += tokens_per_message
        n_other_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

    total_tokens = n_content_tokens + n_other_tokens
    return {"total": total_tokens, "content": n_content_tokens, "other": n_other_tokens}


def get_token_limit(model_name: str) -> int:
    if model_name in MODEL_ZOO:
        return MODEL_CONFIG[model_name]["max_tokens"]
    else:
        raise NotImplementedError(
            f"get_token_limit() is not implemented for model {model_name}. Please choose from following model : {', '.join(sorted(list(MODEL_ZOO)))}."
        )


def calculate_high_resolution_image_tokens(image_size: tuple[int, int] | list[int, int]) -> int:
    h, w = image_size
    short = min(h, w)
    long = max(h, w)

    if long > 2048:
        short = int(short * 2048 / long)
        long = 2048

    if short > 768:
        long = int(long * 768 / short)
        short = 768

    n_bins_long = math.ceil(long / _TILE_SIZE)
    n_bins_short = math.ceil(short / _TILE_SIZE)
    n_tiles = n_bins_long * n_bins_short
    return _TOKENS_PER_TILE * n_tiles
