import math
import os
import re
from typing import Any

import openai
import tiktoken
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from ..utils import decode_image
from .model_config import (
    _TILE_SIZE,
    _TOKENS_PER_TILE,
    EMBEDDING_CONFIG,
    MODEL_CONFIG,
)

MODEL_ZOO = set(MODEL_CONFIG.keys()) | set(EMBEDDING_CONFIG.keys())


def get_encoding(model_name: str) -> tiktoken.Encoding:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")

    return encoding


def get_client(
    api_key_env_name: str,
    api_version: str | None = None,
    endpoint_env_name: str | None = None,
    organization_id_env_name: str | None = None,
    deployment_id_env_name: str | None = None,
    api_type: str | None = "openai",
    timeout: int = 60,
    max_retries: int = 5,
) -> OpenAI | AzureOpenAI:
    if api_type == "azure":
        assert (
            api_version and endpoint_env_name and deployment_id_env_name
        ), "api_version, endpoint_env_name, and deployment_id_env_name must be specified when api_type is 'azure'."
        return AzureOpenAI(
            **get_openai_client_settings(
                api_key_env_name=api_key_env_name,
                organization_id_env_name=organization_id_env_name,
                api_version=api_version,
                endpoint_env_name=endpoint_env_name,
                deployment_id_env_name=deployment_id_env_name,
                max_retries=max_retries,
                timeout=timeout,
            )
        )
    elif api_type == "openai":
        return OpenAI(
            **get_openai_client_settings(
                api_key_env_name=api_key_env_name,
                organization_id_env_name=organization_id_env_name,
                max_retries=max_retries,
                timeout=timeout,
            )
        )
    else:
        raise ValueError(f"api_type must be 'azure' or 'openai'. Got {api_type}.")


def get_async_client(
    api_key_env_name: str,
    api_version: str | None = None,
    endpoint_env_name: str | None = None,
    organization_id_env_name: str | None = None,
    deployment_id_env_name: str | None = None,
    api_type: str | None = "openai",
    timeout: int = 60,
    max_retries: int = 5,
) -> AsyncOpenAI | AsyncAzureOpenAI:
    if api_type == "azure":
        return AsyncAzureOpenAI(
            **get_openai_client_settings(
                api_key_env_name=api_key_env_name,
                organization_id_env_name=organization_id_env_name,
                api_version=api_version,
                endpoint_env_name=endpoint_env_name,
                deployment_id_env_name=deployment_id_env_name,
                max_retries=max_retries,
                timeout=timeout,
            )
        )
    elif api_type == "openai":
        return AsyncOpenAI(
            **get_openai_client_settings(
                api_key_env_name=api_key_env_name,
                organization_id_env_name=organization_id_env_name,
                max_retries=max_retries,
                timeout=timeout,
            )
        )
    else:
        raise ValueError(f"api_type must be 'azure' or 'openai'. Got {api_type}.")


def get_openai_client_settings(
    api_key_env_name: str,
    api_version: str | None = None,
    endpoint_env_name: str | None = None,
    organization_id_env_name: str | None = None,
    deployment_id_env_name: str | None = None,
    timeout: int = 60,
    max_retries: int = 5,
) -> dict[str, Any]:
    outputs = {}
    outputs["api_key"] = os.getenv(api_key_env_name)

    if isinstance(api_version, str):
        outputs["api_version"] = api_version

    if isinstance(endpoint_env_name, str):
        outputs["azure_endpoint"] = os.getenv(endpoint_env_name)

    if isinstance(organization_id_env_name, str):
        outputs["organization"] = os.getenv(organization_id_env_name)

    if isinstance(deployment_id_env_name, str):
        outputs["azure_deployment"] = os.getenv(deployment_id_env_name)

    outputs["timeout"] = timeout
    outputs["max_retries"] = max_retries

    return outputs


def set_openai_envs(
    api_key_env_name: str,
    api_version: str | None = None,
    api_type: str | None = None,
    endpoint_env_name: str | None = None,
    organization_id_env_name: str | None = None,
) -> None:
    openai.api_key = os.getenv(api_key_env_name)

    if isinstance(api_version, str):
        openai.api_version = api_version

    if isinstance(api_type, str):
        openai.api_type = api_type

    if isinstance(endpoint_env_name, str):
        openai.api_base = os.getenv(endpoint_env_name)

    if isinstance(organization_id_env_name, str):
        openai.organization = os.getenv(organization_id_env_name)


def get_n_tokens(message: dict[str, dict[str, str]], model_name: str) -> dict[str, int]:
    """
    Return the number of tokens used by a list of messages.
    Forked and edited from : https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    encoding = get_encoding(model_name)

    if model_name in MODEL_ZOO:
        if model_name == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        else:
            tokens_per_message = 3
            tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"get_n_tokens() is not implemented for model {model_name}. Please choose from following model : {', '.join(sorted(list(MODEL_ZOO)))}."
        )

    n_content_tokens = 0
    n_other_tokens = tokens_per_message
    # num_tokens += tokens_per_message
    for key, value in message.items():
        if value is not None:
            if key == "content" and not isinstance(value, str):
                if isinstance(value, list):
                    for item in value:
                        if item["type"] == "text":
                            n_content_tokens += len(encoding.encode(item["text"]))
                        elif item["type"] == "image_url":
                            n_content_tokens += 85  # Base tokens
                            if item["image_url"]["detail"] in ["high", "auto"]:
                                if item["image_url"]["url"].startswith("data:image/"):
                                    img_encoded = re.sub(
                                        "^(data:image/.+;base64,)", "", item["image_url"]["url"]
                                    )
                                    n_content_tokens += calculate_high_resolution_image_tokens(
                                        decode_image(img_encoded).size
                                    )
                                elif item["image_url"]["url"].startswith("https://"):
                                    raise NotImplementedError(
                                        "Image URL is not acceptable. Please use base64 encoded image."
                                    )
                        else:
                            raise ValueError(f"Unknown type {item['type']} in message['content'].")
                else:
                    n_content_tokens += len(encoding.encode(value))
            elif key == "name":
                n_other_tokens += tokens_per_name
            else:
                n_other_tokens += len(encoding.encode(value))

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
