import os
from typing import Optional, Union

import openai
import tiktoken
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from .model_config import MODEL_CONFIG

MODEL_ZOO = set(MODEL_CONFIG.keys())


def get_client(
    api_key_name: str,
    api_version: Optional[str] = None,
    endpoint_name: Optional[str] = None,
    organization_id_name: Optional[str] = None,
    deployment_id_name: Optional[str] = None,
    api_type: Optional[str] = "openai",
    timeout: int = 60,
    max_retries: int = 5,
):
    if api_type == "azure":
        assert (
            api_version and endpoint_name and deployment_id_name
        ), "api_version, endpoint_name, and deployment_id_name must be specified when api_type is 'azure'."
        return AzureOpenAI(
            **get_openai_client_settings(
                api_key_name=api_key_name,
                organization_id_name=organization_id_name,
                api_version=api_version,
                endpoint_name=endpoint_name,
                deployment_id_name=deployment_id_name,
                max_retries=max_retries,
                timeout=timeout,
            )
        )
    elif api_type == "openai":
        return OpenAI(
            **get_openai_client_settings(
                api_key_name=api_key_name,
                organization_id_name=organization_id_name,
                max_retries=max_retries,
                timeout=timeout,
            )
        )
    else:
        raise ValueError(f"api_type must be 'azure' or 'openai'. Got {api_type}.")


def get_async_client(
    api_key_name: str,
    api_version: Optional[str] = None,
    endpoint_name: Optional[str] = None,
    organization_id_name: Optional[str] = None,
    deployment_id_name: Optional[str] = None,
    api_type: Optional[str] = "openai",
    timeout: int = 60,
    max_retries: int = 5,
):
    if api_type == "azure":
        return AsyncAzureOpenAI(
            **get_openai_client_settings(
                api_key_name=api_key_name,
                organization_id_name=organization_id_name,
                api_version=api_version,
                endpoint_name=endpoint_name,
                deployment_id_name=deployment_id_name,
                max_retries=max_retries,
                timeout=timeout,
            )
        )
    elif api_type == "openai":
        return AsyncOpenAI(
            **get_openai_client_settings(
                api_key_name=api_key_name,
                organization_id_name=organization_id_name,
                max_retries=max_retries,
                timeout=timeout,
            )
        )
    else:
        raise ValueError(f"api_type must be 'azure' or 'openai'. Got {api_type}.")


def get_openai_client_settings(
    api_key_name: str,
    api_version: Optional[str] = None,
    endpoint_name: Optional[str] = None,
    organization_id_name: Optional[str] = None,
    deployment_id_name: Optional[str] = None,
    timeout: int = 60,
    max_retries: int = 5,
) -> None:
    outputs = {}
    outputs["api_key"] = os.getenv(api_key_name)

    if isinstance(api_version, str):
        outputs["api_version"] = api_version

    if isinstance(endpoint_name, str):
        outputs["azure_endpoint"] = os.getenv(endpoint_name)

    if isinstance(organization_id_name, str):
        outputs["organization"] = os.getenv(organization_id_name)

    if isinstance(deployment_id_name, str):
        outputs["azure_deployment"] = os.getenv(deployment_id_name)

    outputs["timeout"] = timeout
    outputs["max_retries"] = max_retries

    return outputs


def set_openai_envs(
    api_key_name: str,
    api_version: Optional[str] = None,
    api_type: Optional[str] = None,
    endpoint_name: Optional[str] = None,
    organization_id_name: Optional[str] = None,
) -> None:
    openai.api_key = os.getenv(api_key_name)

    if isinstance(api_version, str):
        openai.api_version = api_version

    if isinstance(api_type, str):
        openai.api_type = api_type

    if isinstance(endpoint_name, str):
        openai.api_base = os.getenv(endpoint_name)

    if isinstance(organization_id_name, str):
        openai.organization = os.getenv(organization_id_name)


def get_n_tokens(message: dict[str, str], model_name: str) -> int:
    """
    Return the number of tokens used by a list of messages.
    Folked and edited from : https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
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
        if key == "content":
            n_content_tokens += len(encoding.encode(value))
        elif key == "name":
            n_other_tokens += tokens_per_name
        else:
            n_other_tokens += len(encoding.encode(value))
    n_other_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    total_tokens = n_content_tokens + n_other_tokens
    return {"total": total_tokens, "content": n_content_tokens, "other": n_other_tokens}


def get_token_limit(model_name: str):
    if model_name in MODEL_ZOO:
        return MODEL_CONFIG[model_name]["max_tokens"]
    else:
        raise NotImplementedError(
            f"get_token_limit() is not implemented for model {model_name}. Please choose from following model : {', '.join(sorted(list(MODEL_ZOO)))}."
        )


def make_batch(iterable, batch_size=1):
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx : min(ndx + batch_size, length)]
