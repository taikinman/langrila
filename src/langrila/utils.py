import base64
import io
import math
import os
from typing import Any, Generator, Iterable, Optional

import numpy as np
import openai
import tiktoken
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from PIL import Image

from .model_config import _TILE_SIZE, _TOKENS_PER_TILE, _VISION_MODEL, MODEL_CONFIG

MODEL_ZOO = set(MODEL_CONFIG.keys())


def get_encoding(model_name: str):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")

    return encoding


def get_client(
    api_key_env_name: str,
    api_version: Optional[str] = None,
    endpoint_env_name: Optional[str] = None,
    organization_id_env_name: Optional[str] = None,
    deployment_id_env_name: Optional[str] = None,
    api_type: Optional[str] = "openai",
    timeout: int = 60,
    max_retries: int = 5,
):
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
    api_version: Optional[str] = None,
    endpoint_env_name: Optional[str] = None,
    organization_id_env_name: Optional[str] = None,
    deployment_id_env_name: Optional[str] = None,
    api_type: Optional[str] = "openai",
    timeout: int = 60,
    max_retries: int = 5,
):
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
    api_version: Optional[str] = None,
    endpoint_env_name: Optional[str] = None,
    organization_id_env_name: Optional[str] = None,
    deployment_id_env_name: Optional[str] = None,
    timeout: int = 60,
    max_retries: int = 5,
) -> None:
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
    api_version: Optional[str] = None,
    api_type: Optional[str] = None,
    endpoint_env_name: Optional[str] = None,
    organization_id_env_name: Optional[str] = None,
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


def get_n_tokens(
    message: dict[str, str | list[dict[str, str | dict[str, str]]]], model_name: str
) -> int:
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
        if key == "content":
            if model_name in _VISION_MODEL and isinstance(value, list):
                for item in value:  # value type is list[dict[str, str|dict[str, str]]
                    if item["type"] == "text":
                        n_content_tokens += len(encoding.encode(item["text"]))
                    elif item["type"] == "image_url":
                        n_content_tokens += 85  # Base tokens
                        if item["image_url"]["detail"] == "high":
                            if item["image_url"]["url"].startswith("data:image/jpeg;base64,"):
                                img_encoded = item["image_url"]["url"].replace(
                                    "data:image/jpeg;base64,", ""
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


def get_token_limit(model_name: str):
    if model_name in MODEL_ZOO:
        return MODEL_CONFIG[model_name]["max_tokens"]
    else:
        raise NotImplementedError(
            f"get_token_limit() is not implemented for model {model_name}. Please choose from following model : {', '.join(sorted(list(MODEL_ZOO)))}."
        )


def make_batch(
    iterable: Iterable[Any], batch_size: int = 1, overlap: int = 0
) -> Generator[Iterable[Any], None, None]:
    if overlap >= batch_size:
        raise ValueError("overlap must be less than batch_size")

    length = len(iterable)

    st: int = 0
    while st < length:
        en = min(st + batch_size, length)
        batch = iterable[st:en]
        yield batch

        if en == length:
            break

        st += batch_size - overlap


def pil2bytes(image: Image.Image) -> bytes:
    num_byteio = io.BytesIO()
    image.save(num_byteio, format="jpeg")
    image_bytes = num_byteio.getvalue()
    return image_bytes


def encode_image(image):
    if isinstance(image, Image.Image):
        image_bytes = pil2bytes(image)
        return base64.b64encode(image_bytes).decode("utf-8")
    elif isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
        image_bytes = pil2bytes(image_pil)
        return base64.b64encode(image_bytes).decode("utf-8")
    elif isinstance(image, bytes):
        return base64.b64encode(image).decode("utf-8")
    else:
        raise ValueError(f"Type of {type(image)} is not supported for image.")


def decode_image(image_encoded):
    image_encoded_utf = image_encoded.encode("utf-8")
    image_bytes = base64.b64decode(image_encoded_utf)
    byteio = io.BytesIO(image_bytes)
    return Image.open(byteio)


def calculate_high_resolution_image_tokens(image_size: tuple[int, int] | list[int, int]):
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
