import base64
import inspect
import io
import random
import string
from functools import partial
from inspect import signature
from typing import Any, Callable, Generator, Iterable
from urllib.parse import urlparse

import numpy as np
from PIL import Image
from pydantic import BaseModel


def get_variable_name_inspect(obj: object) -> str:
    if current_frame := inspect.currentframe():
        frame = current_frame.f_back
        while frame:
            for name, _obj in frame.f_globals.items():
                if _obj is obj:
                    return name
            frame = frame.f_back
    return ""


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
    image.save(num_byteio, format=image.format.lower() if image.format else "jpeg")
    num_byteio.seek(0)
    image_bytes = num_byteio.getvalue()
    return image_bytes


def image2bytes(image: Image.Image | np.ndarray | bytes) -> bytes:
    if isinstance(image, Image.Image):
        return pil2bytes(image)
    elif isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
        return pil2bytes(image_pil)
    elif isinstance(image, bytes):
        return image
    else:
        raise ValueError(f"Type of {type(image)} is not supported for image.")


def base64_encode(data: Any):
    return base64.b64encode(data)


def base64_decode(data: Any):
    return base64.b64decode(data)


def utf8_to_bytes(text: str) -> bytes:
    return base64_decode(text.encode("utf-8"))


def encode_image(image: Image.Image | np.ndarray | bytes) -> str:
    image_bytes = image2bytes(image)
    encoded = base64_encode(image_bytes)
    return encoded.decode("utf-8")


def decode_image(image_encoded: str) -> Image.Image:
    image_encoded_utf = image_encoded.encode("utf-8")
    image_bytes = base64.b64decode(image_encoded_utf)
    byteio = io.BytesIO(image_bytes)
    return Image.open(byteio)


def generate_dummy_call_id(n: int) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def model2func(model: BaseModel):
    def run_model(model: BaseModel, **kwargs):
        return model(**kwargs).run()

    return partial(run_model, model=model)


def is_valid_uri(text: str) -> bool:
    """Function to determine if a given string is a URI

    Args:
        text (str): The string to check

    Returns:
        bool: True if the string is a URI, False otherwise
    """
    try:
        result = urlparse(text)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def parse_arguments(
    obj: Callable[..., Any],
    include: set[str] | list[str] | None = None,
    exclude: set[str] | list[str] | None = None,
) -> set[str]:
    params = dict(signature(obj).parameters)

    if "self" in params:
        params.pop("self")

    if include is None:
        include = set()

    if exclude is None:
        exclude = set()

    if not isinstance(include, (set, list)):
        raise ValueError("include must be a set or list.")

    if not isinstance(exclude, (set, list)):
        raise ValueError("exclude must be a set or list.")

    if include:
        params = {k: v for k, v in params.items() if k in include}

    if exclude:
        params = {k: v for k, v in params.items() if k not in exclude}

    return set(params.keys())


def create_parameters(
    obj: Callable[..., Any],
    include: set[str] | list[str] | None = None,
    exclude: set[str] | list[str] | None = None,
    **kwargs: dict[str, Any],
) -> dict[str, Any]:
    params = parse_arguments(obj, include=include, exclude=exclude)
    # ignored = set(kwargs.keys()) - params
    return {k: v for k, v in kwargs.items() if k in params}


def snake_to_camel(snake_str: str) -> str:
    if not snake_str:
        return ""

    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])
