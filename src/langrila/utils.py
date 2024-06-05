import base64
import io
from typing import Any, Generator, Iterable

import numpy as np
from PIL import Image


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


def encode_image(image: Image.Image | np.ndarray | bytes) -> str:
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


def decode_image(image_encoded: str) -> Image.Image:
    image_encoded_utf = image_encoded.encode("utf-8")
    image_bytes = base64.b64decode(image_encoded_utf)
    byteio = io.BytesIO(image_bytes)
    return Image.open(byteio)
