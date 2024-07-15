import json
from typing import Any, Optional

import numpy as np
from google.generativeai import protos
from google.generativeai.types import content_types
from google.generativeai.types.content_types import BlobDict, ContentDict, PartDict
from PIL import Image

from ...base import BaseMessage
from ...utils import pil2bytes


def encode_image(image: Image.Image | np.ndarray | bytes) -> str:
    if isinstance(image, Image.Image):
        return pil2bytes(image)
    elif isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
        return pil2bytes(image_pil)
    elif isinstance(image, bytes):
        return image
    else:
        raise ValueError(f"Type of {type(image)} is not supported for image.")


class GeminiMessage(BaseMessage):
    def __init__(
        self,
        content: str,
        name: Optional[str] = None,
        images: Any | list[Any] | None = None,
    ):
        super().__init__(content=content, images=images, name=name)

    @property
    def as_system(self):
        raise NotImplementedError

    @property
    def as_user(self) -> protos.Content:
        parts = [PartDict(text=self.content)]
        if self.images:
            if not isinstance(self.images, list):
                images = [self.images]
            else:
                images = self.images

            for image in images:
                image_bytes = BlobDict(mime_type="image/png", data=encode_image(image))
                parts.append(PartDict(inline_data=image_bytes))

        content_dict = ContentDict(role="user", parts=parts)
        return content_types.to_content(content_dict)

    @property
    def as_assistant(self) -> protos.Content:
        content_dict = ContentDict(role="model", parts=[PartDict(text=self.content)])
        return content_types.to_content(content_dict)

    @property
    def as_function(self) -> protos.Content:
        content_dict = protos.FunctionResponse(name=self.name, response={"content": self.content})
        return content_types.to_content(content_dict)

    @staticmethod
    def to_dict(content: protos.Content) -> dict[str, Any]:
        return json.loads(
            type(content).to_json(
                content,
                including_default_value_fields=False,
                use_integers_for_enums=False,
            )
        )

    @staticmethod
    def from_dict(content_dict: dict[str, Any]) -> protos.Content:
        return protos.Content.from_json(json.dumps(content_dict))
