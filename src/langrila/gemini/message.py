from typing import Any, Optional

from ..base import BaseMessage
from ..utils import pil2bytes


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
        return {"role": "user", "parts": [f"*{self.content}*"]}

    @property
    def as_user(self):
        content = {"role": "user", "parts": [self.content]}
        if self.images:
            if not isinstance(self.images, list):
                images = [self.images]
            else:
                images = self.images

            for image in images:
                image_bytes = {
                    "mime_type": "image/png",
                    "data": pil2bytes(image),
                }
                content["parts"].append(image_bytes)

            return content
        else:
            return content

    @property
    def as_assistant(self):
        return {"role": "model", "parts": [self.content]}

    @property
    def as_function(self):
        return {
            "role": "model",
            "parts": [f"function_response: {{'name': {self.name}, 'response': {self.content}}}"],
        }
