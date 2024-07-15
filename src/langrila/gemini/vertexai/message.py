import json
from typing import Any, Optional

from vertexai.generative_models import Content, Part

from ...base import BaseMessage
from ...utils import pil2bytes


class VertexAIMessage(BaseMessage):
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
    def as_user(self) -> Content:
        parts = [Part.from_text(self.content)]
        if self.images:
            if not isinstance(self.images, list):
                images = [self.images]
            else:
                images = self.images

            for image in images:
                image = Part.from_data(data=pil2bytes(image), mime_type="image/png")
                parts.append(image)

        return Content(role="user", parts=parts)

    @property
    def as_assistant(self) -> Content:
        return Content(role="model", parts=[Part.from_text(self.content)])

    @property
    def as_function(self) -> Content:
        return Content(
            parts=[Part.from_function_response(name=self.name, response={"content": self.content})],
        )

    @staticmethod
    def to_dict(content: Content) -> dict[str, Any]:
        return content.to_dict()

    @staticmethod
    def from_dict(content_dict: dict[str, Any]) -> Content:
        return Content.from_dict(content_dict)
