from typing import Any, Optional

from ..base import BaseMessage
from ..utils import encode_image


class OpenAIMessage(BaseMessage):
    def __init__(
        self,
        content: str,
        name: Optional[str] = None,
        images: Any | list[Any] | None = None,
        image_resolution: str | None = None,
    ):
        super().__init__(content=content, images=images, name=name)
        self._valid_image_resolution_value(image_resolution)

        self.image_resolution = image_resolution

    @property
    def as_system(self):
        return {"role": "system", "content": self.content}

    @property
    def as_user(self):
        if self.images:
            content = [{"type": "text", "text": self.content}]
            if not isinstance(self.images, list):
                images = [self.images]
            else:
                images = self.images

            for image in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image)}",
                            "detail": self.image_resolution if self.image_resolution else "low",
                        },
                    }
                )
            return {"role": "user", "content": content} | ({"name": self.name} if self.name else {})
        else:
            return {"role": "user", "content": self.content} | (
                {"name": self.name} if self.name else {}
            )

    @property
    def as_assistant(self):
        return {"role": "assistant", "content": self.content}

    # @property
    # def as_tool(self):
    #     return {"role": "tool", "content": self.content}

    @property
    def as_function(self):
        return {
            "role": "function",
            "name": self.name,
            "content": self.content,
        }

    def _valid_image_resolution_value(self, image_resolution: str) -> None:
        if image_resolution:
            if image_resolution not in ["low", "high"]:
                raise ValueError(
                    "image_resolution must be either 'low' or 'high' due to token management."
                )
