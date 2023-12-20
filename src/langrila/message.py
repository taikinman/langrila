from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, field_validator

from .utils import encode_image


class Message(BaseModel):
    content: str
    images: Any | list[Any] | None = None
    image_resolution: str | None = None

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
            return {"role": "user", "content": content}
        else:
            return {"role": "user", "content": self.content}

    @property
    def as_assistant(self):
        return {"role": "assistant", "content": self.content}

    @property
    def as_tool(self):
        return {"role": "tool", "content": self.content}

    @property
    def as_function(self):
        return {"role": "function", "content": self.content}

    @field_validator("image_resolution")
    def check_image_resolution_value(cls, val):
        if val not in ["low", "high"]:
            raise ValueError(
                "image_resolution must be either 'low' or 'high' due to token management."
            )
        return val
