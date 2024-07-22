from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, model_validator

from .types import FileType, PathType
from .utils import encode_image


class TextContent(BaseModel):
    text: str


class ImageContent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: Image.Image | PathType
    resolution: Literal["auto", "low", "high"] | None = None

    @property
    def format(self):
        if isinstance(self.image, Image.Image):
            return self.image.format.lower()
        elif isinstance(self.image, PathType):
            return Path(self.image).suffix.lstrip(".")
        else:
            raise ValueError("Invalid image type")

    @model_validator(mode="before")
    def check_image_type(cls, data):
        if isinstance(data, dict) and "image" in data:
            if isinstance(data["image"], Image.Image):
                data["image"] = encode_image(data["image"], as_utf8=True)
            elif isinstance(data["image"], bytes):
                data["image"] = data["image"].decode("utf-8")
            elif isinstance(data["image"], str):
                try:
                    assert Path(data["image"]).is_file()
                    data["image"] = encode_image(Image.open(data["image"]), as_utf8=True)
                except OSError:
                    pass

        return data


class ToolCall(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    args: Any
    call_id: str | None = None


class ToolContent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    output: Any
    call_id: str | None = None
    args: Any | None = None
    funcname: str | None = None


class ApplicationFileContent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    file: FileType


ContentType = str | TextContent | ApplicationFileContent | ToolContent | ImageContent | ToolCall


class Message(BaseModel):
    role: str | None = None
    content: ContentType | list[ContentType]
    name: str | None = None


InputType = Message | list[Message] | ContentType | list[ContentType]
ConversationType = Message | list[Message]
