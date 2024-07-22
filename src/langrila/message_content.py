from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict

from .types import FileType, PathType


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
