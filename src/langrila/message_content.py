from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, model_validator

from .types import PathType
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
            return Path(self.image).suffix.lstrip(".").lower()
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
                except Exception:
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


class PDFContent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    file: PathType
    resolution_scale: float = 2.5
    image_resolution: Literal["auto", "low", "high"] = "high"

    def as_image_content(self) -> list[ImageContent]:
        from .file_utils.pdf import read_pdf_asimage

        return [
            ImageContent(image=image, resolution=self.image_resolution)
            for image in read_pdf_asimage(self.file, scale=self.resolution_scale)
        ]


class VideoContent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    file: PathType
    fps: float = 1.0
    image_resolution: Literal["auto", "low", "high"] = "low"

    def as_image_content(self) -> list[ImageContent]:
        from .file_utils.video import sample_frames

        return [
            ImageContent(image=image, resolution=self.image_resolution)
            for image in sample_frames(self.file, fps=self.fps)
        ]


class URIContent(BaseModel):
    uri: str
    mime_type: str | None = None

    @model_validator(mode="before")
    def check_mime_type(cls, data):
        if isinstance(data, dict) and "uri" in data:
            file_format = Path(data["uri"]).suffix.lstrip(".").lower()
            if file_format in ["jpg", "jpeg", "png", "heic", "heif"]:
                if file_format == "jpg":
                    file_format = "jpeg"
                data["mime_type"] = f"image/{file_format}"
            elif file_format in ["pdf", "json"]:
                data["mime_type"] = f"application/{file_format}"
            elif file_format in ["wav", "mp3", "aiff", "aac", "ogg", "flac"]:
                data["mime_type"] = f"audio/{file_format}"
            elif file_format in ["mp4", "mpeg", "mov", "avi", "wmv", "mpg"]:
                data["mime_type"] = f"video/{file_format}"
            elif file_format in ["txt", "html", "csv", "markdown", "xml"]:
                data["mime_type"] = f"text/{file_format}"
        return data


ContentType = (
    str
    | TextContent
    | PDFContent
    | ToolContent
    | ImageContent
    | ToolCall
    | URIContent
    | VideoContent
)


class Message(BaseModel):
    role: str | None = None
    content: ContentType | list[ContentType]
    name: str | None = None


InputType = Message | list[Message] | ContentType | list[ContentType]
ConversationType = Message | list[Message]
