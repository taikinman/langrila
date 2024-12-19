from typing import Any, Literal, cast

from PIL import Image
from pydantic import model_validator

from ..utils import decode_image, encode_image
from .prompt import Prompt
from .pydantic import BaseModel
from .usage import Usage


class TextResponse(BaseModel):
    text: str


class ToolCallResponse(BaseModel):
    name: str | None = None
    args: str | None = None
    call_id: str | None = None


class ImageResponse(BaseModel):
    image: str | Image.Image

    @model_validator(mode="before")
    @classmethod
    def check_image_type(cls: "ImageResponse", data: Any) -> Any:  # type: ignore[misc]
        if isinstance(data, dict) and isinstance(data.get("image"), Image.Image):
            data["image"] = encode_image(image=cast(Image.Image, data.get("image")))

        return data

    def decode(self) -> Image.Image:
        return decode_image(cast(str, self.image))


class AudioResponse(BaseModel):
    audio: str


class VideoResponse(BaseModel):
    video: str


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]


ResponseType = str | TextResponse | ImageResponse | AudioResponse | VideoResponse | ToolCallResponse

Responses = (
    TextResponse,
    ImageResponse,
    AudioResponse,
    VideoResponse,
    ToolCallResponse,
)


class Response(BaseModel):
    type: Literal["Response"] = "Response"
    role: Literal["assistant"] = "assistant"
    contents: list[ResponseType] | None = None
    usage: Usage
    raw: Any | None = None
    name: str | None = None
    is_last_chunk: bool | None = None
    prompt: Any | None = None

    @model_validator(mode="before")
    @classmethod
    def setup(cls: "Response", data: Any) -> Any:  # type: ignore[misc]
        if isinstance(data, dict) and "contents" in data:
            if not isinstance(data["contents"], list):
                data["contents"] = [data["contents"]]

            data["contents"] = cls.to_universal_contents(data["contents"])

            if data.get("usage") is None:
                data["usage"] = Usage()

        return data

    @staticmethod
    def _to_universal_contents(
        contents: list[ResponseType] | list[dict[str, Any]] | list[Image.Image],
    ) -> list[ResponseType]:
        _contents: list[ResponseType] = []

        for content in contents:
            if isinstance(content, str):
                content = TextResponse(text=content)
            elif isinstance(content, Image.Image):
                content = ImageResponse(image=content)

            if isinstance(
                content,
                (
                    TextResponse,
                    ImageResponse,
                    ToolCallResponse,
                    VideoResponse,
                    AudioResponse,
                ),
            ):
                _contents.append(content)

            elif isinstance(content, dict):
                for response in Responses:
                    try:
                        _contents.append(response(**content))
                        break
                    except Exception:
                        pass
            else:
                raise NotImplementedError(f"Response type {type(content)} not implemented")

        return _contents

    @staticmethod
    def to_universal_contents(
        contents: list[ResponseType] | Image.Image,
    ) -> list[ResponseType]:
        if isinstance(contents, (ResponseType, list)):
            return Response._to_universal_contents(contents=contents)
        elif isinstance(contents, Image.Image):
            return [ImageResponse(image=contents)]
        else:
            raise ValueError("Invalid contents type.")
