import base64
import io
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from PIL import Image
from pydantic import model_validator

from ..utils import decode_image, encode_image
from .pydantic import BaseModel
from .typing import ArrayType, PathType
from .usage import NamedUsage, Usage


class TextResponse(BaseModel):
    text: str


class ToolCallResponse(BaseModel):
    name: str | None = None
    args: str | None = None
    call_id: str | None = None


class ImageResponse(BaseModel):
    image: str | Image.Image
    format: str | None = None

    @model_validator(mode="before")
    @classmethod
    def check_image_type(cls: "ImageResponse", data: Any) -> Any:  # type: ignore[misc]
        if isinstance(data, dict) and isinstance(data.get("image"), Image.Image):
            data["image"] = encode_image(image=cast(Image.Image, data.get("image")))

        return data

    def decode(self) -> Image.Image:
        return decode_image(cast(str, self.image))

    def __str__(self) -> str:
        if isinstance(self.image, str):
            return f"ImageResponse(image={self.image[:20]}..., format={self.format})"
        else:
            raise ValueError("Invalid image type")

    def __repr__(self) -> str:
        return self.__str__()


class AudioResponse(BaseModel):
    audio: PathType | bytes | ArrayType | str
    sr: int | None = None
    mime_type: str | None = None
    audio_id: str | None = None

    def asarray(self) -> ArrayType:
        import soundfile as sf

        if isinstance(self.audio, np.ndarray):
            return self.audio
        elif isinstance(self.audio, bytes):
            return cast(ArrayType, sf.read(io.BytesIO(self.audio))[0])
        elif isinstance(self.audio, str):
            return cast(ArrayType, sf.read(io.BytesIO(base64.b64decode(self.audio)))[0])
        elif isinstance(self.audio, Path):
            return cast(ArrayType, sf.read(self.audio)[0])
        else:
            raise ValueError("Invalid audio type")

    def asbytes(self) -> bytes:
        if isinstance(self.audio, bytes):
            return self.audio
        elif isinstance(self.audio, str):
            return base64.b64decode(self.audio)
        else:
            raise ValueError("Invalid audio type")

    @staticmethod
    def to_bytes(audio: ArrayType, sr: int, mime_type: str) -> bytes:
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(
            file=buffer,
            data=audio,
            samplerate=sr,
            format=mime_type.split("/")[-1].upper() if mime_type else None,
        )

        buffer.seek(0)
        return buffer.read()

    @model_validator(mode="before")
    @classmethod
    def setup(cls: "AudioResponse", data: Any) -> Any:  # type: ignore[misc]
        try:
            import soundfile as sf
        except ModuleNotFoundError:
            return data

        if isinstance(data, dict) and "audio" in data:
            try:
                assert Path(data["audio"]).is_file()
                file_format = Path(data["audio"]).suffix.lstrip(".").lower()
                if file_format in ["wav", "mp3", "aiff", "ogg", "flac"]:
                    data["mime_type"] = f"audio/{file_format}"
                elif file_format in ["mp4", "mpeg", "mov", "avi", "wmv", "mpg"]:
                    data["audio"], data["sr"] = cls._from_video(data["audio"])
                    data["mime_type"] = "audio/wav"
                else:
                    raise ValueError(f"Invalid audio file format: {file_format}")
            except OSError:
                data["mime_type"] = "audio/wav"
            else:
                pass

            if isinstance(data["audio"], (str, Path)):
                try:
                    assert Path(data["audio"]).is_file()
                    if isinstance(data["audio"], Path):
                        data["audio"] = data["audio"].as_posix()
                    data["audio"], data["sr"] = sf.read(data["audio"])
                except OSError:
                    data["audio"], data["sr"] = sf.read(io.BytesIO(base64.b64decode(data["audio"])))
                else:
                    pass

            if isinstance(data["audio"], bytes):
                data["audio"], data["sr"] = sf.read(io.BytesIO(data["audio"]))
            elif isinstance(data["audio"], io.BytesIO):
                data["audio"], data["sr"] = sf.read(data["audio"])
            elif isinstance(data["audio"], np.ndarray):
                pass
            elif isinstance(data["audio"], list):
                data["audio"] = np.array(data["audio"])
            else:
                raise ValueError("Invalid audio data type")

            if isinstance(data["audio"], np.ndarray):
                assert data.get("sr") is not None, "Sample rate must be provided"
                data["audio"] = cls._convert_stereo_to_mono(cast(ArrayType, data["audio"]))
                audio_bytes = cls.to_bytes(
                    data["audio"],
                    data["sr"],
                    data["mime_type"],
                )
                data["audio"] = base64.b64encode(audio_bytes).decode("utf-8")
        return data

    def __str__(self) -> str:
        if isinstance(self.audio, str):
            return f"AudioResponse(audio={self.audio[:20]}..., sr={self.sr}, mime_type={self.mime_type})"
        else:
            raise ValueError("Invalid audio type")

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def _convert_stereo_to_mono(data: ArrayType) -> ArrayType:
        if data.ndim == 2:
            return cast(ArrayType, np.mean(data, axis=1))
        return data

    @staticmethod
    def _from_video(video_file: PathType) -> tuple[ArrayType, int]:
        from moviepy.editor import VideoFileClip

        clip = VideoFileClip(Path(video_file).as_posix())
        audio = clip.audio
        fps = audio.fps
        audio_array = np.array(list(audio.iter_frames()))
        return audio_array, fps


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
    contents: list[ResponseType]
    usage: Usage | NamedUsage | None = None
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
