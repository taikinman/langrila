import io
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator

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


class AudioContent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: PathType | bytes | np.ndarray
    sr: int | None = None
    mime_type: str | None = None

    def as_bytes(self) -> bytes:
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(
            buffer, self.data, samplerate=self.sr, format=self.mime_type.split("/")[-1].upper()
        )

        buffer.seek(0)
        return buffer.getvalue()

    @model_validator(mode="before")
    def setup(cls, data):
        try:
            import soundfile as sf
        except ModuleNotFoundError:
            return data

        if isinstance(data, dict) and "data" in data:
            try:
                assert Path(data["data"]).is_file()
                file_format = Path(data["data"]).suffix.lstrip(".").lower()
                if file_format in ["wav", "mp3", "aiff", "ogg", "flac"]:
                    data["mime_type"] = f"audio/{file_format}"
                elif file_format in ["mp4", "mpeg", "mov", "avi", "wmv", "mpg"]:
                    data["data"], data["sr"] = cls._from_video(data["data"])
                    data["mime_type"] = "audio/wav"
                else:
                    raise ValueError(f"Invalid audio file format: {file_format}")
            except Exception:
                pass

            if isinstance(data["data"], bytes):
                data["data"], data["sr"] = sf.read(io.BytesIO(data["data"]))
            elif isinstance(data["data"], io.BytesIO):
                data["data"], data["sr"] = sf.read(data["data"])
            elif isinstance(data["data"], (str | Path)):
                assert Path(data["data"]).is_file()
                data["data"], data["sr"] = sf.read(data["data"])
            elif isinstance(data["data"], np.ndarray):
                pass
            elif isinstance(data["data"], list):
                data["data"] = np.array(data["data"])
            else:
                raise ValueError("Invalid audio data type")

            data["data"] = cls._convert_stereo_to_mono(data["data"])

        return data

    @field_serializer("data")
    def serialize_data(self, data: np.ndarray) -> list:
        return data.tolist()

    @staticmethod
    def _convert_stereo_to_mono(data: np.ndarray) -> np.ndarray:
        if data.ndim == 2:
            return np.mean(data, axis=1)
        return data

    @staticmethod
    def _from_video(video_file: PathType) -> tuple[np.ndarray, int]:
        from moviepy.editor import VideoFileClip

        clip = VideoFileClip(Path(video_file).as_posix())
        audio = clip.audio
        fps = audio.fps
        audio_array = np.array(list(audio.iter_frames()))
        return audio_array, fps


class VideoContent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    file: PathType
    fps: float = 1.0
    image_resolution: Literal["auto", "low", "high"] = "low"
    include_audio: bool = False

    def as_image_content(self) -> list[ImageContent]:
        from .file_utils.video import sample_frames

        frames = [
            ImageContent(image=image, resolution=self.image_resolution)
            for image in sample_frames(self.file, fps=self.fps)
        ]

        if self.include_audio:
            frames.append(AudioContent(data=self.file))

        return frames


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
    | AudioContent
)


class Message(BaseModel):
    role: str | None = None
    content: ContentType | list[ContentType]
    name: str | None = None


InputType = Message | list[Message] | ContentType | list[ContentType]
ConversationType = Message | list[Message]
