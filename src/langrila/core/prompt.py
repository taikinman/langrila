import io
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from PIL import Image
from pydantic import field_serializer, model_validator

from ..utils import decode_image, encode_image, is_valid_uri
from .pydantic import BaseModel
from .typing import ArrayType, PathType

IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "heic", "heif"]
VIDEO_EXTENSIONS = ["mp4", "mpeg", "mov", "avi", "wmv", "mpg"]
AUDIO_EXTENSIONS = ["wav", "mp3", "aiff", "ogg", "flac"]


class TextPrompt(BaseModel):
    text: str


class ImagePrompt(BaseModel):
    image: Image.Image | PathType
    resolution: Literal["auto", "low", "high"] | None = None

    @property
    def format(self) -> str:
        if isinstance(self.image, Image.Image):
            if self.image.format:
                return self.image.format.lower()
            else:
                return "jpeg"
        elif isinstance(self.image, PathType):
            return Path(self.image).suffix.lstrip(".").lower()
        else:
            raise ValueError("Invalid image type")

    @model_validator(mode="before")
    @classmethod
    def check_image_type(cls: "ImagePrompt", data: Any) -> Any:  # type: ignore[misc]
        if isinstance(data, dict) and "image" in data:
            image = data["image"]

            if isinstance(image, (str | Path)):
                assert Path(image).is_file(), f"File not found: {data['image']}"
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = image.decode("utf-8")

            image = encode_image(image)

            data["image"] = image

        return data

    def decode(self) -> Image.Image:
        if isinstance(self.image, Image.Image):
            return self.image
        elif isinstance(self.image, str):
            return decode_image(self.image)
        else:
            raise ValueError("Invalid image type")


class PDFPrompt(BaseModel):
    pdf: PathType
    resolution_scale: float = 2.5
    image_resolution: Literal["auto", "low", "high"] = "high"

    def as_image_content(self) -> list[ImagePrompt]:
        from ..file_utils.pdf import read_pdf_asimage

        return [
            ImagePrompt(image=image, resolution=self.image_resolution)
            for image in read_pdf_asimage(self.pdf, scale=self.resolution_scale)
        ]


class AudioPrompt(BaseModel):
    audio: PathType | bytes | ArrayType
    sr: int | None = None
    mime_type: str | None = None

    def as_bytes(self) -> bytes:
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(
            file=buffer,
            data=self.audio,
            samplerate=self.sr,
            format=self.mime_type.split("/")[-1].upper() if self.mime_type else None,
        )

        buffer.seek(0)
        return buffer.getvalue()

    @model_validator(mode="before")
    @classmethod
    def setup(cls: "AudioPrompt", data: Any) -> Any:  # type: ignore[misc]
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

            data["data"] = cls._convert_stereo_to_mono(cast(ArrayType, data["data"]))

        return data

    @field_serializer("audio")
    def serialize_data(self, data: ArrayType) -> list[float]:
        return cast(list[float], data.tolist())

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


class VideoPrompt(BaseModel):
    video: PathType
    fps: float = 1.0
    image_resolution: Literal["auto", "low", "high"] = "low"
    # include_audio: bool = False

    def as_image_content(self) -> list[ImagePrompt]:
        from ..file_utils.video import sample_frames

        frames = [
            ImagePrompt(image=image, resolution=self.image_resolution)
            for image in sample_frames(self.video, fps=self.fps)
        ]

        # if self.include_audio:
        #     frames.append(AudioPrompt(audio=self.video))

        return frames


class URIPrompt(BaseModel):
    uri: str
    mime_type: str | None = None

    @model_validator(mode="before")
    @classmethod
    def check_mime_type(cls: "URIPrompt", data: Any) -> Any:  # type: ignore[misc]
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


class ToolUsePrompt(BaseModel):
    output: str | None = None
    error: str | None = None
    call_id: str | None = None
    args: str | None = None
    name: str | None = None
    tag: str | None = None

    @model_validator(mode="before")
    @classmethod
    def check_mime_type(cls: "ToolUsePrompt", data: Any) -> Any:  # type: ignore[misc]
        if isinstance(data, dict):
            if "output" not in data and "error" not in data:
                raise ValueError("Either 'output' or 'error' must be provided")
        return data


class ToolCallPrompt(BaseModel):
    name: str
    args: str
    call_id: str | None = None


PromptType = (
    str
    | TextPrompt
    | PDFPrompt
    | ImagePrompt
    | URIPrompt
    | VideoPrompt
    | AudioPrompt
    | ToolUsePrompt
    | ToolCallPrompt
)


Prompts = (
    TextPrompt,
    ImagePrompt,
    ToolUsePrompt,
    PDFPrompt,
    ToolCallPrompt,
    URIPrompt,
    VideoPrompt,
    AudioPrompt,
)


class Prompt(BaseModel):
    type: Literal["Prompt"] = "Prompt"
    role: Literal["system", "user", "assistant", "tool", "developer"]
    contents: PromptType | list[PromptType]
    name: str | None = None

    @model_validator(mode="before")
    @classmethod
    def setup(cls: "Prompt", data: Any) -> Any:  # type: ignore[misc]
        if isinstance(data, dict) and "contents" in data:
            if not isinstance(data["contents"], list):
                data["contents"] = [data["contents"]]

            data["contents"] = cls.to_universal_contents(data["contents"])

        return data

    @staticmethod
    def _string_to_universal_content(content: str) -> PromptType:
        try:
            is_file = Path(content).is_file()
            is_uri = is_valid_uri(content)
            if is_file:
                file_format = Path(content).suffix.lstrip(".").lower()

                if file_format in IMAGE_EXTENSIONS:
                    return ImagePrompt(image=content)
                elif file_format in ["pdf"]:
                    return PDFPrompt(pdf=content)
                elif file_format in VIDEO_EXTENSIONS:
                    return VideoPrompt(video=content)
                elif file_format in AUDIO_EXTENSIONS:
                    return AudioPrompt(audio=content)
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")
            elif is_uri:
                return URIPrompt(uri=content)
            else:
                return TextPrompt(text=content)
        except OSError:
            try:
                decode_image(content)
                return ImagePrompt(image=content)
            except Exception:
                return TextPrompt(text=content)

    @staticmethod
    def _to_universal_contents(
        contents: list[PromptType] | list[dict[str, Any]],
    ) -> list[PromptType]:
        _contents: list[PromptType] = []

        for content in contents:
            if isinstance(content, str):
                content = Prompt._string_to_universal_content(content=content)

            if isinstance(
                content,
                Prompts,
            ):
                _contents.append(content)
            elif isinstance(content, dict):
                for prompt in Prompts:
                    try:
                        _contents.append(prompt(**content))
                        break
                    except Exception:
                        pass
            else:
                raise NotImplementedError(f"Prompt type {type(content)} not implemented")

        return _contents

    @staticmethod
    def to_universal_contents(
        contents: PromptType | list[PromptType] | Image.Image,
    ) -> list[PromptType]:
        if isinstance(contents, PromptType):
            return Prompt._to_universal_contents(contents=[contents])
        elif isinstance(contents, list):
            return Prompt._to_universal_contents(contents=contents)
        elif isinstance(contents, Image.Image):
            return [ImagePrompt(image=contents)]
        else:
            raise ValueError("Invalid contents type.")
