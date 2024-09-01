from abc import ABC, abstractmethod
from inspect import isfunction, ismethod
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Generator

from PIL import Image
from pydantic import BaseModel

from .message_content import (
    AudioContent,
    ContentType,
    ImageContent,
    InputType,
    Message,
    PDFContent,
    TextContent,
    ToolCall,
    ToolContent,
    URIContent,
    VideoContent,
)
from .result import (
    CompletionResults,
    EmbeddingResults,
    FunctionCallingResults,
    ToolCallResponse,
    ToolOutput,
)
from .types import RoleType
from .utils import decode_image, is_valid_uri, model2func

ROLES = ["system", "user", "assistant", "function", "function_call", "tool"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "heic", "heif"]
VIDEO_EXTENSIONS = ["mp4", "mpeg", "mov", "avi", "wmv", "mpg"]
AUDIO_EXTENSIONS = ["wav", "mp3", "aiff", "ogg", "flac"]


class BaseChatModule(ABC):
    @abstractmethod
    def run(self, messages: list[dict[str, str]]) -> CompletionResults:
        raise NotImplementedError

    async def arun(self, messages: list[dict[str, str]]) -> CompletionResults:
        raise NotImplementedError

    def stream(self, messages: list[dict[str, str]]) -> Generator[CompletionResults, None, None]:
        raise NotImplementedError

    async def astream(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[CompletionResults, None]:
        raise NotImplementedError


class BaseFunctionCallingModule(ABC):
    @abstractmethod
    def run(self, messages: list[dict[str, str]]) -> FunctionCallingResults:
        raise NotImplementedError

    async def arun(self, messages: list[dict[str, str]]) -> FunctionCallingResults:
        raise NotImplementedError

    def _set_runnable_tools_dict(self, tools: list[Callable | BaseModel]) -> dict[str, callable]:
        return {f.__name__: f if (isfunction(f) or ismethod(f)) else model2func(f) for f in tools}


class BaseEmbeddingModule(ABC):
    @abstractmethod
    def run(self, text: str | list[str]) -> EmbeddingResults:
        raise NotImplementedError

    async def arun(self, text: str | list[str]) -> EmbeddingResults:
        raise NotImplementedError


class BaseConversationLengthAdjuster(ABC):
    @abstractmethod
    def run(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        raise NotImplementedError


class BaseFilter(ABC):
    @abstractmethod
    def apply(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        raise NotImplementedError

    def aapply(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def restore(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        raise NotImplementedError


class BaseConversationMemory(ABC):
    @abstractmethod
    def store(self, conversation_history: list[dict[str, str]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> list[dict[str, str]]:
        raise NotImplementedError


class BaseMessage(ABC):
    def __init__(self, contents: list[dict[str, Any]], name: str | None = None):
        self.contents = contents
        self.name = name

    @property
    def as_system(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def as_user(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def as_assistant(self):
        raise NotImplementedError

    @property
    def as_function(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _format_text_content(content: TextContent) -> Any:
        raise NotImplementedError

    @staticmethod
    def _format_image_content(content: ImageContent) -> Any:
        raise NotImplementedError

    @staticmethod
    def _format_tool_content(content: ToolContent) -> Any:
        raise NotImplementedError

    @staticmethod
    def _format_tool_call_content(content: ToolCall) -> Any:
        raise NotImplementedError

    @staticmethod
    def _format_uri_content(content: str | Path) -> Any:
        raise NotImplementedError

    @staticmethod
    def _format_audio_content(content: str | Path) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _from_completion_results(cls, results: CompletionResults) -> list[dict[str, str]]:
        raise NotImplementedError

    @classmethod
    def _from_completion_results(cls, response: CompletionResults) -> Message:
        return cls.from_client_message(response.message)

    @classmethod
    def _to_client_contents(
        cls,
        contents: ContentType | list[ContentType],
    ) -> list[dict[str, Any]]:
        if not isinstance(contents, list):
            contents = [contents]

        _contents = []

        for content in contents:
            if isinstance(content, TextContent):
                _contents.append(cls._format_text_content(content=content))
            elif isinstance(content, ImageContent):
                try:
                    _contents.append(cls._format_image_content(content=content))
                except NotImplementedError:
                    pass
            elif isinstance(content, ToolContent):
                try:
                    _contents.append(cls._format_tool_content(content=content))
                except NotImplementedError:
                    pass
            elif isinstance(content, ToolCall):
                try:
                    _contents.append(cls._format_tool_call_content(content=content))
                except NotImplementedError:
                    pass
            elif isinstance(content, (PDFContent, VideoContent)):
                try:
                    _contents.extend(
                        [
                            cls._format_image_content(content=image_content)
                            for image_content in content.as_image_content()
                        ]
                    )
                except NotImplementedError:
                    pass
            elif isinstance(content, URIContent):
                try:
                    _contents.append(cls._format_uri_content(content=content))
                except NotImplementedError:
                    pass
            elif isinstance(content, AudioContent):
                try:
                    _contents.append(cls._format_audio_content(content=content))
                except NotImplementedError:
                    pass
            else:
                raise NotImplementedError(f"Message type {type(content)} not implemented")

        return _contents

    @classmethod
    def to_client_message(
        cls,
        message: Message,
    ) -> Any:
        if message.role not in ROLES:
            raise ValueError(f"Invalid role: {message.role}")

        _contents = cls._to_client_contents(contents=message.content)

        return getattr(cls(contents=_contents, name=message.name), "as_" + message.role)

    @classmethod
    @abstractmethod
    def from_client_message(cls, message: Any) -> Message:
        raise NotImplementedError

    @classmethod
    def _string_to_universal_content(cls, content: str) -> ContentType:
        try:
            is_file = Path(content).is_file()
            is_uri = is_valid_uri(content)
            file_format = Path(content).suffix.lstrip(".").lower()
            if is_file:
                if file_format in IMAGE_EXTENSIONS:
                    return ImageContent(image=content)
                elif file_format in ["pdf"]:
                    return PDFContent(file=content)
                elif file_format in VIDEO_EXTENSIONS:
                    return VideoContent(file=content)
                elif file_format in AUDIO_EXTENSIONS:
                    return AudioContent(data=content)
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")
            elif is_uri:
                return URIContent(uri=content)
            else:
                return TextContent(text=content)
        except OSError:
            try:
                decode_image(content, as_utf8=True)
                return ImageContent(image=content)
            except Exception:
                return TextContent(text=content)

    @classmethod
    def to_universal_contents(
        cls,
        contents: ContentType | list[ContentType],
    ) -> list[ContentType]:
        if not isinstance(contents, list):
            contents = [contents]

        _contents = []

        for content in contents:
            if isinstance(content, str):
                _string_content = cls._string_to_universal_content(content=content)
                if isinstance(_string_content, list):
                    _contents.extend(_string_content)
                else:
                    _contents.append(_string_content)
            elif isinstance(
                content,
                (
                    TextContent,
                    ImageContent,
                    ToolContent,
                    PDFContent,
                    ToolCall,
                    URIContent,
                    VideoContent,
                    AudioContent,
                ),
            ):
                _contents.append(content)
            else:
                raise NotImplementedError(f"Message type {type(content)} not implemented")

        return _contents

    @classmethod
    def to_universal_message(
        cls,
        message: InputType,
        role: RoleType | None = None,
        name: str | None = None,
    ) -> Message:
        if isinstance(message, Message):
            message.content = cls.to_universal_contents(contents=message.content)
            return message
        elif isinstance(message, ContentType):
            if role is None:
                raise ValueError("Role must be provided to create a message")

            message = Message(role=role, content=message, name=name)
            message.content = cls.to_universal_contents(contents=message.content)
            return message
        elif isinstance(message, list):
            if role is None:
                raise ValueError("Role must be provided to create a message")

            contents = []
            for m in message:
                _content = cls.to_universal_message(role=role, message=m, name=name).content
                contents.extend(_content)

            message = Message(role=role, content=contents, name=name)
            return message
        elif isinstance(message, Image.Image):
            return Message(
                role=role,
                content=[ImageContent(image=message)],
                name=name,
            )
        elif isinstance(message, dict):
            message = Message(**message)
            message.content = cls.to_universal_contents(contents=message.content)
            return message
        else:
            raise ValueError(f"Invalid message type {type(message)}")

    @classmethod
    def to_universal_message_from_completion_response(
        cls, response: CompletionResults
    ) -> dict[str, Any]:
        return cls._from_completion_results(response)

    @classmethod
    def to_universal_message_from_function_response(
        cls, response: FunctionCallingResults
    ) -> list[Message]:
        return [
            Message(
                role="function",  # global role
                content=[
                    ToolContent(
                        output=result.output,
                        args=result.args,
                        call_id=result.call_id,
                        funcname=result.funcname,
                    )
                ],
                name=result.funcname,
            )
            if isinstance(result, ToolOutput)
            else Message(
                role="assistant",  # global role
                content=[result],
            )
            for result in response.results
        ]

    @classmethod
    def to_universal_message_from_function_call(cls, response: FunctionCallingResults) -> Message:
        return Message(
            role="function_call",  # global role
            content=[
                ToolCall(
                    args=result.args,
                    name=result.name,
                    call_id=result.call_id,
                )
                if isinstance(result, ToolCallResponse)
                else result
                for result in response.calls
            ],
        )

    @staticmethod
    def _preprocess_message(messages: list[Message]) -> list[Message]:
        return messages


class BaseMetadataStore(ABC):
    @abstractmethod
    def store(self, ids: list[int], metadatas: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def fetch(self, ids: list[int]) -> list[dict[str, Any]]:
        raise NotImplementedError


class BaseMetadataFilter(ABC):
    @abstractmethod
    def run(self, metadata: dict[str, str]) -> bool:
        raise NotImplementedError
