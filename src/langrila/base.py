from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncGenerator, Generator

from PIL import Image

from .message_content import (
    ApplicationFileContent,
    ContentType,
    ImageContent,
    InputType,
    Message,
    TextContent,
    ToolCall,
    ToolContent,
)
from .result import (
    CompletionResults,
    EmbeddingResults,
    FunctionCallingResults,
    ToolCallResponse,
    ToolOutput,
)
from .types import RoleType
from .utils import decode_image


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
    def _format_application_file_content(content: ApplicationFileContent) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _from_completion_results(cls, results: CompletionResults) -> list[dict[str, str]]:
        raise NotImplementedError

    @classmethod
    def _from_completion_results(cls, response: CompletionResults) -> Message:
        return cls.from_client_message(response.message)

    @classmethod
    def _format_string_content(cls, content: str) -> Any:
        if Path(content).is_file() and Path(content).suffix in [".png", ".jpg", ".jpeg"]:
            return cls._format_image_content(content=ImageContent(image=content))
        elif Path(content).is_file() and Path(content).suffix in [".pdf"]:
            return cls._format_application_file_content(
                content=ApplicationFileContent(file=content)
            )
        else:
            return cls._format_text_content(content=TextContent(text=content))

    @classmethod
    def _format_contents(
        cls,
        contents: ContentType | list[ContentType],
    ) -> list[dict[str, Any]]:
        if not isinstance(contents, list):
            contents = [contents]

        _contents = []

        for content in contents:
            if isinstance(content, str):
                _contents.append(cls._format_string_content(content=content))
            elif isinstance(content, TextContent):
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
            elif isinstance(content, ApplicationFileContent):
                try:
                    _contents.append(cls._format_application_file_content(content=content))
                except NotImplementedError:
                    pass
            else:
                raise NotImplementedError(f"Message type {type(content)} not implemented")

        return _contents

    @classmethod
    def _format_message(
        cls,
        role: RoleType,
        contents: ContentType | list[ContentType],
        name: str | None = None,
    ) -> Any:
        if role not in ["user", "assistant", "system", "function", "function_call", "tool"]:
            raise ValueError(f"Invalid role: {role}")

        _contents = cls._format_contents(contents=contents)

        return getattr(cls(contents=_contents, name=name), "as_" + role)

    @classmethod
    def to_client_message(
        cls,
        message: Message,
    ) -> Any:
        return cls._format_message(role=message.role, contents=message.content, name=message.name)

    @classmethod
    @abstractmethod
    def from_client_message(cls, message: Any) -> Message:
        raise NotImplementedError

    @classmethod
    def _string2content(cls, content: str) -> ContentType:
        try:
            if Path(content).is_file() and Path(content).suffix in [".png", ".jpg", ".jpeg"]:
                return ImageContent(image=content)
            elif Path(content).is_file() and Path(content).suffix in [".pdf"]:
                return ApplicationFileContent(file=content)
            else:
                return TextContent(text=content)
        except OSError:
            try:
                decode_image(content, as_utf8=True)
                return ImageContent(image=content)
            except Exception:
                return TextContent(text=content)

    @classmethod
    def to_contents(
        cls,
        contents: ContentType | list[ContentType],
    ) -> list[ContentType]:
        if not isinstance(contents, list):
            contents = [contents]

        _contents = []

        for content in contents:
            if isinstance(content, str):
                _contents.append(cls._string2content(content=content))
            elif isinstance(
                content, (TextContent, ImageContent, ToolContent, ApplicationFileContent, ToolCall)
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
            message.content = cls.to_contents(contents=message.content)
            return message
        elif isinstance(message, ContentType):
            if role is None:
                raise ValueError("Role must be provided to create a message")

            message = Message(role=role, content=message, name=name)
            message.content = cls.to_contents(contents=message.content)
            return message
        elif isinstance(message, list):
            if role is None:
                raise ValueError("Role must be provided to create a message")

            contents = [
                cls.to_universal_message(role=role, message=m, name=name).content[0]
                for m in message
            ]
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
            message.content = cls.to_contents(contents=message.content)
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
