from typing import Any, overload

from anthropic.types import (
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from anthropic.types.image_block_param import Source

from ..base import BaseMessage
from ..message_content import ImageContent, Message, TextContent, ToolCall, ToolContent
from ..utils import decode_image


class ClaudeMessage(BaseMessage):
    @property
    def as_user(self) -> MessageParam:
        return MessageParam(role="user", content=self.contents)

    @property
    def as_assistant(self) -> MessageParam:
        return MessageParam(role="assistant", content=self.contents)

    @property
    def as_function(self) -> MessageParam:
        return MessageParam(role="user", content=self.contents)

    @property
    def as_function_call(self) -> MessageParam:
        return MessageParam(role="assistant", content=self.contents)

    @staticmethod
    def _format_text_content(content: TextContent) -> TextBlockParam:
        return TextBlockParam(text=content.text, type="text")

    @staticmethod
    def _format_image_content(content: ImageContent) -> ImageBlockParam:
        file_format = decode_image(content.image, as_utf8=True).format.lower()

        return ImageBlockParam(
            type="image",
            source=Source(data=content.image, media_type=f"image/{file_format}", type="base64"),
        )

    @staticmethod
    def _format_tool_content(content: ToolContent) -> ToolResultBlockParam:
        return ToolResultBlockParam(
            tool_use_id=content.call_id,
            type="tool_result",
            content=content.output,
        )

    @overload
    @staticmethod
    def _format_tool_call_content(content: TextContent) -> TextBlockParam:
        ...

    @overload
    @staticmethod
    def _format_tool_call_content(content: ToolCall) -> ToolUseBlockParam:
        ...

    @staticmethod
    def _format_tool_call_content(
        content: TextContent | ToolCall
    ) -> TextBlockParam | ToolUseBlockParam:
        if isinstance(content, TextContent):
            return TextBlockParam(text=content.text, type="text")
        elif isinstance(content, ToolCall):
            return ToolUseBlockParam(
                id=content.call_id,
                type="tool_use",
                input=content.args,
                name=content.name,
            )
        else:
            raise ValueError("Invalid content type")

    @classmethod
    def from_client_message(cls, message: Any) -> Message:
        serializable = cls._to_dict(message)

        return Message(
            role=serializable.get("role"),
            content=serializable.get("content"),
            name=serializable.get("name"),
        )

    @staticmethod
    def _to_dict(message: MessageParam | dict) -> dict[str, Any]:
        return {
            "role": message["role"],
            "content": [c.to_dict() if hasattr(c, "to_dict") else c for c in message["content"]],
        }
