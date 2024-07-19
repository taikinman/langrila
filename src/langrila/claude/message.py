from typing import Any

from anthropic.types import (
    ImageBlockParam,
    MessageParam,
    TextBlock,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
)
from anthropic.types.image_block_param import Source

from ..base import BaseMessage
from ..utils import encode_image


class ClaudeMessage(BaseMessage):
    def __init__(
        self,
        content: Any | None = None,
        call_id: str | None = None,
        name: str | None = None,
        images: Any | list[Any] | None = None,
    ):
        super().__init__(content=content, images=images, name=name)
        self.call_id = call_id

    @property
    def as_system(self):
        raise NotImplementedError

    @property
    def as_user(self):
        contents = []
        if self.images:
            if not isinstance(self.images, list):
                images = [self.images]
            else:
                images = self.images

            for image in images:
                image_bytes = encode_image(image)
                contents.append(
                    ImageBlockParam(
                        type="image",
                        source=Source(data=image_bytes, media_type="image/jpeg", type="base64"),
                    )
                )

        contents.append(TextBlockParam(text=self.content, type="text"))
        return MessageParam(role="user", content=contents)

    @property
    def as_assistant(self):
        return MessageParam(
            role="assistant", content=[TextBlockParam(text=self.content, type="text")]
        )

    @property
    def as_tool_result(self):
        return MessageParam(
            role="user",
            content=[
                ToolResultBlockParam(
                    tool_use_id=c.call_id, type="tool_result", content=c.output["tool_result"]
                )
                for c in self.content
                if c.output["content"].type == "tool_use"
            ],
        )

    @property
    def as_function(self):
        return MessageParam(role="assistant", content=self.content)

    @staticmethod
    def to_dict(message: MessageParam | dict) -> dict[str, Any]:
        return {
            "role": message["role"],
            "content": [c.to_dict() if hasattr(c, "to_dict") else c for c in message["content"]],
        }

    @staticmethod
    def from_dict(message: dict[str, Any]):
        for content in message["content"]:
            if content["type"] == "text":
                content = TextBlockParam(**content)
            elif content["type"] == "tool_use":
                content = ToolUseBlockParam(**content)
            elif content["type"] == "tool_result":
                content = ToolResultBlockParam(**content)
            elif content["type"] == "image":
                content = ImageBlockParam(**content)
            else:
                raise ValueError(f"Unknown content type {content['type']}")

        return MessageParam(role=message["role"], content=message["content"])
