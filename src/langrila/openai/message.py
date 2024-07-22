import re
from typing import Any

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL

from ..base import BaseMessage
from ..message_content import ImageContent, Message, TextContent, ToolContent
from ..utils import decode_image, encode_image


class OpenAIMessage(BaseMessage):
    @property
    def as_user(self):
        return ChatCompletionUserMessageParam(
            role="user",
            content=self.contents,
            name=self.name,
        )

    @property
    def as_assistant(self):
        return ChatCompletionAssistantMessageParam(
            role="assistant",
            content=self.contents,
            name=self.name,
        )

    @property
    def as_system(self):
        return ChatCompletionSystemMessageParam(
            role="system",
            content=self.contents,
            name=self.name,
        )

    @property
    def as_function(self):
        return ChatCompletionToolMessageParam(
            role="function",
            content=self.contents,
            name=self.name,
        )

    @staticmethod
    def _format_text_content(content: TextContent) -> ChatCompletionContentPartTextParam:
        return ChatCompletionContentPartTextParam(
            type="text",
            text=content.text,
        )

    @staticmethod
    def _format_image_content(content: ImageContent) -> ChatCompletionContentPartImageParam:
        file_format = decode_image(content.image, as_utf8=True).format.lower()
        return ChatCompletionContentPartImageParam(
            type="image_url",
            image_url=ImageURL(
                url=f"data:image/{file_format};base64,{content.image}",
                detail=content.resolution if content.resolution else "auto",
            ),
        )

    @staticmethod
    def _format_tool_content(content: ToolContent) -> ChatCompletionContentPartTextParam:
        return ChatCompletionContentPartTextParam(
            type="text",
            text=content.output,
        )

    @classmethod
    def from_client_message(cls, message: dict[str, Any]) -> Message:
        contents = message.get("content", [])
        common_contents = []
        for content in contents:
            if content["type"] == "text":
                common_contents.append(TextContent(text=content["text"]))
            elif content["type"] == "image_url":
                pattern = re.compile("^(data:image/.+;base64,)")
                url = pattern.sub("", content["image_url"]["url"])
                common_contents.append(
                    ImageContent(
                        image=decode_image(url, as_utf8=True),
                        resolution=content["image_url"]["detail"],
                    )
                )

            else:
                raise ValueError(f"Unknown content type: {content['type']}")

        return Message(
            role=message.get("role"),
            content=common_contents,
            name=message.get("name"),
        )
