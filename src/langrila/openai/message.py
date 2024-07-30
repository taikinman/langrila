import re
from typing import Any

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_message_tool_call import Function

from ..base import BaseMessage
from ..message_content import ImageContent, Message, TextContent, ToolCall, ToolContent
from ..utils import decode_image, encode_image


class OpenAIMessage(BaseMessage):
    @property
    def as_user(self):
        return ChatCompletionUserMessageParam(
            role="user",
            content=self.contents,
            name=self.name if self.name else "User",
        )

    @property
    def as_assistant(self):
        return ChatCompletionAssistantMessageParam(
            role="assistant",
            content=self.contents,
            name=self.name if self.name else "Assistant",
        )

    @property
    def as_system(self):
        return ChatCompletionSystemMessageParam(
            role="system",
            content=self.contents,
            name=self.name if self.name else "System",
        )

    @property
    def as_function(self):
        return {
            "role": "tool",
            "tool_call_id": self.contents[0]["tool_call_id"],
            "name": self.contents[0]["name"],
            "content": self.contents[0]["content"],
        }

    @property
    def as_function_call(self) -> dict[str, Any]:
        return ChatCompletionMessage(
            role="assistant",
            tool_calls=[content for content in self.contents if content["type"] == "function"],
        ).model_dump()

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
    def _format_tool_content(content: ToolContent) -> dict[str, str]:
        return {
            "tool_call_id": "call_" + content.call_id.split("_")[-1],
            "name": content.funcname,
            "content": content.output,
        }

    @staticmethod
    def _format_tool_call_content(content: ToolCall) -> ChatCompletionMessageToolCall:
        return ChatCompletionMessageToolCall(
            id="call_" + content.call_id.split("_")[-1],
            type="function",
            function=Function(arguments=str(content.args), name=content.name),
        ).model_dump()

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

    @staticmethod
    def _preprocess_message(messages: list[Message]) -> list[Message]:
        new_messages = []
        for message in messages:
            role = message.role
            name = message.name
            content = message.content

            if role == "function":
                for content in message.content:
                    new_messages.append(
                        Message(
                            role=role,
                            content=[content],
                            name=name,
                        )
                    )
            else:
                new_messages.append(message)
        return new_messages
