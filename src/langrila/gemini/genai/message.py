import base64
import json
from typing import Any

from google.generativeai import protos
from google.generativeai.types import content_types
from google.generativeai.types.content_types import BlobDict, ContentDict, PartDict

from ...base import BaseMessage
from ...message_content import ImageContent, Message, TextContent, ToolCall, ToolContent
from ...utils import decode_image


class GeminiMessage(BaseMessage):
    @property
    def as_user(self) -> protos.Content:
        content_dict = ContentDict(role="user", parts=self.contents)
        return content_types.to_content(content_dict)

    @property
    def as_assistant(self) -> protos.Content:
        content_dict = ContentDict(role="model", parts=self.contents)
        return content_types.to_content(content_dict)

    @property
    def as_function(self) -> protos.Content:
        content_dict = ContentDict(
            role="function",
            parts=self.contents,
        )
        return content_types.to_content(content_dict)

    @property
    def as_function_call(self) -> protos.Content:
        content_dict = ContentDict(role="model", parts=self.contents)
        return content_types.to_content(content_dict)

    @staticmethod
    def _format_text_content(content: TextContent) -> PartDict:
        return PartDict(text=content.text)

    @staticmethod
    def _format_image_content(content: ImageContent) -> PartDict:
        file_format = decode_image(content.image, as_utf8=True).format.lower()
        _image_bytes = base64.b64decode(content.image.encode("utf-8"))

        image_bytes = BlobDict(mime_type=f"image/{file_format}", data=_image_bytes)
        return PartDict(inline_data=image_bytes)

    @staticmethod
    def _format_tool_content(content: ToolContent) -> protos.FunctionResponse:
        return protos.FunctionResponse(name=content.funcname, response={"content": content.output})

    @staticmethod
    def _format_tool_call_content(content: ToolCall) -> protos.FunctionCall:
        return protos.FunctionCall(name=content.name, args=content.args)

    @classmethod
    def from_client_message(cls, message: protos.Content) -> Message:
        serializable = cls._to_dict(message)

        common_contents = []

        for part in serializable.get("parts"):
            if part.get("text"):
                common_contents.append(TextContent(text=part.get("text")))
            elif part.get("inlineData"):
                inline_data = part.get("inlineData", {})
                mime_type = inline_data.get("mimeType")
                file_format = mime_type.split("/")[1]
                if file_format in ["jpeg", "png", "jpg"]:
                    image_data = inline_data.get("data")
                    common_contents.append(
                        ImageContent(
                            image=image_data,
                        )
                    )
            else:
                raise ValueError(f"Unsupported part type: {part.get('type')}")

        return Message(
            role=serializable.get("role").replace("model", "assistant"),
            content=common_contents,
            name=serializable.get("name"),
        )

    @staticmethod
    def _to_dict(content: protos.Content) -> dict[str, Any]:
        return json.loads(
            type(content).to_json(
                content,
                including_default_value_fields=False,
                use_integers_for_enums=False,
            )
        )
