import base64
import json
from typing import Any

from google.generativeai import protos

from ...base import BaseMessage
from ...message_content import ImageContent, Message, TextContent, ToolCall, ToolContent
from ...utils import decode_image


class GeminiMessage(BaseMessage):
    @property
    def as_user(self) -> protos.Content:
        return protos.Content(role="user", parts=self.contents)

    @property
    def as_assistant(self) -> protos.Content:
        return protos.Content(role="model", parts=self.contents)

    @property
    def as_function(self) -> protos.Content:
        return protos.Content(
            role="function",
            parts=self.contents,
        )

    @property
    def as_function_call(self) -> protos.Content:
        return protos.Content(role="model", parts=self.contents)

    @staticmethod
    def _format_text_content(content: TextContent) -> protos.Part:
        return protos.Part(text=content.text)

    @staticmethod
    def _format_image_content(content: ImageContent) -> protos.Part:
        file_format = decode_image(content.image, as_utf8=True).format.lower()
        _image_bytes = base64.b64decode(content.image.encode("utf-8"))

        image_bytes = protos.Blob(mime_type=f"image/{file_format}", data=_image_bytes)
        return protos.Part(inline_data=image_bytes)

    @staticmethod
    def _format_tool_content(content: ToolContent) -> protos.Part:
        return protos.Part(
            function_response=protos.FunctionResponse(
                name=content.funcname, response={"content": content.output}
            )
        )

    @staticmethod
    def _format_tool_call_content(content: ToolCall) -> protos.Part:
        return protos.Part(
            function_call=protos.FunctionCall(
                name=content.name,
                args=content.args if isinstance(content.args, dict) else json.loads(content.args),
            )
        )

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
            protos.Content.to_json(
                content,
                including_default_value_fields=False,
                use_integers_for_enums=False,
            )
        )
