import base64
from typing import Any

from google.cloud.aiplatform_v1beta1.types import (
    content as gapic_content_types,
)
from google.cloud.aiplatform_v1beta1.types import tool as gapic_tool_types
from vertexai.generative_models import Content, Part

from ...base import BaseMessage
from ...message_content import ImageContent, Message, TextContent, ToolCall, ToolContent
from ...utils import decode_image


class CustomPart(Part):
    @staticmethod
    def from_function_call(name: str, args: dict[str, Any]) -> "Part":
        return Part._from_gapic(
            raw_part=gapic_content_types.Part(
                function_call=gapic_tool_types.FunctionCall(
                    name=name,
                    args=args,
                )
            )
        )


class VertexAIMessage(BaseMessage):
    @property
    def as_user(self) -> Content:
        return Content(role="user", parts=self.contents)

    @property
    def as_assistant(self) -> Content:
        return Content(role="model", parts=self.contents)

    @property
    def as_function(self) -> Content:
        return Content(role="user", parts=self.contents)

    @property
    def as_function_call(self) -> Content:
        return Content(role="model", parts=self.contents)

    @staticmethod
    def _format_text_content(content: TextContent) -> Part:
        return Part.from_text(text=content.text)

    @staticmethod
    def _format_image_content(content: ImageContent) -> Part:
        file_format = decode_image(content.image, as_utf8=True).format.lower()
        _image_bytes = base64.b64decode(content.image.encode("utf-8"))

        return Part.from_data(mime_type=f"image/{file_format}", data=_image_bytes)

    @staticmethod
    def _format_tool_content(content: ToolContent) -> Part:
        return Part.from_function_response(
            name=content.funcname, response={"content": content.output}
        )

    @staticmethod
    def _format_tool_call_content(content: ToolCall) -> Part:
        return CustomPart.from_function_call(name=content.name, args=content.args)

    @classmethod
    def from_client_message(cls, message: Content) -> Message:
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
    def _to_dict(content: Content) -> dict[str, Any]:
        return content.to_dict()
