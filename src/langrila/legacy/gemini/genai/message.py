import base64
import json
from typing import Any

from google.ai.generativelanguage import (
    Blob,
    Content,
    FileData,
    FunctionCall,
    FunctionResponse,
    Part,
)

from ...base import BaseMessage
from ...message_content import (
    AudioContent,
    ImageContent,
    Message,
    TextContent,
    ToolCall,
    ToolContent,
    URIContent,
)
from ...utils import decode_image


class GeminiMessage(BaseMessage):
    @property
    def as_user(self) -> Content:
        return Content(role="user", parts=self.contents)

    @property
    def as_assistant(self) -> Content:
        return Content(role="model", parts=self.contents)

    @property
    def as_function(self) -> Content:
        return Content(
            role="function",
            parts=self.contents,
        )

    @property
    def as_function_call(self) -> Content:
        return Content(role="model", parts=self.contents)

    @staticmethod
    def _format_text_content(content: TextContent) -> Part:
        return Part(text=content.text)

    @staticmethod
    def _format_image_content(content: ImageContent) -> Part:
        file_format = decode_image(content.image, as_utf8=True).format.lower()
        _image_bytes = base64.b64decode(content.image.encode("utf-8"))

        image_bytes = Blob(mime_type=f"image/{file_format}", data=_image_bytes)
        return Part(inline_data=image_bytes)

    @staticmethod
    def _format_uri_content(content: URIContent) -> Part:
        file_data = FileData(file_uri=content.uri, mime_type=content.mime_type)
        return Part(file_data=file_data)

    @staticmethod
    def _format_audio_content(content: AudioContent) -> Part:
        _audio_bytes = content.as_bytes()
        audio_blob = Blob(data=_audio_bytes, mime_type=content.mime_type)
        return Part(inline_data=audio_blob)

    @staticmethod
    def _format_tool_content(content: ToolContent) -> Part:
        return Part(
            function_response=FunctionResponse(
                name=content.funcname, response={"content": content.output}
            )
        )

    @staticmethod
    def _format_tool_call_content(content: ToolCall) -> Part:
        return Part(
            function_call=FunctionCall(
                name=content.name,
                args=content.args if isinstance(content.args, dict) else json.loads(content.args),
            )
        )

    @classmethod
    def from_client_message(cls, message: Content) -> Message:
        serializable = cls._to_dict(message)

        common_contents = []

        for part in serializable.get("parts", []):
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
        return json.loads(
            Content.to_json(
                content,
                including_default_value_fields=False,
                use_integers_for_enums=False,
            )
        )
