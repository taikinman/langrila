from typing import Any, Optional

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL

from ..base import BaseMessage
from ..utils import encode_image


class OpenAIMessage(BaseMessage):
    def __init__(
        self,
        content: str,
        name: Optional[str] = None,
        images: Any | list[Any] | None = None,
        image_resolution: str | None = None,
    ):
        super().__init__(content=content, images=images, name=name)
        self._valid_image_resolution_value(image_resolution)

        self.image_resolution = image_resolution

    @property
    def as_system(self):
        return ChatCompletionSystemMessageParam(role="system", content=self.content)

    @property
    def as_user(self):
        if self.images:
            content = [ChatCompletionContentPartTextParam(text=self.content, type="text")]
            if not isinstance(self.images, list):
                images = [self.images]
            else:
                images = self.images

            for image in images:
                content.append(
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url=ImageURL(
                            url=f"data:image/jpeg;base64,{encode_image(image)}",
                            detail=self.image_resolution if self.image_resolution else "low",
                        ),
                    )
                )
            return ChatCompletionUserMessageParam(role="user", content=content, name=self.name)
        else:
            return ChatCompletionUserMessageParam(role="user", content=self.content, name=self.name)

    @property
    def as_assistant(self):
        return ChatCompletionAssistantMessageParam(role="assistant", content=self.content)

    # @property
    # def as_tool(self):
    #     return {"role": "tool", "content": self.content}

    @property
    def as_function(self):
        return ChatCompletionFunctionMessageParam(
            role="function", name=self.name, content=self.content
        )

    def _valid_image_resolution_value(self, image_resolution: str) -> None:
        if image_resolution:
            if image_resolution not in ["low", "high"]:
                raise ValueError(
                    "image_resolution must be either 'low' or 'high' due to token management."
                )
