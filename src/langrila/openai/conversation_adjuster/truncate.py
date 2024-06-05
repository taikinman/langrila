from typing import Any

import tiktoken

from ...base import BaseConversationLengthAdjuster
from ..model_config import _VISION_MODEL, MODEL_CONFIG, MODEL_POINT
from ..openai_utils import get_encoding, get_n_tokens


class OldConversationTruncationModule(BaseConversationLengthAdjuster):
    """
    Adjust the number of tokens to be less than or equal to context_length, starting from the oldest message forward
    """

    def __init__(self, model_name: str, context_length: int):
        if model_name in MODEL_POINT.keys():
            print(f"{model_name} is automatically converted to {MODEL_POINT[model_name]}")
            model_name = MODEL_POINT[model_name]

        assert (
            model_name in MODEL_CONFIG.keys()
        ), f"model_name must be one of {', '.join(sorted(MODEL_CONFIG.keys()))}."

        self.model_name = model_name
        self.context_length = context_length
        self.encoding = get_encoding(self.model_name)

    def run(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        adjusted_messages: list[dict[str, str]] = []
        total_n_tokens: int = 0
        for message in messages[::-1]:
            if total_n_tokens <= self.context_length:
                message, total_n_tokens = self.adjust_message_length_and_update_total_tokens(
                    message, total_n_tokens
                )

                if message:
                    adjusted_messages.append(message)

                if message is None:
                    break

        return adjusted_messages[::-1]

    def adjust_message_length_and_update_total_tokens(
        self, message: dict[str, Any], total_n_tokens: int = 0
    ) -> str:
        n_tokens = get_n_tokens(message, self.model_name)
        if total_n_tokens + n_tokens["total"] <= self.context_length:
            total_n_tokens += n_tokens["total"]
            return message, total_n_tokens
        else:
            available_n_tokens = max(
                self.context_length - total_n_tokens - n_tokens["other"], 0
            )  # available_n_tokens for content
            if available_n_tokens > 0:
                if isinstance(message["content"], str):
                    message["content"] = self.truncate(message["content"], available_n_tokens)
                    total_n_tokens += available_n_tokens + n_tokens["other"]
                    print(
                        "Input message is truncated because total length of messages exceeds context length."
                    )
                    return message, total_n_tokens
                elif self.model_name in _VISION_MODEL and isinstance(message["content"], list):
                    return "", total_n_tokens  # truncate whole image
                else:
                    raise ValueError(
                        f"message['content'] must be str or list, but {type(message['content'])} is given."
                    )
            else:
                return None, total_n_tokens

    def truncate(self, text: str, n_tokens: int) -> str:
        if n_tokens > 0:
            return self.encoding.decode(self.encoding.encode(text)[-n_tokens:])
        else:
            return ""
