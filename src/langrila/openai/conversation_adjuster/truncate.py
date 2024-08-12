from typing import Any

from ...base import BaseConversationLengthAdjuster
from ..model_config import MODEL_CONFIG, MODEL_POINT
from ..openai_utils import get_encoding, get_n_tokens


class OldConversationTruncationModule(BaseConversationLengthAdjuster):
    """
    Adjust the number of tokens to be less than or equal to context_length, starting from the oldest message forward

    FIXME: Truncation is not accurate especially when tool is called.
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

    def run(self, messages: list[dict[str, dict[str, str]]]) -> list[dict[str, dict[str, str]]]:
        adjusted_messages: list[dict[str, dict[str, str]]] = []
        total_n_tokens: int = 0

        used_tool_call_ids = []
        used_tool_use_message = []
        total_tool_use_tokens = 0
        for message in messages[::-1]:
            if message.get("role") == "tool":
                n_tool_use_tokens = get_n_tokens(message, self.model_name)["total"]
                total_tool_use_tokens += n_tool_use_tokens
                if total_n_tokens + total_tool_use_tokens <= self.context_length:
                    used_tool_use_message.append(message)
                    continue
                else:
                    new_message = None
                    break
            elif "tool_calls" in message:
                for n_iter in range(1, len(message["tool_calls"]) + 1):
                    tool_calls = message["tool_calls"][-n_iter:]
                    _tmp_message = {k: v for k, v in message.items() if k != "tool_calls"}
                    _tmp_message["tool_calls"] = tool_calls

                    total_tool_call_tokens = get_n_tokens(_tmp_message, self.model_name)["total"]
                    if (
                        total_n_tokens + total_tool_use_tokens + total_tool_call_tokens
                        <= self.context_length
                    ):
                        used_tool_call_ids = [tool_call["id"] for tool_call in tool_calls]
                        tool_call_message = _tmp_message
                    else:
                        break

                if used_tool_call_ids:
                    new_message = [
                        tool_use_message
                        for tool_use_message in used_tool_use_message
                        if tool_use_message["tool_call_id"] in used_tool_call_ids
                    ]
                    new_message.append(tool_call_message)
                    for m in new_message:
                        n_tokens = get_n_tokens(m, self.model_name)["total"]
                        total_n_tokens += n_tokens

                else:
                    new_message = None
                    break
            else:
                new_message, total_n_tokens = self.adjust_message_length_and_update_total_tokens(
                    message, total_n_tokens
                )

            if new_message:
                if isinstance(new_message, dict):
                    adjusted_messages.append(new_message)
                elif isinstance(new_message, list):
                    adjusted_messages.extend(new_message)

                used_tool_call_ids = []
                used_tool_use_message = []
                total_tool_use_tokens = 0

            if new_message is None:
                break

        return adjusted_messages[::-1]

    def _truncate_text(self, text: str, available_n_tokens: int) -> tuple[dict[str, str], int]:
        new_text = self.truncate(text, available_n_tokens)
        print("Input message is truncated because total length of messages exceeds context length.")
        return new_text

    def _to_text_message(self, text) -> dict[str, str]:
        return {"type": "text", "text": text}

    def adjust_message_length_and_update_total_tokens(
        self, message: dict[str, dict[str, str]], total_n_tokens: int = 0
    ) -> dict[str, Any]:
        n_tokens = get_n_tokens(message, self.model_name)

        if total_n_tokens + n_tokens["total"] <= self.context_length:
            total_n_tokens += n_tokens["total"]
            return message, total_n_tokens
        else:
            available_n_tokens = max(
                self.context_length - total_n_tokens - n_tokens["other"], 0
            )  # available_n_tokens for content
            if available_n_tokens > 0:
                new_message = {}
                for key, value in message.items():
                    if key == "role":
                        new_message[key] = value
                    elif key == "name":
                        new_message[key] = value
                    elif key == "content" and isinstance(value, list):
                        new_contents = []
                        for item in value:
                            if item.get("type") == "image_url":
                                pass  # Image is entirely truncated
                            elif item.get("type") == "text":
                                total_n_tokens += available_n_tokens + n_tokens["other"]
                                new_text = self._truncate_text(
                                    text=item["text"],
                                    available_n_tokens=available_n_tokens,
                                )

                                new_contents.append({"type": "text", "text": new_text})
                            else:
                                raise ValueError(
                                    f"Unknown type {item['type']} in message['content']."
                                )
                        new_message[key] = new_contents

                    else:
                        raise ValueError(f"Unknown type {item['type']} in message['content'].")

                return new_message, total_n_tokens

            else:
                return None, total_n_tokens

    def truncate(self, text: str, n_tokens: int) -> str:
        if n_tokens > 0:
            return self.encoding.decode(self.encoding.encode(text)[-n_tokens:])
        else:
            return ""
