from collections import UserDict
from typing import Any

from pydantic import BaseModel, field_validator


class Usage(BaseModel):
    model_name: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: Any) -> "Usage":
        if isinstance(other, dict):
            other = Usage(**other)

        if (
            hasattr(other, "model_name")
            and other.model_name
            and self.model_name != other.model_name
        ):
            raise ValueError("model_name must be the same: {self.model_name} != {other.model_name}")

        if hasattr(other, "prompt_tokens"):
            prompt_tokens = self.prompt_tokens + other.prompt_tokens
        else:
            prompt_tokens = self.prompt_tokens
        if hasattr(other, "completion_tokens"):
            completion_tokens = self.completion_tokens + other.completion_tokens
        else:
            completion_tokens = self.completion_tokens
        return Usage(
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def __sub__(self, other: Any) -> "Usage":
        if isinstance(other, dict):
            other = Usage(**other)

        if (
            hasattr(other, "model_name")
            and other.model_name
            and self.model_name != other.model_name
        ):
            raise ValueError("model_name must be the same: {self.model_name} != {other.model_name}")

        if hasattr(other, "prompt_tokens"):
            prompt_tokens = self.prompt_tokens - other.prompt_tokens
        else:
            prompt_tokens = self.prompt_tokens
        if hasattr(other, "completion_tokens"):
            completion_tokens = self.completion_tokens - other.completion_tokens
        else:
            completion_tokens = self.completion_tokens
        return Usage(
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @field_validator("prompt_tokens")
    def check_prompt_tokens(cls, v):
        if v < 0:
            raise ValueError("prompt_tokens must be greater or equal to 0")
        return v

    @field_validator("completion_tokens")
    def check_completion_tokens(cls, v):
        if v < 0:
            raise ValueError("completion_tokens must be greater or equal to 0")
        return v

    def __repr__(self):
        return f"Usage(prompt_tokens={self.prompt_tokens}, completion_tokens={self.completion_tokens}, total_tokens={self.total_tokens})"


class TokenCounter(UserDict):
    def __init__(self, model_names: str | list[str] | None = None):
        tokens = {}
        if model_names is not None:
            if not isinstance(model_names, list):
                model_names = [model_names]

            for model_name in model_names:
                tokens[model_name] = Usage(model_name=model_name)

        super().__init__(tokens)

    def __add__(self, other: Usage):
        if other.model_name:
            if other.model_name not in self.data:
                self.data[other.model_name] = Usage(model_name=other.model_name)
            self.data[other.model_name] += other
        else:
            model_name = "unknown_model"
            if model_name not in self.data:
                self.data[model_name] = Usage(model_name=model_name)
            self.data[model_name] += other
        return self

    def __setitem__(self, model_name: str, usage: Usage):
        if model_name not in self.data:
            self.data[model_name] = Usage(model_name=model_name)
        self.data[model_name] += usage
