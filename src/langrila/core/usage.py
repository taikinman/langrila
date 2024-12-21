from typing import Any

from .pydantic import BaseModel


class Usage(BaseModel):
    model_name: str | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens
