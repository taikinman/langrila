from typing import Optional

from pydantic import BaseModel


class ToolProperty(BaseModel):
    name: str
    type: str
    description: str
    enum: list[str | int | float] | None = None


class ToolParameter(BaseModel):
    type: str = "object"
    properties: list[ToolProperty]
    required: Optional[list[str]] = None


class ToolConfig(BaseModel):
    name: str
    description: str
    parameters: ToolParameter
