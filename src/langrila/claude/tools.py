from anthropic.types import ToolParam
from pydantic import BaseModel

from ..tools import ToolConfig, ToolParameter, ToolProperty


class ClaudeToolProperty(ToolProperty):
    name: str
    type: str
    description: str
    enum: list[str | int | float] | None = None

    def format(self):
        return {self.name: self.model_dump(exclude=["name"], exclude_none=True)}


class ClaudeToolParameter(ToolParameter):
    type: str = "object"
    properties: list[ClaudeToolProperty]
    required: list[str | int | float] | None = None

    def format(self):
        dumped = self.model_dump(exclude=["properties"], exclude_none=True)

        _properties = {}
        for p in self.properties:
            _properties.update(p.format())
        dumped["properties"] = _properties

        return dumped


class ClaudeToolConfig(ToolConfig):
    name: str
    description: str
    parameters: ClaudeToolParameter

    def format(self):
        dumped = self.model_dump(exclude=["parameters"], exclude_none=True)
        dumped["input_schema"] = self.parameters.format()
        return ToolParam(**dumped)

    @classmethod
    def from_universal_configs(cls, configs: list[ToolConfig]) -> list["ClaudeToolConfig"]:
        return [cls(**config.model_dump(exclude_none=True)) for config in configs]
