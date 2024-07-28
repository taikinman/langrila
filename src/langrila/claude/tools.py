from anthropic.types import ToolParam
from pydantic import BaseModel

from ..tools import ToolConfig, ToolParameter, ToolProperty


class ClaudeToolProperty(ToolProperty):
    name: str
    type: str
    description: str
    enum: list[str | int | float] | None = None

    def format(self):
        return {
            self.name: self.model_dump(
                exclude=["name"] if self.enum else ["name", "enum"],
            )
        }


class ClaudeToolParameter(ToolParameter):
    type: str = "object"
    properties: list[ClaudeToolProperty]
    required: list[str | int | float] | None = None

    def format(self):
        dumped = self.model_dump(exclude=["properties", "required"])

        _properties = {}
        for p in self.properties:
            _properties.update(p.format())
        dumped["properties"] = _properties

        if self.required is not None:
            dumped["required"] = self.required
        return dumped


class ClaudeToolConfig(ToolConfig):
    name: str
    description: str
    parameters: ClaudeToolParameter

    def format(self):
        dumped = self.model_dump(exclude=["parameters"])
        dumped["input_schema"] = self.parameters.format()
        return ToolParam(**dumped)

    @classmethod
    def from_universal_configs(cls, configs: list[ToolConfig]) -> list["ClaudeToolConfig"]:
        return [cls(**config.model_dump()) for config in configs]
