from anthropic.types import ToolParam
from pydantic import BaseModel


class ToolProperty(BaseModel):
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


class ToolParameter(BaseModel):
    type: str = "object"
    properties: list[ToolProperty]
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


class ToolConfig(BaseModel):
    name: str
    description: str
    parameters: ToolParameter

    def format(self):
        dumped = self.model_dump(exclude=["parameters"])
        dumped["input_schema"] = self.parameters.format()
        return ToolParam(**dumped)
