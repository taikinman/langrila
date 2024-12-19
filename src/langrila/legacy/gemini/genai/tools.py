from typing import Optional

from google.ai.generativelanguage import FunctionDeclaration, Schema, Type

from ...tools import ToolConfig, ToolParameter, ToolProperty


class GeminiToolProperty(ToolProperty):
    name: str
    type: str
    description: str
    enum: list[str] | None = None

    def format(self):
        return {
            self.name: Schema(
                type=getattr(Type, self.type.upper()),
                **self.model_dump(exclude=["name", "type"], exclude_none=True),
            ),
        }


class GeminiToolParameter(ToolParameter):
    type: str = "object"
    properties: list[GeminiToolProperty]
    required: Optional[list[str]] = []

    def format(self):
        properties = {}
        for prop in self.properties:
            prop_dict = prop.format()
            for key, value in prop_dict.items():
                properties.update({key: value})

        return Schema(
            type=getattr(Type, self.type.upper()),
            properties=properties,
            required=self.required,
        )


class GeminiToolConfig(ToolConfig):
    name: str
    description: str
    parameters: GeminiToolParameter

    def format(self):
        return FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=self.parameters.format(),
        )

    @classmethod
    def from_universal_configs(cls, configs: list[ToolConfig]) -> list["GeminiToolConfig"]:
        return [cls(**config.model_dump(exclude_none=True)) for config in configs]
