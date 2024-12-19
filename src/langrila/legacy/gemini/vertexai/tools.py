from typing import Optional

from pydantic import BaseModel
from vertexai.generative_models import FunctionDeclaration

from ...tools import ToolConfig, ToolParameter, ToolProperty


class VertexAIToolProperty(ToolProperty):
    name: str
    type: str
    description: str
    enum: list[str] | None = None

    def format(self):
        return {self.name: self.model_dump(exclude=["name"], exclude_none=True)}


class VertexAIToolParameter(ToolParameter):
    type: str = "object"
    properties: list[VertexAIToolProperty]
    required: Optional[list[str]] = []

    def format(self):
        properties = {}
        for prop in self.properties:
            prop_dict = prop.format()
            for key, value in prop_dict.items():
                properties.update({key: value})

        return {
            "type": self.type,
            "properties": properties,
            "required": self.required,
        }


class VertexAIToolConfig(ToolConfig):
    name: str
    description: str
    parameters: VertexAIToolParameter

    def format(self):
        return FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=self.parameters.format(),
        )

    @classmethod
    def from_universal_configs(cls, configs: list[ToolConfig]) -> list["VertexAIToolConfig"]:
        return [cls(**config.model_dump(exclude_none=True)) for config in configs]
