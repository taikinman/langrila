from typing import Optional

from pydantic import BaseModel
from vertexai.generative_models import FunctionDeclaration


class ToolProperty(BaseModel):
    name: str
    type: str
    description: str
    enum: list[str] | None = None

    def format(self):
        return {self.name: {"type": self.type, "description": self.description, "enum": self.enum}}


class ToolParameter(BaseModel):
    type: str = "object"
    properties: list[ToolProperty]
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


class ToolConfig(BaseModel):
    name: str
    description: str
    parameters: ToolParameter

    def format(self):
        return FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=self.parameters.format(),
        )
