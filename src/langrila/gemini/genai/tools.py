from typing import Optional

import google.generativeai as genai
from pydantic import BaseModel


class ToolProperty(BaseModel):
    name: str
    type: str
    description: str
    enum: list[str] | None = None

    def format(self):
        return {
            self.name: genai.protos.Schema(
                type=getattr(genai.protos.Type, self.type.upper()),
                description=self.description,
                enum=self.enum,
            ),
        }


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

        return genai.protos.Schema(
            type=getattr(genai.protos.Type, self.type.upper()),
            properties=properties,
            required=self.required,
        )


class ToolConfig(BaseModel):
    name: str
    description: str
    parameters: ToolParameter

    def format(self):
        return genai.protos.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=self.parameters.format(),
        )
