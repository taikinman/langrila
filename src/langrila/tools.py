from typing import Any, Optional

from pydantic import BaseModel


class ToolProperty(BaseModel):
    name: str
    type: str
    description: str
    enum: list[str | int | float] | None = None
    items: dict[str, Any] | None = None


class ToolParameter(BaseModel):
    type: str = "object"
    properties: list[ToolProperty]
    required: Optional[list[str]] = None


class ToolConfig(BaseModel):
    name: str
    description: str
    parameters: ToolParameter
    strict: bool | None = None

    @classmethod
    def from_pydantic(cls, model: BaseModel, **kwargs) -> "ToolConfig":
        configs = {}
        params = {}

        configs["name"] = model.__name__
        schema = model.model_json_schema()
        defs = schema.pop("$defs", {})
        params["required"] = schema.pop("required", None)
        params["type"] = schema.pop("type", "object")
        configs["description"] = schema.pop("description", "No description.")

        properties = schema.get("properties", {})
        props = []
        for k, v in properties.items():
            if ref := v.get("$ref"):
                ref = ref.split("/")[-1]
                v = defs[ref]

            props.append(
                {
                    "name": k,
                    "type": v.get("type"),
                    "description": v.get("description"),
                    "enum": v.get("enum"),
                }
            )
        params["properties"] = props
        configs["parameters"] = params

        return cls(**configs, **kwargs)
