from typing import Optional

from pydantic import BaseModel, field_validator


class ToolProperty(BaseModel):
    name: str
    type: str
    description: str
    enum: list[str | int | float] | None = None

    def format(self):
        return {
            self.name: self.model_dump(exclude=["name"]) | {"enum": self.enum} if self.enum else {}
        }

    @field_validator("type")
    def check_type_value(cls, v):
        if v not in {"string", "number", "boolean"}:
            raise ValueError("type must be one of string or number.")

        return v


class ToolParameter(BaseModel):
    type: str = "object"
    properties: list[ToolProperty]
    required: Optional[list[str]] = None

    def format(self):
        dumped = self.model_dump(exclude=["properties", "required"])

        _properties = {}
        for p in self.properties:
            _properties.update(p.format())
        dumped["properties"] = _properties

        if self.required is not None:
            dumped["required"] = self.required
        return dumped

    @field_validator("type")
    def check_type_value(cls, v):
        if v not in {"object"}:
            raise ValueError("supported type is only object")

        return v

    @field_validator("required")
    def check_required_value(cls, required, values):
        properties = values.data["properties"]
        property_names = {p.name for p in properties}
        if required is not None:
            for r in required:
                if r not in property_names:
                    raise ValueError(f"required property '{r}' is not defined in properties.")
        return required


class ToolConfig(BaseModel):
    name: str
    type: str = "function"
    description: str
    parameters: ToolParameter

    def format(self):
        dumped = self.model_dump(exclude=["parameters", "type"])
        dumped["parameters"] = self.parameters.format()
        return {"type": self.type, self.type: dumped}

    @field_validator("type")
    def check_type_value(cls, v):
        if v not in {"function"}:
            raise ValueError("supported type is only function")

        return v
