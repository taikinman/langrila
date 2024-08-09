from pydantic import field_validator

from ..tools import ToolConfig, ToolParameter, ToolProperty


class OpenAIToolProperty(ToolProperty):
    name: str
    type: str
    description: str
    enum: list[str | int | float] | None = None
    items: dict[str, str] | None = None

    def format(self):
        return {self.name: self.model_dump(exclude=["name"], exclude_none=True)}

    @field_validator("type")
    def check_type_value(cls, v):
        if v not in {"string", "number", "boolean", "array"}:
            raise ValueError("type must be one of string or number.")

        return v


class OpenAIToolParameter(ToolParameter):
    type: str = "object"
    properties: list[OpenAIToolProperty]
    required: list[str] | None = None

    def format(self):
        dumped = self.model_dump(exclude=["properties"], exclude_none=True)

        _properties = {}
        for p in self.properties:
            _properties.update(p.format())
        dumped["properties"] = _properties

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


class OpenAIToolConfig(ToolConfig):
    name: str
    type: str = "function"
    description: str
    parameters: OpenAIToolParameter
    strict: bool | None = None

    def format(self):
        dumped = self.model_dump(exclude=["parameters", "type", "strict"], exclude_none=True)
        dumped["parameters"] = self.parameters.format()

        output = {"type": self.type, self.type: dumped}

        if self.strict is not None:
            output["strict"] = self.strict

        return output

    @field_validator("type")
    def check_type_value(cls, v):
        if v not in {"function"}:
            raise ValueError("supported type is only function")

        return v

    @classmethod
    def from_universal_configs(cls, configs: list[ToolConfig]) -> list["OpenAIToolConfig"]:
        return [cls(**config.model_dump(exclude_none=True)) for config in configs]
