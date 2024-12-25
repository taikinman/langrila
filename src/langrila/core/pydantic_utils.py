from __future__ import annotations as _annotations

from types import GenericAlias
from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from pydantic.json_schema import JsonSchemaValue
from typing_extensions import TypeAlias

ObjectJsonSchema: TypeAlias = dict[str, Any]


def is_model_like(type_: Any) -> bool:
    """Check if something is a pydantic model.

    These should all generate a JSON Schema with `{"type": "object"}` and therefore be usable directly as
    function parameters.
    """
    return (
        isinstance(type_, type)
        and not isinstance(type_, GenericAlias)
        and issubclass(type_, PydanticBaseModel)
    )


def check_object_json_schema(schema: JsonSchemaValue) -> ObjectJsonSchema:
    if schema.get("type") == "object":
        return schema
    else:
        raise ValueError("Schema must be an object")
