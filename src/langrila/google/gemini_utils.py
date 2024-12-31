import copy
from typing import Any

from pydantic import BaseModel

from ..core.tool import JsonSchemaHandler


def recurse_transform_type_to_upper(schema: dict[str, Any]) -> dict[str, Any]:
    new_schema = copy.deepcopy(schema)
    if isinstance(new_schema, dict):
        new_schema.pop("title", None)
        for key, value in new_schema.items():
            if isinstance(value, dict):
                new_schema[key] = recurse_transform_type_to_upper(value)
            elif isinstance(value, list):
                new_schema[key] = [recurse_transform_type_to_upper(v) for v in value]
            elif isinstance(value, str):
                if key == "type":
                    new_schema[key] = value.upper()
            else:
                new_schema[key] = value

    return new_schema


def to_gemini_schema(model: BaseModel) -> dict[str, Any]:
    return recurse_transform_type_to_upper(
        JsonSchemaHandler(schema=model.model_json_schema()).simplify()
    )
