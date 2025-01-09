from __future__ import annotations as _annotations

import copy
import inspect
import json
import re
from inspect import signature
from typing import Any, Callable, Literal, cast, get_type_hints

import pydantic
from griffe import Docstring, DocstringSectionKind
from griffe import Object as GriffeObject
from pydantic import (
    Field,
    ValidationInfo,
    create_model,
    field_validator,
    model_validator,
)

from .pydantic import BaseModel

ToolType = Callable[..., Any]
DocstringStyle = Literal["google", "numpy", "sphinx"]


def get_docstring_and_field_descriptions(
    func: ToolType, *, style: DocstringStyle | None = None
) -> tuple[str, dict[str, str]]:
    """
    Extract the function description and parameter descriptions from a function's docstring.
    This function is heavily based on the v0.0.12 of :
    https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/_griffe.py .
    The original script is licensed under the MIT License.

    Returns:
        A tuple of (main function description, parameter descriptions).
    """
    doc = func.__doc__
    if doc is None:
        return "", {}

    sig = signature(func)

    # see https://github.com/mkdocstrings/griffe/issues/293
    parent = cast(GriffeObject, sig)

    docstring = Docstring(doc, lineno=1, parser=style or _infer_docstring_style(doc), parent=parent)
    sections = docstring.parse()

    params = {}
    if parameters := next((p for p in sections if p.kind == DocstringSectionKind.parameters), None):
        params = {p.name: p.description for p in parameters.value}

    main_desc = ""
    if main := next((p for p in sections if p.kind == DocstringSectionKind.text), None):
        main_desc = main.value

    return main_desc, params


def _infer_docstring_style(doc: str) -> DocstringStyle:
    """Simplistic docstring style inference."""
    for pattern, replacements, style in _docstring_style_patterns:
        matches = (
            re.search(pattern.format(replacement), doc, re.IGNORECASE | re.MULTILINE)
            for replacement in replacements
        )
        if any(matches):
            return style
    # fallback to google style
    return "google"


# See https://github.com/mkdocstrings/griffe/issues/329#issuecomment-2425017804
_docstring_style_patterns: list[tuple[str, list[str], DocstringStyle]] = [
    (
        r"\n[ \t]*:{0}([ \t]+\w+)*:([ \t]+.+)?\n",
        [
            "param",
            "parameter",
            "arg",
            "argument",
            "key",
            "keyword",
            "type",
            "var",
            "ivar",
            "cvar",
            "vartype",
            "returns",
            "return",
            "rtype",
            "raises",
            "raise",
            "except",
            "exception",
        ],
        "sphinx",
    ),
    (
        r"\n[ \t]*{0}:([ \t]+.+)?\n[ \t]+.+",
        [
            "args",
            "arguments",
            "params",
            "parameters",
            "keyword args",
            "keyword arguments",
            "other args",
            "other arguments",
            "other params",
            "other parameters",
            "raises",
            "exceptions",
            "returns",
            "yields",
            "receives",
            "examples",
            "attributes",
            "functions",
            "methods",
            "classes",
            "modules",
            "warns",
            "warnings",
        ],
        "google",
    ),
    (
        r"\n[ \t]*{0}\n[ \t]*---+\n",
        [
            "deprecated",
            "parameters",
            "other parameters",
            "returns",
            "yields",
            "receives",
            "raises",
            "warns",
            "attributes",
            "functions",
            "methods",
            "classes",
            "modules",
        ],
        "numpy",
    ),
]


class FunctionValidator:
    def __init__(self, func: Callable[..., Any], context: dict[str, Any] | None = None):
        self.func = func
        self.context = context if context is not None else {}

        description, field_descriptions = get_docstring_and_field_descriptions(func)
        fields = self._get_fields(func, field_descriptions)

        self.description = description
        self.field_descriptions = field_descriptions
        self.model = self._create_model_for_func(func, fields)
        self.schema = self._create_partial_schema(func, description, fields, self.context)

    def _get_fields(
        self,
        func: Callable[..., Any],
        field_descriptions: dict[str, str],
    ) -> dict[str, tuple[type, Any]]:
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Create field definitions to pass to create_model
        fields = {}
        for name, param in sig.parameters.items():
            # Raise error if there is no type annotation for the argument
            if annotation := type_hints.get(name, Any):
                if annotation is sig.empty:
                    raise ValueError(
                        f"Argument '{name}' of function '{func.__name__}' has no type annotation."
                    )

                if param.default is sig.empty:
                    # No default value -> required
                    field = Field(..., description=field_descriptions.get(name, ""))
                else:
                    # Default value exists
                    field = Field(
                        default=param.default, description=field_descriptions.get(name, "")
                    )

                fields[name] = (annotation, field)
            else:
                raise ValueError(
                    f"Argument '{name}' of function '{func.__name__}' has no type annotation."
                )
        return fields

    def _create_model_for_func(
        self,
        func: Callable[..., Any],
        fields: dict[str, tuple[type, Any]],
    ) -> type[pydantic.BaseModel]:
        DynamicModel = create_model(
            f"{func.__name__}_ArgsModel",
            __base__=BaseModel,
            **fields,
        )
        return cast(type[pydantic.BaseModel], DynamicModel)

    def validate(self, obj: dict[str, Any]) -> dict[str, Any]:
        return self.model.model_validate({**obj, **self.context}).__dict__

    def _create_partial_schema(
        self,
        func: Callable[..., Any],
        description: str,
        fields: dict[str, tuple[type, Any]],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        fields_not_in_context = {}

        # delete key which is in context
        for key, field in fields.items():
            if key not in context:
                fields_not_in_context[key] = field

        # create model
        schema_model = create_model(
            f"{func.__name__}Schema",
            __base__=BaseModel,
            __doc__=description,
            **fields_not_in_context,
        )

        # create schema
        schema = schema_model.model_json_schema()
        return JsonSchemaHandler(schema).simplify()


class Tool(BaseModel):
    """
    A tool is a function that can be run with arguments. It can be used to run a function with
    arguments and validate them using a JSON schema.

    Parameters
    ----------
    tool : Callable[..., Any]
        The function to run.
    name : str, optional
        The name of the tool. If not provided, the name of the function will be used.
    description : str, optional
        The description of the tool. If not provided, the function's docstring will be used.
    context : dict[str, Any], optional
        The dict of the arguments injected to the function. The context arguments will not be generated by the
        llm, but are validated.
    schema_dict : dict[str, Any], optional
        The JSON schema to validate the arguments. If not provided, the schema will be generated
        from the function's type hints.
    validator : FunctionValidator, optional
        The schema validator to use. If not provided, a new schema validator will be created.
    serializer : Callable[[Any], str], optional
        Callable to serialize the result of the tool. If not provided, the result will be converted to a string.
    """

    tool: Callable[..., Any] | None = None
    name: str | None = None
    description: str | None = None
    context: dict[str, Any] | None = None
    schema_dict: dict[str, Any] | None = None
    validator: FunctionValidator | None = None
    serializer: Callable[[Any], str] = str

    @model_validator(mode="before")
    def setup(cls, data: dict[str, Any]) -> dict[str, Any]:
        if data.get("tool") is not None:
            if data.get("validator"):
                validator = data["validator"]
            else:
                validator = FunctionValidator(data["tool"], data.get("context"))

            data["name"] = data["tool"].__name__
        else:
            validator = None
            data["name"] = data.get("name")

        data["validator"] = data.get("validator") or validator
        data["description"] = data.get("description") or validator.description

        if data.get("schema_dict") is not None:
            data["schema_dict"] = JsonSchemaHandler(data["schema_dict"]).simplify()
        else:
            if validator:
                data["schema_dict"] = validator.schema

        return data

    @field_validator("tool", "schema_dict", "context")
    @classmethod
    def validate_tool_or_schema(cls: "Tool", v: Any, info: ValidationInfo) -> Any:  # type: ignore[misc]
        if info.field_name == "tool" and v is not None:
            return v
        if info.field_name in {"schema_dict", "context"} and v is not None:
            if isinstance(v, str):
                v = json.loads(v)
            return v

        # This validator is applied to both fields, so if both are None, it will generate an error
        if info.context is None or info.context.get("validated_once", None) is not None:
            raise ValueError("Either 'tool' or 'schema_dict' must be not None")

        # Record that one of them has already been Validated
        info.context["validated_once"] = True
        return v

    def run(self, args: str | dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(self.tool):
            raise ValueError("Cannot run async tool with run method")

        if isinstance(args, str):
            args = cast(dict[str, Any], json.loads(args))

        assert self.tool is not None, "Tool is not set"
        assert self.validator is not None, "Validator is not set"
        valid_args = self.validator.validate(args)

        return self.serializer(self.tool(**valid_args))

    async def run_async(self, args: str | dict[str, Any]) -> Any:
        if isinstance(args, str):
            args = cast(dict[str, Any], json.loads(args))

        assert self.tool is not None, "Tool is not set"
        assert self.validator is not None, "Validator is not set"
        valid_args = self.validator.validate(args)

        if inspect.iscoroutinefunction(self.tool):
            result = await self.tool(**valid_args)
            return self.serializer(result)
        else:
            return self.serializer(self.tool(**valid_args))


# This class is heavily based on the PydanticAI v0.0.12.
# See: https://github.com/pydantic/pydantic-ai
class JsonSchemaHandler:
    def __init__(self, schema: dict[str, Any]):
        self.schema = copy.deepcopy(schema)
        self.defs = self.schema.pop("$defs", {})

    def simplify(self) -> dict[str, Any]:
        self._simplify(self.schema, refs_stack=())
        return self.schema

    def _simplify(self, schema: dict[str, Any], refs_stack: tuple[str, ...]) -> None:
        schema.pop("title", None)
        schema.pop("default", None)
        if ref := schema.pop("$ref", None):
            # noinspection PyTypeChecker
            key = re.sub(r"^#/\$defs/", "", ref)
            if key in refs_stack:
                raise ValueError("Recursive `$ref`s in JSON Schema are not supported.")
            refs_stack += (key,)
            schema_def = self.defs[key]
            self._simplify(schema_def, refs_stack)
            schema.update(schema_def)
            return

        if any_of := schema.get("anyOf"):
            for schema in any_of:
                self._simplify(schema, refs_stack)

        type_ = schema.get("type")

        if type_ == "object":
            self._object(schema, refs_stack)
        elif type_ == "array":
            return self._array(schema, refs_stack)

    def _object(self, schema: dict[str, Any], refs_stack: tuple[str, ...]) -> None:
        ad_props = schema.pop("additionalProperties", None)
        if ad_props:
            raise ValueError("Additional properties in JSON Schema are not supported.")

        if properties := schema.get("properties"):  # pragma: no branch
            for value in properties.values():
                self._simplify(value, refs_stack)

    def _array(self, schema: dict[str, Any], refs_stack: tuple[str, ...]) -> None:
        if prefix_items := schema.get("prefixItems"):
            # TODO I think this not is supported by Gemini, maybe we should raise an error?
            for prefix_item in prefix_items:
                self._simplify(prefix_item, refs_stack)

        if items_schema := schema.get("items"):  # pragma: no branch
            self._simplify(items_schema, refs_stack)
