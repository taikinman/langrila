from __future__ import annotations as _annotations

import copy
import inspect
import json
import re
from inspect import Parameter, signature
from typing import Any, Callable, Literal, cast

from griffe import Docstring, DocstringSectionKind
from griffe import Object as GriffeObject
from pydantic import ConfigDict, ValidationInfo, field_validator, model_validator
from pydantic._internal import _decorators, _generate_schema, _typing_extra
from pydantic._internal._config import ConfigWrapper
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema
from pydantic.plugin._schema_validator import create_schema_validator
from pydantic_core import SchemaValidator, core_schema

from .pydantic import BaseModel
from .pydantic_utils import check_object_json_schema, is_model_like

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


# This function is heavily based on the PydanticAI v0.0.12.
# See: https://github.com/pydantic/pydantic-ai
def function_schema(function: ToolType) -> tuple[str, dict[str, Any], SchemaValidator]:
    """Build a Pydantic validator and JSON schema from a tool function.

    Args:
        function: The function to build a validator and JSON schema for.
        takes_ctx: Whether the function takes a `RunContext` first argument.

    Returns:
        A `FunctionSchema` instance.
    """
    config = ConfigDict(title=function.__name__, arbitrary_types_allowed=True)
    config_wrapper = ConfigWrapper(config)
    gen_schema = _generate_schema.GenerateSchema(config_wrapper)

    sig = signature(function)

    type_hints = _typing_extra.get_function_type_hints(function)

    var_kwargs_schema: core_schema.CoreSchema | None = None
    fields: dict[str, core_schema.TypedDictField] = {}
    positional_fields: list[str] = []
    decorators = _decorators.DecoratorInfos()
    description: str
    field_descriptions: dict[str, Any]
    description, field_descriptions = get_docstring_and_field_descriptions(function)

    for name, p in sig.parameters.items():
        if p.annotation is sig.empty:
            # TODO warn?
            annotation = Any
        else:
            annotation = type_hints[name]

        field_name = p.name
        if p.kind == Parameter.VAR_KEYWORD:
            var_kwargs_schema = gen_schema.generate_schema(annotation)
        else:
            if p.kind == Parameter.VAR_POSITIONAL:
                annotation = list[annotation]

            # FieldInfo.from_annotation expects a type, `annotation` is Any
            annotation = cast(type[Any], annotation)
            field_info = FieldInfo.from_annotation(annotation)
            if field_info.description is None:
                field_info.description = field_descriptions.get(field_name)

            fields[field_name] = td_schema = gen_schema._generate_td_field_schema(  # type: ignore
                field_name,
                field_info,
                decorators,
            )
            # noinspection PyTypeChecker
            td_schema.setdefault("metadata", {})["is_model_like"] = is_model_like(annotation)

            if p.kind == Parameter.POSITIONAL_ONLY:
                positional_fields.append(field_name)

    core_config = config_wrapper.core_config(None)
    # noinspection PyTypedDict
    core_config["extra_fields_behavior"] = "allow" if var_kwargs_schema else "forbid"

    schema, _ = _build_schema(fields, var_kwargs_schema, gen_schema, core_config)
    schema = gen_schema.clean_schema(schema)

    schema_validator = create_schema_validator(
        schema,
        function,
        function.__module__,
        function.__qualname__,
        "validate_call",
        core_config,
        config_wrapper.plugin_settings,
    )
    # PluggableSchemaValidator is api compatible with SchemaValidator
    schema_validator = cast(SchemaValidator, schema_validator)

    # PluggableSchemaValidator is api compatible with SchemaValidator
    json_schema = GenerateJsonSchema().generate(schema)

    # instead of passing `description` through in core_schema, we just add it here
    if description:
        json_schema = {"description": description, **json_schema}

    json_schema = check_object_json_schema(json_schema)

    return description, json_schema, schema_validator


def _build_schema(
    fields: dict[str, core_schema.TypedDictField],
    var_kwargs_schema: core_schema.CoreSchema | None,
    gen_schema: _generate_schema.GenerateSchema,
    core_config: core_schema.CoreConfig,
) -> tuple[core_schema.CoreSchema, str | None]:
    """Generate a typed dict schema for function parameters.

    Args:
        fields: The fields to generate a typed dict schema for.
        var_kwargs_schema: The variable keyword arguments schema.
        gen_schema: The `GenerateSchema` instance.
        core_config: The core configuration.

    Returns:
        tuple of (generated core schema, single arg name).
    """
    if len(fields) == 1 and var_kwargs_schema is None:
        name = next(iter(fields))
        td_field = fields[name]
        if td_field["metadata"]["is_model_like"]:
            return td_field["schema"], name

    td_schema = core_schema.typed_dict_schema(
        fields,
        config=core_config,
        extras_schema=gen_schema.generate_schema(var_kwargs_schema) if var_kwargs_schema else None,
    )
    return td_schema, None


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
    schema_validator : SchemaValidator, optional
        The schema validator to use. If not provided, a new schema validator will be created.
    serializer : Callable[[Any], str], optional
        Callable to serialize the result of the tool. If not provided, the result will be converted to a string.
    """

    tool: Callable[..., Any] | None = None
    name: str | None = None
    description: str | None = None
    context: dict[str, Any] | None = None
    schema_dict: dict[str, Any] | None = None
    schema_validator: SchemaValidator | None = None
    serializer: Callable[[Any], str] = str

    @model_validator(mode="before")
    def setup(cls, data: dict[str, Any]) -> dict[str, Any]:
        if data.get("tool") is not None:
            description, schema, schema_validator = function_schema(data["tool"])

            if data.get("context") is not None:
                if isinstance(data["context"], str):
                    data["context"] = json.loads(data["context"])

                for k in data["context"]:
                    schema["properties"].pop(k, None)
                    schema["required"].remove(k)

            data["name"] = data.get("name") or data["tool"].__name__
        else:
            schema = None
            schema_validator = None

        data["schema_dict"] = data.get("schema_dict") or schema
        data["schema_validator"] = data.get("schema_validator") or schema_validator
        data["description"] = data.get("description") or description
        data["schema_dict"]["description"] = data["description"]

        if data["schema_dict"] is not None:
            data["schema_dict"] = JsonSchemaHandler(data["schema_dict"]).simplify()

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

    def validate_args(self, args: str | dict[str, Any]) -> dict[str, Any]:
        if self.schema_validator is None:
            raise ValueError("Schema validator is not set")

        if isinstance(args, str):
            return cast(dict[str, Any], self.schema_validator.validate_json(args))
        elif isinstance(args, dict):
            return cast(dict[str, Any], self.schema_validator.validate_python(args))

    def run(self, args: str | dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(self.tool):
            raise ValueError("Cannot run async tool with run method")

        if isinstance(args, str):
            args = cast(dict[str, Any], json.loads(args))

        assert self.tool is not None, "Tool is not set"
        all_args = ({**args, **(self.context or {})}).copy()
        valid_args = self.validate_args(all_args)

        return self.serializer(self.tool(**valid_args))

    async def run_async(self, args: str | dict[str, Any]) -> Any:
        if isinstance(args, str):
            args = cast(dict[str, Any], json.loads(args))

        all_args = {**args, **(self.context or {})}
        valid_args = self.validate_args(all_args)

        assert self.tool is not None, "Tool is not set"
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
