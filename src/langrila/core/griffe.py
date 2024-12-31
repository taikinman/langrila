# This script is heavily based on the v0.0.12 of : https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/_griffe.py
# The original script is licensed under the MIT License:

from __future__ import annotations as _annotations

import re
from inspect import signature
from typing import Any, Callable, Literal, cast

from griffe import Docstring, DocstringSectionKind
from griffe import Object as GriffeObject

DocstringStyle = Literal["google", "numpy", "sphinx"]


def get_docstring_and_params_descriptions(
    func: Callable[..., Any], *, style: DocstringStyle | None = None
) -> tuple[str, dict[str, str]]:
    """Extract the function description and parameter descriptions from a function's docstring.

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
