import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, model_validator


class PromptTemplate(BaseModel):
    args: dict[str, Any] = {}
    template: str = ""

    @staticmethod
    def from_text_file(file_path: str | Path) -> "PromptTemplate":
        file_path = Path(file_path)

        with open(file_path, "r") as f:
            template = f.read()
        return PromptTemplate(template=template)

    def format(self) -> str:
        self._check_args(self.template, self.args)
        return self.template.format(**self.args)

    def set_args(self, **args: dict[str, Any]) -> "PromptTemplate":
        if self.template:
            self._check_args(self.template, args)
        self.args = args
        return self

    def set_template(self, template: str) -> "PromptTemplate":
        if self.args:
            self._check_args(template, self.args)
        self.template = template
        return self

    @model_validator(mode="after")
    def check_fields(self) -> "PromptTemplate":
        if self.args and self.template:
            self._check_args(self.template, self.args)
        return self

    @staticmethod
    def _check_args(
        template: str,
        args: dict[str, Any],
    ) -> None:
        def odd_repeat_pattern(s):
            return f"(?<!{s}){s}(?:{s}{s})*(?!{s})"

        pattern = re.compile(odd_repeat_pattern("{") + "[a-zA-Z0-9_]+" + odd_repeat_pattern("}"))
        pattern_sub = re.compile("{|}")
        found_args = set([pattern_sub.sub("", m) for m in pattern.findall(template)])
        input_args = set(args.keys())
        if found_args != input_args:
            if found_args:
                _found_args = "{" + ", ".join(found_args) + "}"
            else:
                _found_args = "{}"

            if input_args:
                _input_args = "{" + ", ".join(input_args) + "}"
            else:
                _input_args = "{}"

            raise ValueError(f"Template has args {_found_args} but {_input_args} were input")
