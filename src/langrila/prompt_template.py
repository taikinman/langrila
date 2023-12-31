import re
from typing import Any

from pydantic import BaseModel, model_validator


class PromptTemplate(BaseModel):
    args: dict[str, Any] = {}
    template: str = ""

    def format(self):
        self._check_args(self.template, self.args)
        return self.template.format(**self.args)

    def set_args(self, **args: dict[str, Any]):
        if self.template:
            self._check_args(self.template, args)
        self.args = args
        return self

    def set_template(self, template: str):
        if self.args:
            self._check_args(template, self.args)
        self.template = template
        return self

    @model_validator(mode="after")
    def check_fields(self):
        if self.args and self.template:
            self._check_args(self.template, self.args)
        return self

    @staticmethod
    def _check_args(template, args):
        pattern_args = re.compile(r"{[a-zA-Z0-9_]+}")
        pattern_sub = re.compile("{|}")
        found_args = {pattern_sub.sub("", a) for a in pattern_args.findall(template)}
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

            raise ValueError(
                f"Template has args {_found_args} but {_input_args} were input"
            )
