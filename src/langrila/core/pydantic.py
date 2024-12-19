from typing import Any

from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True

    def update(self, **kwargs: Any) -> "BaseModel":
        attribs = self.__dict__.copy()
        attribs.update(kwargs)
        return self.__class__(**attribs)
