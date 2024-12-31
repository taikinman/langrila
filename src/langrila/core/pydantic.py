from typing import Any

from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True

    def update(self, **kwargs: Any) -> "BaseModel":
        return self.model_copy(update=kwargs)
