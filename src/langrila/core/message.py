from typing import Generic, TypeVar

from .prompt import Prompt
from .pydantic import BaseModel
from .response import Response

MessageType = TypeVar("MessageType", Prompt, Response)


class Message(BaseModel, Generic[MessageType]):
    message: MessageType
