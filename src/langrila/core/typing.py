from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
from PIL import Image

PathType = Path | str
ImageType = Image.Image | PathType
RoleType = Literal["system", "user", "assistant", "function", "tool_call", "tool"]
ArrayType = np.ndarray[tuple[int, ...], np.dtype[np.float32 | np.float64]]

ClientMessage = TypeVar("ClientMessage")
ClientMessageContent = TypeVar("ClientMessageContent")
ClientTool = TypeVar("ClientTool")
ClientSystemMessage = TypeVar("ClientSystemMessage")
