from os import PathLike
from pathlib import Path
from typing import Literal

from PIL import Image

PathType = Path | PathLike | str
ImageType = Image.Image | PathType
RoleType = Literal["system", "user", "assistant", "function", "function_call", "tool"]
