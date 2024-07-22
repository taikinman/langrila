from os import PathLike
from pathlib import Path

from PIL import Image

PathType = Path | PathLike | str
ImageType = Image.Image | PathType
FileType = str | bytes | PathType
