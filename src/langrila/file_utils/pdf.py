from pathlib import Path
from typing import Generator

import pypdfium2
from PIL import Image

from ..core.typing import PathType


def read_pdf_asimage(path: PathType, scale: float = 2.5) -> Generator[Image.Image, None, None]:
    pdf = pypdfium2.PdfDocument(path)

    for page in pdf:
        pil_image = page.render(
            scale=scale,
            rotation=0,
            no_smoothtext=False,
            no_smoothpath=False,
            no_smoothimage=False,
            prefer_bgrx=False,
        ).to_pil()

        yield pil_image
