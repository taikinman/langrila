from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).parent


class SampleData:
    @staticmethod
    def load_image() -> Image.Image:
        return Image.open(BASE_DIR.parent.parent / "data" / "sample.jpg")
