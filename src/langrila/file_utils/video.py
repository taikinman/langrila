from pathlib import Path
from typing import Generator

import imageio
from PIL import Image

from ..core.typing import PathType


def sample_frames(video_path: PathType, fps: float = 1.0) -> Generator[Image.Image, None, None]:
    reader = imageio.get_reader(video_path)
    meta_data = reader.get_meta_data()
    original_fps = meta_data.get("fps")

    assert original_fps, "Could not extract fps from video"

    frame_interval = round(original_fps / fps)

    for i, frame in enumerate(reader):
        if i % frame_interval == 0:
            # imageio.imwrite(f"{output_folder}/frame_{frame_count:04d}.jpg", frame)
            # frame_count += 1

            yield Image.fromarray(frame)
