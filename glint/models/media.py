"""Transport-level multimodal types; prompts remain owned by Evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class Text:
    value: str


@dataclass(frozen=True)
class Video:
    path: str
    mime_type: str = "video/mp4"

    def __post_init__(self):
        if not Path(self.path).is_file():
            raise ValueError(f"Video file not found: {self.path}")


@dataclass(frozen=True)
class Image:
    path: str
    mime_type: str = "image/jpeg"

    def __post_init__(self):
        if not Path(self.path).is_file():
            raise ValueError(f"Image file not found: {self.path}")


MediaPart = Union[Text, Video, Image]