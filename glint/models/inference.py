"""Shared inference contract; SDK-specific implementations live in providers/."""
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict
from .media import MediaPart

class Message(TypedDict):
    role: str
    content: list[MediaPart]

@dataclass
class ModelResponse:
    text: str
    model: str
    usage: Any = None
    raw_response: Any = None
    attempts: int = 1

class Model(Protocol):
    def generate(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        """Execute native multimodal inference."""
        raise NotImplementedError
