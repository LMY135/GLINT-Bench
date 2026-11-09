"""Shared evaluation records."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Sample:
    question_id: str
    video_id: str
    category: str
    question: str
    answer: str
    evidence_windows: list[dict]
    evidence_boxes: list[dict]
    metadata: dict
    video_path: Path


@dataclass
class VideoPrediction:
    answer: str | None = None
    evidence_windows: list[dict] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


class ParseError(ValueError):
    """A response cannot be interpreted as the requested prediction."""


class CallFailure(RuntimeError):
    def __init__(self, cause: Exception, attempts: int):
        super().__init__(str(cause))
        self.cause = cause
        self.attempts = attempts
