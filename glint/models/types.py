"""Configuration objects produced by the resolver."""
from dataclasses import dataclass, field
from typing import Any, Mapping

@dataclass(frozen=True)
class ProviderConfig:
    name: str
    backend: str
    options: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_id: str
    provider: ProviderConfig
    options: Mapping[str, Any] = field(default_factory=dict)
