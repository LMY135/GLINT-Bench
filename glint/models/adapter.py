"""Public factory: configuration -> resolution -> provider -> inference-ready model."""
from .config import load_config
from .resolver import resolve_model
from .registry import load_provider
from .inference import Model


def get_model(name: str, *, config_path=None, env_path=None) -> Model:
    config = load_config(config_path=config_path, env_path=env_path)
    spec = resolve_model(name, config)
    return load_provider(spec)
