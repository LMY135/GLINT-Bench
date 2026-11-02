"""Load YAML model registry and environment variables without leaking secrets."""
import os
from pathlib import Path
from typing import Any
from .errors import ModelConfigError

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"
DEFAULT_ENV = Path(__file__).resolve().parents[2] / ".env"

def load_config(config_path=None, env_path=None) -> dict[str, Any]:
    try:
        import yaml
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ModelConfigError("Install PyYAML and python-dotenv") from exc
    if env_path is None:
        env_path = DEFAULT_ENV
    load_dotenv(dotenv_path=env_path, override=False)
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG
    if not path.is_file():
        raise ModelConfigError(f"Missing model configuration: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ModelConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("providers"), dict) or not isinstance(data.get("models"), dict):
        raise ModelConfigError("models.yaml must define 'providers' and 'models' mappings")
    return data

def env_value(var_name: str, *, required=True):
    if not isinstance(var_name, str) or not var_name:
        raise ModelConfigError("Environment variable name must be a non-empty string")
    value = os.getenv(var_name)
    if required and not value:
        raise ModelConfigError(f"Missing required environment variable: {var_name}")
    return value
