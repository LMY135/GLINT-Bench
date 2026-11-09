"""GLINT-Bench two-stage evaluation."""
from .config import load_config
from .runner import evaluate

__all__ = ['load_config', 'evaluate']
