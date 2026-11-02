"""Discover provider implementations automatically, without importing SDKs eagerly.

Each provider module exposes BACKEND (string) and load(spec) (callable).
Discovery imports trusted local Python modules only; never scan arbitrary user paths.
"""
import importlib
import pkgutil
from functools import lru_cache
from .errors import ModelConfigError


@lru_cache(maxsize=1)
def discover_providers():
    from . import providers
    loaders = {}
    for info in pkgutil.iter_modules(providers.__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{providers.__name__}.{info.name}")
        backend = getattr(module, "BACKEND", None)
        loader = getattr(module, "load", None)
        if backend is None and loader is None:
            continue  # Utility/legacy module; not a provider plugin.
        if not isinstance(backend, str) or not backend.strip() or not callable(loader):
            raise ModelConfigError(f"Invalid provider plugin: {module.__name__}")
        if backend in loaders:
            raise ModelConfigError(f"Duplicate provider backend: {backend}")
        loaders[backend] = loader
    return loaders


def load_provider(spec):
    backend = spec.provider.backend
    loader = discover_providers().get(backend)
    if loader is None:
        raise ModelConfigError(
            f"No provider registered for backend {backend!r}; "
            f"available: {', '.join(sorted(discover_providers()))}"
        )
    return loader(spec)
