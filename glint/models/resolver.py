"""Validate model configuration and resolve its provider; no SDK knowledge."""
from .types import ModelConfig, ProviderConfig
from .errors import ModelConfigError


def resolve_model(name: str, registry: dict) -> ModelConfig:

    if not isinstance(registry, dict):
        raise ModelConfigError("Model registry must be a mapping")
    models, providers = registry.get("models"), registry.get("providers")


    if not isinstance(models, dict) or not isinstance(providers, dict):
        raise ModelConfigError("'models' and 'providers' must be mappings")


    if name not in models:
        raise ModelConfigError(f"Unknown model {name!r}; available: {', '.join(models)}")
    
    spec = models[name]


    if not isinstance(spec, dict):
        raise ModelConfigError(f"Model {name!r} must be a mapping")
    
    provider_name = spec.get("provider")


    if not isinstance(provider_name, str) or provider_name not in providers:
        raise ModelConfigError(f"Model {name!r} references unknown provider {provider_name!r}")

    provider_spec = providers[provider_name]


    if not isinstance(provider_spec, dict):
        raise ModelConfigError(f"Provider {provider_name!r} must be a mapping")

    backend = provider_spec.get("backend")


    if not isinstance(backend, str) or not backend.strip():
        raise ModelConfigError(f"Provider {provider_name!r} requires a non-empty backend")

    model_id = spec.get("model_id")


    if not isinstance(model_id, str) or not model_id.strip():
        raise ModelConfigError(f"Model {name!r} requires a non-empty model_id")


    if spec.get("supports_native_video") is not True or spec.get("supports_images") is not True:
        raise ModelConfigError(f"Model {name!r} must declare supports_native_video=true and supports_images=true")
    return ModelConfig(
        name=name, model_id=model_id,
        provider=ProviderConfig(name=provider_name, backend=backend,
            options={k: v for k, v in provider_spec.items() if k != "backend"}),
        options={k: v for k, v in spec.items() if k not in {"provider", "model_id"}},
    )
