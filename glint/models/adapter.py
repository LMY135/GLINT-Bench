"""Public factory: configuration -> resolution -> provider -> inference runtime."""
import hashlib
import json
from .config import load_config, env_value
from .resolver import resolve_model
from .registry import load_provider
from .inference import Model
from .errors import ModelConfigError
from .runtime import InferenceModel, validate_runtime


def get_model(name: str, *, config_path=None, env_path=None) -> Model:
    config = load_config(config_path=config_path, env_path=env_path)
    spec = resolve_model(name, config)
    runtime = config.get('inference', {})
    retry = validate_runtime(runtime)
    generation = spec.options.get('generation', {})
    if not isinstance(generation, dict):
        raise ModelConfigError('Model generation options must be a mapping')
    model = load_provider(spec)
    if not runtime and not generation:
        return model
    return InferenceModel(model, retry, generation)


def model_fingerprint(name: str, *, config_path=None, env_path=None) -> str:
    """Identify a resolved model without loading weights or making requests."""
    config = load_config(config_path=config_path, env_path=env_path)
    spec = resolve_model(name, config)
    runtime = config.get('inference', {})
    validate_runtime(runtime)
    provider_options = dict(spec.provider.options)
    endpoint_var = provider_options.get('base_url_env')
    if endpoint_var:
        provider_options['resolved_base_url'] = env_value(endpoint_var, required=False)
    identity = {'name': spec.name, 'model_id': spec.model_id,
                'backend': spec.provider.backend, 'provider': provider_options,
                'options': dict(spec.options), 'inference': runtime}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
