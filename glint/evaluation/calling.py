"""Call the public model interface and record unified responses."""
from glint.models.runtime import InferenceModel
from .types import CallFailure


def call_model(model, messages, *, stage=None):
    try:
        # Only the configured adapter receives stage-specific generation options.
        # Lightweight test doubles retain their original generate(messages) API.
        if isinstance(model, InferenceModel) and stage is not None:
            response = model.generate(messages, stage=stage)
        else:
            response = model.generate(messages)
    except Exception as exc:
        raise CallFailure(exc, getattr(exc, 'attempts', 1)) from exc
    return response, response.attempts


def response_record(response):
    usage = response.usage
    if hasattr(usage, 'model_dump'):
        usage = usage.model_dump(mode='json')
    elif not isinstance(usage, (dict, list, str, int, float, bool, type(None))):
        usage = str(usage)
    return {'text': response.text, 'model': response.model, 'usage': usage}

