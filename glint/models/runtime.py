"""Provider-independent inference runtime owned by the model Adapter."""
from dataclasses import replace
import math
import random
import time
from .errors import ModelConfigError, ModelInferenceError


RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


def is_retryable(exc):
    current, seen = exc, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, 'status_code', None)
        if status is None:
            status = getattr(getattr(current, 'response', None), 'status_code', None)
        if status is None:
            status = getattr(current, 'code', None)
        if isinstance(status, int):
            return status in RETRY_STATUSES
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True
        if type(current).__name__ in {'APITimeoutError', 'APIConnectionError', 'ConnectTimeout',
                                      'ReadTimeout', 'WriteTimeout', 'PoolTimeout', 'ConnectError',
                                      'ReadError', 'RemoteProtocolError'}:
            return True
        current = current.__cause__
    return False



def validate_runtime(settings):
    if not isinstance(settings, dict):
        raise ModelConfigError('inference must be a mapping')
    retry = {'max_retries': 0, 'initial_delay': 1.0, 'max_delay': 30.0}
    supplied = settings.get('retry', {})
    if not isinstance(supplied, dict):
        raise ModelConfigError('inference.retry must be a mapping')
    retry.update(supplied)
    if type(retry['max_retries']) is not int or retry['max_retries'] < 0:
        raise ModelConfigError('max_retries must be a nonnegative integer')
    for key in ('initial_delay', 'max_delay'):
        value = retry[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ModelConfigError(f'{key} must be finite and nonnegative')
    return retry


class InferenceModel:
    def __init__(self, model, retry, generation):
        self._model = model
        self._retry = retry
        self._generation = generation

    def generate(self, messages, **kwargs):
        options = {**self._generation, **kwargs}
        for attempt in range(self._retry['max_retries'] + 1):
            try:
                response = self._model.generate(messages, **options)
                return replace(response, attempts=attempt + 1)
            except Exception as exc:
                if attempt >= self._retry['max_retries'] or not is_retryable(exc):
                    raise ModelInferenceError(exc, attempt + 1) from exc
                delay = min(self._retry['max_delay'], self._retry['initial_delay'] * 2 ** attempt)
                time.sleep(random.uniform(delay / 2, delay))
