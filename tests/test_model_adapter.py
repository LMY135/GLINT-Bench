from unittest.mock import patch
import pytest
from glint.models import get_model, ModelResponse
from glint.models.errors import ModelConfigError
from glint.models.resolver import resolve_model

REGISTRY = {
    'providers': {'api': {'backend': 'gemini', 'api_key_env': 'GEMINI_API_KEY'},
                  'local': {'backend': 'huggingface'}},
    'models': {'a': {'provider': 'api', 'model_id': 'dummy', 'supports_native_video': True, 'supports_images': True},
               'b': {'provider': 'local', 'model_id': 'dummy-local', 'supports_native_video': True, 'supports_images': True}}
}

def test_resolve_cloud_and_local():
    assert resolve_model('a', REGISTRY).provider.backend == 'gemini'
    assert resolve_model('b', REGISTRY).provider.backend == 'huggingface'

def test_reject_missing_video_capability():
    registry = {'providers': REGISTRY['providers'], 'models': {'x': {'provider': 'api', 'model_id': 'x', 'supports_images': True}}}
    with pytest.raises(ModelConfigError, match='supports_native_video'):
        resolve_model('x', registry)

def test_unknown_model():
    with pytest.raises(ModelConfigError, match='Unknown model'):
        resolve_model('unknown', REGISTRY)

def test_factory_returns_provider_model_directly():
    class FakeModel:
        def generate(self, messages, **kwargs):
            return ModelResponse(text='ok', model='dummy')
    fake = FakeModel()
    with patch('glint.models.adapter.load_config', return_value=REGISTRY), \
         patch('glint.models.adapter.load_provider', return_value=fake) as loader:
        got = get_model('a')
        assert got is fake
        assert got.generate([]).text == 'ok'
        loader.assert_called_once()
