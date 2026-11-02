from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from glint.models.registry import discover_providers, load_provider
from glint.models.resolver import resolve_model
from glint.models.errors import ModelConfigError
from glint.models.providers.openai_video import _encode_parts, OpenAIVideoModel
from glint.models.providers.gemini import GeminiModel
from glint.models.media import Video, Text, Image

REGISTRY = {'providers': {'x': {'backend': 'bedrock'}}, 'models': {'m': {
    'provider': 'x', 'model_id': 'example', 'supports_native_video': True,
    'supports_images': True}}}

def test_resolver_accepts_new_backend():
    assert resolve_model('m', REGISTRY).provider.backend == 'bedrock'

def test_discovery_has_three_providers():
    assert {'gemini', 'openai_compatible', 'huggingface'} <= discover_providers().keys()

def test_unknown_backend_fails_at_registry_not_resolver():
    with pytest.raises(ModelConfigError, match='No provider registered'):
        load_provider(resolve_model('m', REGISTRY))

def test_moonshot_native_video_data_uri(tmp_path):
    file = tmp_path / 'test.mp4'; file.write_bytes(b'MP4')
    parts = _encode_parts([Text('hi'), Video(path=str(file))], 'moonshot')
    assert parts[1]['type'] == 'video_url'
    assert parts[1]['video_url'].startswith('data:video/mp4;base64,')

def test_qwen_video_protocol(tmp_path):
    file = tmp_path / 'test.mp4'; file.write_bytes(b'MP4')
    parts = _encode_parts([Video(path=str(file))], 'qwen')
    assert parts[0]['video_url']['url'].startswith('data:video/mp4;base64,')

def test_unknown_protocol_rejected(tmp_path):
    file = tmp_path / 'test.mp4'; file.write_bytes(b'MP4')
    with pytest.raises(ModelConfigError, match='video_protocol'):
        _encode_parts([Video(path=str(file))], None)

def test_image_local_path(tmp_path):
    file = tmp_path / 'test.jpg'; file.write_bytes(b'JPEG')
    assert _encode_parts([Image(path=str(file))], 'qwen')[0]['image_url']['url'].startswith('data:image/jpeg;base64,')

def test_openai_model_generate_calls_sdk(tmp_path):
    file = tmp_path / 'test.mp4'; file.write_bytes(b'MP4')
    raw = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='done'))], usage={'tokens': 10})
    from unittest.mock import MagicMock
    client = MagicMock(); client.chat.completions.create.return_value = raw
    spec = SimpleNamespace(name='qwen', model_id='model-id', options={'video_protocol': 'qwen'})
    model = OpenAIVideoModel(client, spec)
    result = model.generate([{'role': 'user', 'content': [Video(path=str(file))]}])
    assert result.text == 'done'
    assert client.chat.completions.create.call_args.kwargs['model'] == 'model-id'

def test_gemini_model_loader_without_real_api(tmp_path):
    import sys
    from unittest.mock import MagicMock
    file = tmp_path / 'test.mp4'; file.write_bytes(b'MP4')
    class FakePart:
        @staticmethod
        def from_text(text): return ('text', text)
        @staticmethod
        def from_uri(file_uri, mime_type): return ('uri', file_uri, mime_type)
    class FakeContent:
        def __init__(self, role, parts): self.role, self.parts = role, parts
    fake_types = SimpleNamespace(Part=FakePart, Content=FakeContent)
    client = MagicMock()
    client.files.upload.return_value = SimpleNamespace(uri='files/test', mime_type='video/mp4', state=SimpleNamespace(name='ACTIVE'))
    client.models.generate_content.return_value = SimpleNamespace(text='answer', usage_metadata=None)
    spec = SimpleNamespace(name='gemini', model_id='gemini-test', provider=SimpleNamespace(options={}))
    with patch.dict(sys.modules, {'google': SimpleNamespace(genai=SimpleNamespace(types=fake_types)), 'google.genai': SimpleNamespace(types=fake_types), 'google.genai.types': fake_types}):
        model = GeminiModel(client, spec)
        result = model.generate([{'role': 'user', 'content': [Video(path=str(file)), Text('Describe')]}])
    assert result.text == 'answer'
    client.files.upload.assert_called_once_with(file=str(file))
