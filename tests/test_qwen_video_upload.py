"""Qwen native MP4 transport stays video_url, without large Base64 payloads."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from glint.models.media import Video, Text
from glint.models.errors import ModelConfigError
from glint.models.providers.openai_video import _encode_parts, OpenAIVideoModel
from glint.models.providers.qwen_upload import MAX_INLINE_VIDEO_BYTES, QwenTemporaryUploader


def test_small_mp4_inline(tmp_path):
    path = tmp_path / 'small.mp4'
    path.write_bytes(b'abc')
    result = _encode_parts([Video(str(path))], 'qwen')
    assert result[0]['video_url']['url'].startswith('data:video/mp4;base64,')


def test_large_mp4_requires_uploader(tmp_path):
    path = tmp_path / 'large.mp4'
    with path.open('wb') as f:
        f.truncate(MAX_INLINE_VIDEO_BYTES + 1)
    with pytest.raises(ModelConfigError):
        _encode_parts([Video(str(path))], 'qwen')


def test_large_mp4_upload_and_header(tmp_path):
    path = tmp_path / 'large.mp4'
    with path.open('wb') as f:
        f.truncate(MAX_INLINE_VIDEO_BYTES + 1)

    class FakeUploader:
        def get_url(self, video):
            assert Path(video.path) == path
            return 'oss://dashscope-instant/qa/large.mp4'

    class Completion:
        def create(self, **kwargs):
            content = kwargs['messages'][0]['content'][0]
            assert content['video_url']['url'] == 'oss://dashscope-instant/qa/large.mp4'
            assert kwargs['extra_headers']['X-DashScope-OssResourceResolve'] == 'enable'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='A'))], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completion()))
    spec = SimpleNamespace(name='test', model_id='qwen-video', options={'video_protocol': 'qwen', 'qwen_stream': False})
    model = OpenAIVideoModel(client, spec, video_uploader=FakeUploader())
    assert model.generate([{'role': 'user', 'content': [Video(str(path))]}]).text == 'A'


def test_uploader_policy_and_oss_cache(tmp_path):
    path = tmp_path / 'v.mp4'
    path.write_bytes(b'movie')
    policy = {
        'upload_dir': 'dashscope-instant/test/dir', 'upload_host': 'https://oss.example.com',
        'oss_access_key_id': 'a', 'signature': 'b', 'policy': 'c',
        'x_oss_object_acl': 'private', 'x_oss_forbid_overwrite': 'true',
    }

    class Response:
        def __init__(self, value=None):
            self.value = value
        def raise_for_status(self):
            pass
        def json(self):
            return {'data': self.value}

    class Session:
        gets = 0
        posts = 0
        def get(self, url, **kwargs):
            self.gets += 1
            assert 'dashscope-intl.aliyuncs.com/api/v1/uploads' in url
            assert kwargs['params'] == {'action': 'getPolicy', 'model': 'qwen-video'}
            return Response(policy)
        def post(self, url, **kwargs):
            self.posts += 1
            assert url == policy['upload_host']
            assert kwargs['files']['key'][1].endswith('/v.mp4')
            assert kwargs['files']['file'][1].read() == b'movie'
            return Response()

    session = Session()
    up = QwenTemporaryUploader('fake-key', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
                               'qwen-video', session=session)
    first = up.get_url(Video(str(path)))
    assert first == 'oss://dashscope-instant/test/dir/v.mp4'
    assert up.get_url(Video(str(path))) == first
    assert session.gets == session.posts == 1


def test_evaluation_stage_is_not_forwarded_to_openai(tmp_path):
    """The runtime may supply stage, but Chat Completions must never receive it."""
    path = tmp_path / 'small.mp4'
    path.write_bytes(b'abc')

    class Completions:
        def create(self, *, model, messages):
            assert model == 'qwen-video'
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"A"}'))],
                usage=None
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    spec = SimpleNamespace(name='test', model_id='qwen-video',
                           options={'video_protocol': 'qwen', 'qwen_stream': False})
    model = OpenAIVideoModel(client, spec)
    response = model.generate(
        [{'role': 'user', 'content': [Video(str(path))]}], stage='video'
    )
    assert response.text == '{"answer":"A"}'


def test_qwen_stream_assembles_complete_json_and_usage(tmp_path):
    path = tmp_path / "small.mp4"
    path.write_bytes(b"abc")

    class Stream:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def __iter__(self):
            for part in ('{"answer":', '"A",', '"evidence_windows":[]}'):
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=part))],
                    usage=None
                )
            yield SimpleNamespace(choices=[], usage={"total_tokens": 5})

    class Completions:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            assert kwargs["stream_options"] == {"include_usage": True}
            assert kwargs["model"] == "qwen-video"
            assert "stage" not in kwargs
            return Stream()

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    spec = SimpleNamespace(name="test", model_id="qwen-video",
                           options={"video_protocol": "qwen"})
    model = OpenAIVideoModel(client, spec)
    response = model.generate(
        [{"role": "user", "content": [Video(str(path))]}], stage="video"
    )
    assert response.text == '{"answer":"A","evidence_windows":[]}'
    assert response.usage == {"total_tokens": 5}
