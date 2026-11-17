"""MiMo V2.6 native video and image transport tests (no paid API calls)."""
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from glint.models.media import Text, Video, Image
from glint.models.providers.openai_video import _encode_parts, OpenAIVideoModel
from glint.models.providers.mimo_video import mimo_video_url, MAX_INLINE_MIMO_VIDEO_BYTES
from glint.models.errors import ModelConfigError


def _video(tmp_path, name="video.mp4", size=10):
    path = tmp_path / name
    with path.open("wb") as f:
        f.truncate(size)
    return Video(str(path))


def test_mimo_video_matches_official_shape(tmp_path):
    result = _encode_parts([Text("Question"), _video(tmp_path)], "mimo")
    assert result[0] == {"type": "text", "text": "Question"}
    video = result[1]
    assert video["type"] == "video_url"
    assert video["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert video["fps"] == 2
    assert video["media_resolution"] == "default"
    assert "fps" not in video["video_url"]


def test_mimo_spatial_image_input(tmp_path):
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"jpeg")
    result = _encode_parts([Text("Locate the target"), Image(str(path))], "mimo")
    assert result[1]["type"] == "image_url"
    assert result[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_large_video_requires_explicit_url(tmp_path, monkeypatch):
    monkeypatch.delenv("MIMO_VIDEO_URL_MAP_PATH", raising=False)
    with pytest.raises(ModelConfigError, match="50 MB"):
        mimo_video_url(_video(tmp_path, size=MAX_INLINE_MIMO_VIDEO_BYTES + 1))


def test_large_video_uses_https_url_map(tmp_path, monkeypatch):
    video = _video(tmp_path, size=MAX_INLINE_MIMO_VIDEO_BYTES + 1)
    mapping = tmp_path / "urls.json"
    mapping.write_text(json.dumps({"video.mp4": "https://example.org/video.mp4"}))
    monkeypatch.setenv("MIMO_VIDEO_URL_MAP_PATH", str(mapping))
    encoded = _encode_parts([video], "mimo")
    assert encoded[0]["video_url"]["url"] == "https://example.org/video.mp4"
    assert encoded[0]["fps"] == 2


def test_mimo_drops_stage_and_keeps_structured_response(tmp_path):
    video = _video(tmp_path)
    class Completions:
        def create(self, *, model, messages):
            assert model == "mimo-v2.6-pro"
            assert messages[0]["content"][0]["type"] == "video_url"
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content='{"answer":"A","evidence_windows":[{"start":0.1,"end":0.5}]}'))],
                usage={"prompt_tokens": 10, "completion_tokens": 20}
            )
    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    spec = SimpleNamespace(name="mimo-v2.6-pro", model_id="mimo-v2.6-pro",
                           options={"video_protocol": "mimo", "mimo_fps": 2})
    response = OpenAIVideoModel(client, spec).generate(
        [{"role": "user", "content": [video]}], stage="video")
    assert '"evidence_windows"' in response.text
    assert response.usage["prompt_tokens"] == 10
