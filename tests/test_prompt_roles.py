"""Evaluation prompt role separation regression tests."""
from types import SimpleNamespace

from glint.models import Text, Video, Image
from glint.evaluation.prompts import video_messages, spatial_messages, qa_messages


def sample():
    return SimpleNamespace(
        question="Which option is correct?",
        metadata={"duration": 3.0, "width": 640, "height": 480},
        video_path="/tmp/example.mp4",
        evidence_windows=[{"start": 0.2, "end": 0.5}],
        evidence_boxes=[{"frame": 2, "box": [1, 2, 10, 12]}],
    )


def assert_roles(messages):
    assert [message["role"] for message in messages] == ["system", "user"]
    assert all(isinstance(part, Text) for part in messages[0]["content"])


def test_video_prompt_roles():
    messages = video_messages(sample())
    assert_roles(messages)
    assert any(isinstance(part, Video) for part in messages[1]["content"])


def test_spatial_prompt_roles():
    frames = [{"frame": 2, "time": 0.2, "path": "/tmp/frame.jpg"}]
    messages = spatial_messages(sample(), {"answer": "A"}, frames)
    assert_roles(messages)
    assert any(isinstance(part, Image) for part in messages[1]["content"])
    assert "2" in messages[1]["content"][0].value


def test_progressive_prompt_roles():
    for condition in ("baseline", "temporal_gt", "spatiotemporal_gt"):
        messages = qa_messages(sample(), condition)
        assert_roles(messages)
        assert any(isinstance(part, Video) for part in messages[1]["content"])
