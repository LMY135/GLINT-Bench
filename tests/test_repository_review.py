from types import SimpleNamespace

from glint.models import Text, Video, Image
from glint.models.providers.huggingface import HuggingFaceModel
from glint.models.types import ModelConfig, ProviderConfig


def test_local_processor_receives_native_media_paths(tmp_path):
    video = tmp_path / 'clip.mp4'
    image = tmp_path / 'frame.jpg'
    video.touch()
    image.touch()

    class Inputs(dict):
        def to(self, device):
            return self

    class Generated:
        def __getitem__(self, key):
            return []

    class Processor:
        def apply_chat_template(self, conversation, **kwargs):
            parts = conversation[0]['content']
            assert parts[1] == {'type': 'video', 'path': str(video)}
            assert parts[2] == {'type': 'image', 'path': str(image)}
            return Inputs(input_ids=SimpleNamespace(shape=(1, 3)))

        def batch_decode(self, values, **kwargs):
            return ['ok']

    def generate(**kwargs):
        assert kwargs['max_new_tokens'] == 512
        return Generated()

    spec = ModelConfig(
        'test',
        'test',
        ProviderConfig('local', 'huggingface'),
        {'native_video_protocol': 'qwen3_vl'},
    )
    model = HuggingFaceModel(
        SimpleNamespace(device='cpu', generate=generate),
        Processor(),
        spec,
    )

    response = model.generate(
        [{
            'role': 'user',
            'content': [
                Text('question'),
                Video(str(video)),
                Image(str(image)),
            ],
        }],
        max_new_tokens=512,
    )

    assert response.text == 'ok'