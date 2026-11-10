import copy
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest
from glint.models import ModelResponse, Text, Video, Image
from glint.models.runtime import InferenceModel, is_retryable
from glint.models.errors import ModelInferenceError
from glint.evaluation.config import load_config
from glint.evaluation.dataset import load_samples
from glint.evaluation.metrics import tnwd, temporal_gate, bbox_iou
from glint.evaluation.parsing import parse_video_response, parse_spatial_response
from glint.evaluation.prompts import spatial_messages
from glint.evaluation.runner import evaluate
from glint.evaluation.types import CallFailure, ParseError


@pytest.fixture
def config(tmp_path):
    root = tmp_path
    (root / 'videos/magic').mkdir(parents=True)
    (root / 'videos/magic/magic_0001.mp4').touch()
    (root / 'configs').mkdir()
    registry = {'providers': {'test': {'backend': 'gemini'}},
                'models': {'test': {'provider': 'test', 'model_id': 'fake',
                                    'supports_native_video': True, 'supports_images': True}}}
    import yaml
    (root / 'configs/models.yaml').write_text(yaml.safe_dump(registry))
    meta = {'video_id': 'magic_0001', 'width': 64, 'height': 48, 'fps': 10,
            'duration': 1.0, 'num_frames': 10}
    sample = {'question_id': 'magic_0001_q01', 'video_id': 'magic_0001', 'category': 'magic',
              'question_type': 'mcq', 'question': 'Choose A or B.', 'answer': 'A',
              'evidence_windows': [{'start': 0.2, 'end': 0.6}],
              'evidence_boxes': [{'frame': f, 'time': f / 10, 'box': [1, 2, 10, 12]} for f in range(2, 7)]}
    (root / 'qa.jsonl').write_text(json.dumps(sample) + '\n')
    (root / 'meta.jsonl').write_text(json.dumps(meta) + '\n')
    settings = {'task': 'grounding', 'project_root': '..', 'model': 'test', 'qa_path': 'qa.jsonl',
                'metadata_path': 'meta.jsonl', 'output_dir': 'output', 'k': 3}
    path = root / 'configs/evaluation.yaml'
    path.write_text(yaml.safe_dump(settings))
    return load_config(path)


def test_local_video_layout_and_cwd(config, tmp_path, monkeypatch):
    other_cwd = tmp_path / 'unrelated_cwd'
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    sample = load_samples(config)[0]
    assert sample.video_path == Path(config['videos_dir']) / 'magic/magic_0001.mp4'
    sample.video_path.unlink()
    (Path(config['videos_dir']) / 'magic/magic_0001_raw.mp4').touch()
    with pytest.raises(FileNotFoundError, match=r'magic_0001\.mp4'):
        load_samples(config)


def test_external_video_root(config, tmp_path):
    external = tmp_path / 'external'
    (external / 'magic').mkdir(parents=True)
    (external / 'magic/magic_0001.mp4').touch()
    config['videos_dir'] = str(external)
    assert load_samples(config)[0].video_path.parent == external / 'magic'


def test_duplicate_ids_and_multiwindow_rejected(config):
    path = Path(config['qa_path'])
    original = path.read_text()
    path.write_text(original + original)
    with pytest.raises(ValueError, match='Duplicate question_id'):
        load_samples(config)
    item = json.loads(original)
    item['evidence_windows'] *= 2
    path.write_text(json.dumps(item))
    with pytest.raises(ValueError, match='exactly one'):
        load_samples(config)


def test_metric_formula_and_gate():
    gt = {'start': 1.0, 'end': 1.2}
    pred = {'start': 1.5, 'end': 1.7}
    c = -0.5 / math.log(0.5)
    assert tnwd(gt, gt, c) == 1
    assert tnwd(pred, gt, c) == pytest.approx(0.5)
    assert temporal_gate([pred], [gt], c, 0.1) == (False, None)
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert bbox_iou([0, 0, 10, 10], [20, 0, 10, 10]) == 0
    fractional = [582.85, 1144.0, 254.05, 224.7]
    assert 0 <= bbox_iou(fractional, fractional) <= 1
    long = {'start': 0, 'end': 3}
    assert not temporal_gate([long], [gt], c, 0.99)[0]


def test_independent_parsing():
    p = parse_video_response('{"answer":"a","evidence_windows":[{"start":2,"end":1}]}', 3)
    assert p.answer == 'A' and 'temporal' in p.errors
    p = parse_video_response('{"answer":"invalid","evidence_windows":[{"start":1,"end":2}]}', 3)
    assert p.evidence_windows and p.errors == {'answer': 'Expected answer A/B/C/D'}
    assert parse_video_response('bad JSON', 3).errors
    assert parse_video_response('{"answer":"A","evidence_windows":[{"start":0,"end":NaN}]}', 3).errors
    frames = [{'frame': 1}, {'frame': 2}]
    assert parse_spatial_response('{"evidence_boxes":[]}', frames, 64, 48) == []
    with pytest.raises(ParseError):
        parse_spatial_response('{"evidence_boxes":[{"frame":3,"box":[1,2,3,4]}]}', frames, 64, 48)


def test_retry_budget_and_nonretryable():
    class Failing:
        count = 0
        def generate(self, messages, **kwargs):
            self.count += 1
            raise TimeoutError('timeout')
    model = Failing()
    retry = {'max_retries': 2, 'initial_delay': 0, 'max_delay': 0}
    with pytest.raises(ModelInferenceError) as exc:
        InferenceModel(model, retry, {}).generate([])
    assert model.count == 3 and exc.value.attempts == 3
    class APIError(Exception):
        status_code = 401
    assert not is_retryable(APIError())
    APIError.status_code = 429
    assert is_retryable(APIError())
    assert not is_retryable(ValueError('bad codec'))


class PerfectModel:
    def __init__(self, sample, *, interrupt=False, missing=False):
        self.sample = sample
        self.video_calls = 0
        self.spatial_calls = 0
        self.interrupt = interrupt
        self.missing = missing

    def generate(self, messages, **kwargs):
        content = messages[0]['content']
        if any(isinstance(part, Video) for part in content):
            self.video_calls += 1
            data = {'answer': self.sample.answer, 'evidence_windows': self.sample.evidence_windows}
        else:
            self.spatial_calls += 1
            if self.interrupt:
                self.interrupt = False
                raise KeyboardInterrupt()
            frames = [int(part.value.split(',')[0].split()[1]) for part in content
                      if isinstance(part, Text) and part.value.startswith('Frame ')]
            boxes = {b['frame']: b['box'] for b in self.sample.evidence_boxes}
            if self.missing:
                frames = frames[:1]
            data = {'evidence_boxes': [{'frame': f, 'box': boxes[f]} for f in frames]}
        return ModelResponse(json.dumps(data), 'fake')


def make_video(config):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe unavailable')
    path = load_samples(config)[0].video_path
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    'testsrc=size=64x48:rate=10', '-t', '1', '-pix_fmt', 'yuv420p', str(path)],
                   check=True, capture_output=True)


def test_real_decoding_resume_and_rescore(config):
    make_video(config)
    model = PerfectModel(load_samples(config)[0], interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        evaluate(config, model=model)
    scores = evaluate(config, model=model)
    assert model.video_calls == 1
    assert scores[0]['accqa'] == scores[0]['tnwd'] == scores[0]['tg_siou'] == 1
    assert scores[0]['sampled_frames'] == 3
    evaluate(config, model=model)
    assert model.spatial_calls == 2
    changed = copy.deepcopy(config)
    changed['c'] = 1
    evaluate(changed, score_only=True)
    assert json.loads((Path(config['output_dir']) / '.run_state' / 'score_config.json').read_text())['c'] == 1
    assert not (Path(config['output_dir']) / '.run.lock').exists()
    changed['k'] = 7
    with pytest.raises(ValueError, match='changed'):
        evaluate(changed, model=model)


def test_missing_boxes_score_zero_and_short_sampling(config):
    make_video(config)
    config['k'] = 10
    scores = evaluate(config, model=PerfectModel(load_samples(config)[0], missing=True))
    assert scores[0]['sampled_frames'] == 5
    assert scores[0]['tg_siou'] == pytest.approx(0.2)


def test_failed_calls_in_denominator(config):
    make_video(config)
    class Broken:
        def generate(self, messages, **kwargs):
            raise TimeoutError('connection timeout')
    scores = evaluate(config, model=Broken())
    assert scores[0]['status'] == 'call_error'
    summary = json.loads((Path(config['output_dir']) / 'magic' / 'grounding' / 'summary.json').read_text())
    assert summary['overall']['count'] == 1 and summary['overall']['accqa'] == 0
    model = PerfectModel(load_samples(config)[0])
    assert evaluate(config, model=model, retry_failed=True)[0]['tg_siou'] == 1


def test_gt_not_in_prompt(config, tmp_path):
    sample = load_samples(config)[0]
    frame = tmp_path / 'image.jpg'
    frame.touch()
    messages = spatial_messages(sample, {'answer': 'B', 'evidence_windows': []},
                                [{'frame': 2, 'time': 0.2, 'path': str(frame)}])
    text = '\n'.join(part.value for part in messages[0]['content'] if isinstance(part, Text))
    assert '"box"' in text and '1, 2, 10, 12' not in text
    assert any(isinstance(part, Image) for part in messages[0]['content'])


def test_cloud_timeout_and_sdk_retry_options(monkeypatch):
    import sys
    from types import SimpleNamespace
    from glint.models.types import ProviderConfig, ModelConfig
    from glint.models.providers import openai_video, gemini
    monkeypatch.setenv('TEST_API_KEY', 'test-key')
    calls = []
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=lambda **kwargs: calls.append(kwargs) or object()))
    options = {'api_key_env': 'TEST_API_KEY', 'timeout': 12, 'max_retries': 0}
    spec = ModelConfig('test', 'test-model', ProviderConfig('api', 'openai_compatible', options),
                       {'video_protocol': 'qwen'})
    openai_video.load(spec)
    assert calls[-1]['timeout'] == 12 and calls[-1]['max_retries'] == 0
    genai = pytest.importorskip('google.genai')
    monkeypatch.setattr(genai, 'Client', lambda **kwargs: calls.append(kwargs) or object())
    gemini.load(spec)
    assert calls[-1]['http_options'] == {'timeout': 12000, 'retry_options': {'attempts': 1}}


def test_gemini_processing_timeout_is_retryable(tmp_path):
    pytest.importorskip('google.genai')
    from types import SimpleNamespace
    from glint.models.providers.gemini import GeminiModel
    from glint.models.types import ProviderConfig, ModelConfig
    path = tmp_path / 'video.mp4'
    path.touch()
    client = SimpleNamespace(files=SimpleNamespace(upload=lambda **kwargs: SimpleNamespace(
        state=SimpleNamespace(name='PROCESSING'))))
    spec = ModelConfig('test', 'test', ProviderConfig('api', 'gemini', {'file_timeout_seconds': 0}))
    with pytest.raises(TimeoutError) as caught:
        GeminiModel(client, spec)._upload(Video(str(path)))
    assert is_retryable(caught.value)
