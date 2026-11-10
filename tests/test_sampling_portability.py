from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess

import pytest
from glint.evaluation import evaluate
from glint.evaluation.dataset import load_samples
from glint.evaluation.sampling import run_media, sample_frames
from test_evaluation import config, make_video, PerfectModel


def test_media_error_reports_stderr_without_locale_decoding(monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(
        returncode=1, stdout=b'', stderr=b'encoder failure \xff'))
    with pytest.raises(ValueError, match='ffmpeg failed .*encoder failure'):
        run_media(['ffmpeg'], timeout=1)


def test_media_timeout_reports_stderr(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(['ffmpeg'], 1, stderr=b'decoder stalled')
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(ValueError, match='timed out.*decoder stalled'):
        run_media(['ffmpeg'], timeout=1)


def test_actual_image_pipe_and_unicode_space_paths(config, tmp_path):
    make_video(config)
    sample = load_samples(config)[0]
    folder = tmp_path / '含 空格'
    folder.mkdir()
    video = folder / '视频.mp4'
    shutil.copyfile(sample.video_path, video)
    sample = replace(sample, video_path=video)
    frames = sample_frames(sample, (0.2, 0.6), 3, folder / '图片')
    assert [entry['frame'] for entry in frames] == [2, 4, 6]
    for entry in frames:
        assert Path(entry['path']).read_bytes().startswith(b'\xff\xd8')


def test_sampling_failure_prints_diagnostic(config, monkeypatch, capsys):
    make_video(config)
    import glint.evaluation.runner as runner
    def fail(*args, **kwargs):
        raise ValueError('ffmpeg failed (exit 1): test decoder detail')
    monkeypatch.setattr(runner, 'sample_frames', fail)
    result = evaluate(config, model=PerfectModel(load_samples(config)[0]))
    assert result[0]['status'] == 'sampling_error'
    assert 'test decoder detail' in capsys.readouterr().out
