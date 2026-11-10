import copy
import json
import sys
from pathlib import Path

import pytest
import yaml
from glint.evaluation import evaluate
from glint.evaluation.__main__ import main
from glint.evaluation.config import load_config, normalize_categories
from glint.evaluation.dataset import load_samples
from test_evaluation import config
from test_progressive_qa import AnswerModel


def add_game(config, *, video_exists=False):
    qa = Path(config['qa_path'])
    original = json.loads(qa.read_text())
    game = {**original, 'question_id': 'game_0001_q01', 'video_id': 'game_0001', 'category': 'game'}
    qa.write_text(json.dumps(original) + '\n' + json.dumps(game) + '\n')
    metadata = Path(config['metadata_path'])
    original_meta = json.loads(metadata.read_text())
    metadata.write_text(json.dumps(original_meta) + '\n' + json.dumps({**original_meta, 'video_id': 'game_0001'}) + '\n')
    if video_exists:
        video = Path(config['videos_dir']) / 'game/game_0001.mp4'
        video.parent.mkdir()
        video.touch()


def test_filter_before_video_and_metadata_validation(config):
    add_game(config)
    config['categories'] = ['magic']
    assert [s.question_id for s in load_samples(config)] == ['magic_0001_q01']
    metadata = Path(config['metadata_path'])
    rows = [json.loads(line) for line in metadata.read_text().splitlines()]
    rows[1]['duration'] = -1
    metadata.write_text('\n'.join(json.dumps(row) for row in rows))
    assert len(load_samples(config)) == 1
    config['categories'] = ['game']
    with pytest.raises(ValueError, match='Invalid metadata'):
        load_samples(config)


def test_multiple_categories_and_missing_selected_video(config):
    add_game(config)
    config['categories'] = ['magic', 'game']
    with pytest.raises(FileNotFoundError, match=r'game_0001\.mp4'):
        load_samples(config)
    path = Path(config['videos_dir']) / 'game/game_0001.mp4'
    path.parent.mkdir()
    path.touch()
    assert len(load_samples(config)) == 2
    config['categories'] = []
    assert len(load_samples(config)) == 2


def test_unknown_empty_and_yaml_selector(config):
    assert normalize_categories(['magic', 'game', 'magic']) == ['game', 'magic']
    with pytest.raises(ValueError, match='Unknown categories'):
        normalize_categories(['magci'])
    with pytest.raises(ValueError, match='must be a list'):
        normalize_categories('magic')
    config['categories'] = ['traffic']
    with pytest.raises(ValueError, match='No QA samples matched'):
        load_samples(config)
    path = Path(config['project_root']) / 'configs/evaluation.yaml'
    data = yaml.safe_load(path.read_text())
    data['categories'] = ['magic']
    path.write_text(yaml.safe_dump(data))
    assert load_config(path)['categories'] == ['magic']


def test_cli_category_override(config, monkeypatch, capsys):
    add_game(config)
    path = Path(config['project_root']) / 'configs/evaluation.yaml'
    data = yaml.safe_load(path.read_text())
    data['categories'] = ['game']
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(sys, 'argv', ['evaluation', '--config', str(path), '--categories', 'magic', '--validate-only'])
    main()
    output = capsys.readouterr().out
    assert 'Validated 1 questions' in output and 'magic_0001_q01' in output
    assert 'game_0001_q01' not in output


def test_selected_qa_counts_resume_and_scope_change(config):
    add_game(config, video_exists=True)
    config['task'] = 'progressive_qa'
    config['categories'] = ['magic']
    model = AnswerModel()
    scores = evaluate(config, model=model)
    assert len(scores) == 3 and len(model.requests) == 3
    report = json.loads((Path(config['output_dir']) / 'summary.json').read_text())
    assert set(report['by_category']) == {'magic'}
    assert all(result['count'] == 1 for result in report['by_condition'].values())
    evaluate(config, model=model)
    assert len(model.requests) == 3
    changed = copy.deepcopy(config)
    changed['categories'] = ['game']
    with pytest.raises(ValueError, match='changed'):
        evaluate(changed, model=model)
    assert len(model.requests) == 3


def test_legacy_all_category_run_can_resume(config):
    config['task'] = 'progressive_qa'
    model = AnswerModel()
    evaluate(config, model=model)
    path = Path(config['output_dir']) / 'run_config.json'
    saved = json.loads(path.read_text())
    saved['settings'].pop('categories')
    saved['settings']['failure_scoring'] = 'zero; denominator=all_questions'
    saved['fingerprint'] = 'old-version-fingerprint'
    path.write_text(json.dumps(saved))
    evaluate(config, model=model)
    assert len(model.requests) == 3
