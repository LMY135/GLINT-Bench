import copy
import json
import sys
from pathlib import Path

import pytest
import yaml
from glint.evaluation import evaluate
from glint.evaluation.__main__ import main
from glint.evaluation.config import load_config, normalize_selection
from glint.evaluation.dataset import load_samples
from test_evaluation import config
from test_progressive_qa import AnswerModel


def add_records(config, create_extra_videos=False):
    path = Path(config['qa_path'])
    first = json.loads(path.read_text())
    records = [first,
               {**first, 'question_id': 'magic_0001_q02'},
               {**first, 'question_id': 'game_0001_q01', 'video_id': 'game_0001', 'category': 'game'},
               {**first, 'question_id': 'magic_0001_q03'},
               {**first, 'question_id': 'magic_0002_q01', 'video_id': 'magic_0002'}]
    path.write_text('\n'.join(json.dumps(row) for row in records))
    meta_path = Path(config['metadata_path'])
    first_meta = json.loads(meta_path.read_text())
    metadata = [first_meta, {**first_meta, 'video_id': 'game_0001'}, {**first_meta, 'video_id': 'magic_0002'}]
    meta_path.write_text('\n'.join(json.dumps(row) for row in metadata))
    if create_extra_videos:
        for category, video_id in [('game', 'game_0001'), ('magic', 'magic_0002')]:
            video = Path(config['videos_dir']) / category / f'{video_id}.mp4'
            video.parent.mkdir(exist_ok=True)
            video.touch()


def test_limit_qa_records_before_video_validation(config):
    add_records(config)
    config['limit'] = 2
    samples = load_samples(config)
    assert [s.question_id for s in samples] == ['magic_0001_q01', 'magic_0001_q02']
    assert len({s.video_id for s in samples}) == 1


def test_video_limit_preserves_all_questions_and_original_order(config):
    add_records(config, create_extra_videos=True)
    config['video_limit'] = 2
    samples = load_samples(config)
    assert [s.question_id for s in samples] == [
        'magic_0001_q01', 'magic_0001_q02', 'game_0001_q01', 'magic_0001_q03']
    assert len({s.video_id for s in samples}) == 2


def test_exact_video_and_question_ids(config):
    add_records(config)
    config['video_ids'] = ['magic_0001']
    assert len(load_samples(config)) == 3
    config['question_ids'] = ['magic_0001_q03', 'magic_0001_q01']
    assert [s.question_id for s in load_samples(config)] == ['magic_0001_q01', 'magic_0001_q03']
    config['limit'] = 1
    assert load_samples(config)[0].question_id == 'magic_0001_q01'


def test_categories_ids_and_limits_intersect(config):
    add_records(config)
    config.update({'categories': ['magic'], 'video_limit': 1, 'limit': 2})
    assert [s.question_id for s in load_samples(config)] == ['magic_0001_q01', 'magic_0001_q02']
    config['question_ids'] = ['game_0001_q01']
    with pytest.raises(ValueError, match='No QA samples matched'):
        load_samples(config)


def test_missing_id_rejected_even_if_another_id_matches(config):
    config['video_ids'] = ['magic_0001', 'magic_9999']
    with pytest.raises(ValueError, match='Unknown video_ids'):
        load_samples(config)
    config['video_ids'] = []
    config['question_ids'] = ['does_not_exist']
    with pytest.raises(ValueError, match='Unknown question_ids'):
        load_samples(config)


@pytest.mark.parametrize('key,value', [('limit', 0), ('limit', -1), ('limit', True),
                                     ('video_limit', 1.5), ('video_ids', 'magic_0001'),
                                     ('video_ids', False), ('question_ids', [''])])
def test_invalid_selection_parameters(key, value):
    with pytest.raises(ValueError):
        normalize_selection({key: value})


def test_yaml_and_cli_id_filters(config, monkeypatch, capsys):
    add_records(config)
    path = Path(config['project_root']) / 'configs/evaluation.yaml'
    data = yaml.safe_load(path.read_text())
    data.update({'video_ids': ['magic_0001'], 'limit': 2})
    path.write_text(yaml.safe_dump(data))
    loaded = load_config(path)
    assert loaded['limit'] == 2 and loaded['video_ids'] == ['magic_0001']
    monkeypatch.setattr(sys, 'argv', ['evaluation', '--config', str(path), '--question-ids',
                                    'magic_0001_q03', '--limit', '1', '--validate-only'])
    main()
    output = capsys.readouterr().out
    assert 'Validated 1 questions' in output and 'magic_0001_q03' in output


def test_subset_checkpoint_and_summary(config):
    add_records(config)
    config.update({'task': 'progressive_qa', 'limit': 2})
    model = AnswerModel()
    scores = evaluate(config, model=model)
    assert len(scores) == 6 and len(model.requests) == 6
    output = Path(config['output_dir'])
    summary = json.loads((output / 'summary.json').read_text())
    assert all(row['count'] == 2 for row in summary['by_condition'].values())
    evaluate(config, model=model)
    assert len(model.requests) == 6
    changed = copy.deepcopy(config)
    changed.update({'limit': 1})
    with pytest.raises(ValueError, match='changed'):
        evaluate(changed, model=model)
    identity = json.loads((output / 'run_config.json').read_text())
    assert identity['settings']['limit'] == 2
