import json
from pathlib import Path

from glint.evaluation import evaluate
from glint.evaluation.dataset import load_samples
from glint.evaluation.prompts import QA_CONDITIONS
from test_evaluation import config, make_video, PerfectModel
from test_category_selection import add_game
from test_progressive_qa import AnswerModel


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_category_condition_partitions_and_weighted_summary(config):
    add_game(config, video_exists=True)
    qa = Path(config['qa_path'])
    rows = read_jsonl(qa)
    rows[1]['answer'] = 'B'
    rows.append({**rows[0], 'question_id': 'magic_0001_q02'})
    qa.write_text('\n'.join(json.dumps(row) for row in rows))
    config['task'] = 'progressive_qa'
    model = AnswerModel()
    evaluate(config, model=model)
    output = Path(config['output_dir'])
    seen = set()
    for category, count in [('magic', 2), ('game', 1)]:
        for condition in QA_CONDITIONS:
            predictions = read_jsonl(output / category / condition / 'predictions.jsonl')
            assert len(predictions) == count
            for row in predictions:
                assert row['category'] == category and row['condition'] == condition
                assert 'history' not in row and 'usage' not in row
                key = (row['question_id'], row['condition'])
                assert key not in seen
                seen.add(key)
    assert len(seen) == 9
    overall = json.loads((output / 'summary.json').read_text())
    assert overall['by_condition']['baseline']['accqa'] == 1 / 3
    assert overall['by_category']['magic']['baseline']['accqa'] == 0
    assert overall['by_category']['game']['baseline']['accqa'] == 1
    assert {p.name for p in output.iterdir()} == {'magic', 'game', 'summary.json', 'run_config.json', 'checkpoints'}
    assert len(model.requests) == 9
    (output / 'magic/temporal_gt/predictions.jsonl').unlink()
    evaluate(config, score_only=True)
    assert (output / 'magic/temporal_gt/predictions.jsonl').is_file()
    assert len(model.requests) == 9


def test_grouped_grounding_results(config):
    make_video(config)
    evaluate(config, model=PerfectModel(load_samples(config)[0]))
    output = Path(config['output_dir'])
    result_dir = output / 'magic' / 'grounding'
    assert {p.name for p in output.iterdir()} == {'magic', '.run_state'}
    assert {p.name for p in result_dir.iterdir()} == {'predictions.jsonl', 'scores.jsonl', 'summary.json'}
    predictions = read_jsonl(result_dir / 'predictions.jsonl')
    scores = read_jsonl(result_dir / 'scores.jsonl')
    assert predictions[0]['question_id'] == scores[0]['question_id'] == 'magic_0001_q01'
    assert json.loads((result_dir / 'summary.json').read_text())['overall']['count'] == 1

    # Recreate the pre-change state layout and verify it can still be rescored.
    state_dir = output / '.run_state'
    for path in state_dir.iterdir():
        path.rename(output / path.name)
    state_dir.rmdir()
    assert evaluate(config, score_only=True) == scores
    assert not state_dir.exists()
