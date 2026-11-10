import json
from pathlib import Path

import pytest
from glint.models import ModelResponse, Text, Video
from glint.evaluation import evaluate
from glint.evaluation.dataset import load_samples
from glint.evaluation.parsing import parse_answer_response
from glint.evaluation.prompts import QA_CONDITIONS, qa_messages
from glint.evaluation.types import ParseError
from test_evaluation import config


class AnswerModel:
    def __init__(self, *, interrupt=False, malformed_baseline=False):
        self.requests = []
        self.interrupt = interrupt
        self.malformed_baseline = malformed_baseline

    def generate(self, messages, **kwargs):
        self.requests.append(messages)
        text = messages[0]['content'][0].value
        if self.interrupt and len(self.requests) == 2:
            self.interrupt = False
            raise KeyboardInterrupt()
        if 'supporting visual regions' in text:
            answer = 'A'
        elif 'Supporting visual evidence is located' in text:
            answer = 'A'
        else:
            answer = 'invalid output' if self.malformed_baseline else 'B'
        return ModelResponse(answer, 'fake-qa')


def test_independent_qa_inputs(config):
    sample = load_samples(config)[0]
    prompts = {condition: qa_messages(sample, condition) for condition in QA_CONDITIONS}
    for messages in prompts.values():
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        assert isinstance(messages[0]['content'][1], Video)
        assert messages[0]['content'][1].path == str(sample.video_path)
        text = messages[0]['content'][0].value
        assert 'Previous' not in text and 'predict' not in text
        assert 'Return only the selected option letter' in text
    text = prompts['baseline'][0]['content'][0].value
    assert '"start"' not in text and '"box"' not in text
    text = prompts['temporal_gt'][0]['content'][0].value
    assert '"start"' in text and '"box"' not in text
    text = prompts['spatiotemporal_gt'][0]['content'][0].value
    assert '"start"' in text and '"box"' in text


def test_three_separate_records_and_condition_scores(config):
    config['task'] = 'progressive_qa'
    model = AnswerModel()
    scores = evaluate(config, model=model)
    assert len(model.requests) == 3
    assert [row['accqa'] for row in scores] == [0, 1, 1]
    output = Path(config['output_dir'])
    for condition in QA_CONDITIONS:
        rows = [json.loads(line) for line in (output / f'predictions_{condition}.jsonl').read_text().splitlines()]
        assert len(rows) == 1
        row = rows[0]
        assert row['condition'] == condition and row['question_id'] == 'magic_0001_q01'
        assert 'raw_response' in row and 'predicted_answer' in row and 'correct' in row
        assert 'evidence_windows' not in row and 'evidence_boxes' not in row
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['accuracy_deltas']['temporal_minus_baseline'] == 1
    assert summary['by_condition']['baseline']['count'] == 1
    evaluate(config, model=model)
    assert len(model.requests) == 3
    evaluate(config, score_only=True)


def test_resume_each_condition(config):
    config['task'] = 'progressive_qa'
    model = AnswerModel(interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        evaluate(config, model=model)
    scores = evaluate(config, model=model)
    assert len(scores) == 3 and len(model.requests) == 4
    baseline_calls = [messages for messages in model.requests
                      if 'Supporting visual evidence is located' not in messages[0]['content'][0].value]
    assert len(baseline_calls) == 1


def test_parse_failure_does_not_stop_other_conditions(config):
    config['task'] = 'progressive_qa'
    scores = evaluate(config, model=AnswerModel(malformed_baseline=True))
    assert scores[0]['status'] == 'parse_error' and scores[0]['accqa'] == 0
    assert scores[1]['status'] == scores[2]['status'] == 'ok'
    model = AnswerModel()
    evaluate(config, model=model, retry_failed=True)
    assert len(model.requests) == 1


def test_qa_parser():
    assert parse_answer_response(' d\n') == 'D'
    with pytest.raises(ParseError):
        parse_answer_response('{"answer":"D","evidence_windows":[]}')
