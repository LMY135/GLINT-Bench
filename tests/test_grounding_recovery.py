"""Tests for conservative offline Grounding recovery."""
import json
import math
import pytest
from glint.evaluation.grounding_recovery import _extract_single_json, _video_offline, _spatial_offline, recover_grounding_parse_errors
from glint.evaluation.types import ParseError


def test_explanatory_note():
    text = '{"answer":"D","evidence_windows":[{"start":12.1,"end":12.5}]} (Note: visual check)'
    parsed = _video_offline(text, 15)
    assert parsed.answer == 'D'
    assert parsed.evidence_windows == [{'start': 12.1, 'end': 12.5}]
    assert not parsed.errors


def test_single_array_and_spatial():
    parsed = _video_offline('[{"answer":"A","evidence_windows":[{"start":1,"end":2}]}]', 10)
    assert parsed.answer == 'A'
    assert _spatial_offline('{"evidence_boxes":[{"frame":2,"box":[1,2,3,4]}]}',
                            [{'frame': 2}], 30, 40) == [{'frame': 2, 'box': [1, 2, 3, 4]}]


@pytest.mark.parametrize('raw', [
    '{"answer":"A"}{"answer":"B"}',
    '{"answer":"A"} Wait no, answer is B',
    '{"answer":"A","evidence_windows":[{"start":0,"end":1}]} (Actually, change the answer to B)',
    '{"answer":"A","evidence_windows":[{"start":0,"end":1}]',
    '<think>{"answer":"A"} still reasoning',
])
def test_ambiguous_or_incomplete_json_rejected(raw):
    with pytest.raises(ParseError):
        _extract_single_json(raw)


def test_strict_temporal_spatial():
    p = _video_offline('{"answer":"A","evidence_windows":[{"start":9,"end":15}]}', 10)
    assert p.answer == 'A' and not p.evidence_windows and 'temporal' in p.errors
    with pytest.raises(ParseError):
        _spatial_offline('{"evidence_boxes":[{"frame":0,"box":[1,2,3,4]}]}',
                         [{'frame': 2}], 30, 40)
    with pytest.raises(ParseError):
        _spatial_offline('{"evidence_boxes":[{"frame":2,"box":[29,2,3,4]}]}',
                         [{'frame': 2}], 30, 40)


def test_full_recovery_without_video_or_model(tmp_path):
    root = tmp_path / 'result'
    p = root / 'crowd' / 'grounding'
    p.mkdir(parents=True)
    qa = tmp_path / 'qa.jsonl'
    meta = tmp_path / 'meta.jsonl'
    qa.write_text(json.dumps({
        'question_id': 'crowd_0002_q01', 'video_id': 'crowd_0002', 'category': 'crowd',
        'answer': 'D', 'evidence_windows': [{'start': 12.1, 'end': 12.5}],
        'evidence_boxes': [{'frame': 123, 'box': [1, 2, 3, 4]}],
    }) + '\n')
    meta.write_text(json.dumps({
        'video_id': 'crowd_0002', 'duration': 15, 'width': 50, 'height': 50,
    }) + '\n')
    record = {
        'question_id': 'crowd_0002_q01', 'video_id': 'crowd_0002', 'category': 'crowd',
        'status': 'parse_error', 'frames': [],
        'video': {'status': 'parse_error', 'prediction': {'answer': None, 'evidence_windows': [],
                  'errors': {'answer': 'Expected one JSON object', 'temporal': 'Expected one JSON object'}},
                  'history': [{'text': '{"answer":"D","evidence_windows":[{"start":12.1,"end":12.5}]} (Note: checked)'}]},
        'spatial': {'status': 'skipped_gate', 'prediction': []},
    }
    score = {'question_id': 'crowd_0002_q01', 'category': 'crowd', 'accqa': 0,
             'tnwd': 0, 'gate_passed': False, 'siou': None, 'tg_siou': 0,
             'sampled_frames': 0, 'status': 'parse_error'}
    original_p, original_s = p / 'predictions.jsonl', p / 'scores.jsonl'
    original_p.write_text(json.dumps(record) + '\n')
    original_s.write_text(json.dumps(score) + '\n')
    before_p, before_s = original_p.read_bytes(), original_s.read_bytes()
    recovered = recover_grounding_parse_errors({
        'output_dir': str(root), 'qa_path': str(qa), 'metadata_path': str(meta),
        'categories': [], 'c': -0.5 / math.log(0.5), 'tau': 0.5,
    })
    assert recovered[0]['accqa'] == 1 and recovered[0]['tnwd'] == 1
    assert recovered[0]['tg_siou'] == 0
    assert original_p.read_bytes() == before_p and original_s.read_bytes() == before_s
    stats = json.loads((root / 'recovered' / 'recovery_summary.json').read_text())
    assert stats['video_recovered'] == 1 and stats['original_files_unchanged']
