"""Strict JSON parsing, with independent answer and temporal validity."""
import json
from .dataset import validate_box, validate_window
from .types import ParseError, VideoPrediction


def parse_object(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ParseError('Expected one JSON object') from exc
    if not isinstance(value, dict):
        raise ParseError('Expected one JSON object')
    return value


def parse_video_response(text, duration):
    result = VideoPrediction()
    try:
        data = parse_object(text)
    except ParseError as exc:
        result.errors = {'answer': str(exc), 'temporal': str(exc)}
        return result
    answer = data.get('answer')
    if isinstance(answer, str) and answer.strip().upper() in ('A', 'B', 'C', 'D'):
        result.answer = answer.strip().upper()
    else:
        result.errors['answer'] = 'Expected answer A/B/C/D'
    try:
        windows = data.get('evidence_windows')
        if not isinstance(windows, list) or len(windows) != 1:
            raise ValueError('Expected exactly one evidence window')
        validate_window(windows[0], duration)
        result.evidence_windows = [{'start': windows[0]['start'], 'end': windows[0]['end']}]
    except ValueError as exc:
        result.errors['temporal'] = str(exc)
    return result


def parse_spatial_response(text, frames, width, height):
    entries = parse_object(text).get('evidence_boxes')
    if not isinstance(entries, list):
        raise ParseError('Expected evidence_boxes list')
    allowed = {entry['frame'] for entry in frames}
    boxes = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ParseError('Expected a frame/box object')
        frame = entry.get('frame')
        if type(frame) is not int or frame not in allowed or frame in boxes:
            raise ParseError(f'Unexpected or duplicate frame: {frame}')
        try:
            validate_box(entry.get('box'), width, height)
        except ValueError as exc:
            raise ParseError(f'Frame {frame}: {exc}') from exc
        boxes[frame] = entry['box']
    # Missing frames remain absent; scoring gives each missing box zero.
    return [{'frame': frame, 'box': box} for frame, box in boxes.items()]


def parse_answer_response(text):
    """The QA-only protocol requests a single option, independent of the GT schema."""
    answer = text.strip().upper()
    if answer not in ('A', 'B', 'C', 'D'):
        raise ParseError('Expected only the selected option letter A/B/C/D')
    return answer
