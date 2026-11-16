"""Strict JSON parsing, with independent answer and temporal validity."""
import json
import re
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
    # Accept the documented object envelope as well as a bare list of boxes.
    # Some models emit a list of envelopes instead of a single envelope.
    raw = text.strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw = raw.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ParseError('Expected valid spatial JSON object or array') from exc

    if isinstance(data, dict):
        entries = data.get('evidence_boxes')
    elif isinstance(data, list):
        if data and all(isinstance(item, dict) and 'evidence_boxes' in item for item in data):
            entries = []
            for item in data:
                if not isinstance(item['evidence_boxes'], list):
                    raise ParseError('Expected evidence_boxes list')
                entries.extend(item['evidence_boxes'])
        else:
            entries = data
    else:
        raise ParseError('Expected spatial JSON object or array')
    if not isinstance(entries, list):
        raise ParseError('Expected evidence_boxes list')

    allowed = {entry['frame'] for entry in frames}
    boxes = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ParseError('Expected a frame/box object')
        frame = entry.get('frame')
        # Convert numeric frame identifiers only; do not reinterpret frame order.
        if isinstance(frame, str) and frame.isascii() and frame.isdecimal():
            frame = int(frame)
        if type(frame) is not int or frame not in allowed or frame in boxes:
            raise ParseError(f'Unexpected or duplicate frame: {frame}')
        try:
            # The evaluation schema is strictly [x, y, width, height].
            # Never guess whether a model intended xyxy coordinates.
            validate_box(entry.get('box'), width, height)
        except ValueError as exc:
            raise ParseError(f'Frame {frame}: {exc}') from exc
        boxes[frame] = entry['box']
    # Missing frames remain absent; scoring gives each missing box zero.
    return [{'frame': frame, 'box': box} for frame, box in boxes.items()]


def parse_answer_with_method(text):
    """Parse only an explicit final QA choice, never infer from reasoning."""
    if not isinstance(text, str) or not text.strip():
        raise ParseError('Missing final option A/B/C/D')
    raw = text.strip()

    # Thinking models can repeat malformed <answer> tags. Accept a single
    # A/B/C/D only when it terminates the response, including optional closing
    # </answer>. Earlier explicit answer tags must not conflict with it.
    tags = list(re.finditer(
        r'<answer\b[^>]*>\s*(?:\*\*)?([A-D])(?:\*\*)?\s*(?:</answer\s*>)?',
        raw, flags=re.IGNORECASE,
    ))
    if tags:
        last = tags[-1]
        if raw[last.end():].strip():
            raise ParseError('Text follows final answer tag')
        if any(tag.group(1).upper() != last.group(1).upper() for tag in tags):
            raise ParseError('Conflicting final answer tags')
        return last.group(1).upper(), 'answer_tag'

    if re.search(r'<answer\b', raw, re.IGNORECASE):
        raise ParseError('Incomplete final answer tag')
    if re.search(r'<think\b', raw, re.IGNORECASE) and not re.search(
        r'</think\s*>', raw, re.IGNORECASE,
    ):
        raise ParseError('Unfinished thinking response')

    # Accept one valid JSON object (or a single-element object array) with
    # optional safe trailing punctuation/note, but never additional answers.
    json_raw = raw
    fenced = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', raw,
                          re.IGNORECASE | re.DOTALL)
    if fenced:
        json_raw = fenced.group(1).strip()
    if json_raw.startswith(('{', '[')):
        try:
            data, end = json.JSONDecoder().raw_decode(json_raw)
        except (ValueError, TypeError) as exc:
            raise ParseError('Invalid final-answer JSON') from exc
        trailing = json_raw[end:].strip()
        if trailing and not (
            trailing == '.' or
            (trailing.startswith('(') and trailing.endswith(')')
             and not re.search(r'<answer\b|\{\s*"answer"\s*:', trailing, re.IGNORECASE))
        ):
            raise ParseError('Unexpected text after final JSON answer')
        if isinstance(data, list) and len(data) == 1:
            data = data[0]
        if not isinstance(data, dict):
            raise ParseError('Expected one final-answer JSON object')
        answer = data.get('answer')
        if isinstance(answer, str) and answer.strip().upper() in ('A', 'B', 'C', 'D'):
            return answer.strip().upper(), 'json_answer'
        raise ParseError('Missing JSON answer A/B/C/D')

    option = r'(?:answer\s*:\s*)?(?:\*\*([A-D])\*\*|([A-D]))\s*\.?'
    match = re.fullmatch(option, raw, flags=re.IGNORECASE)
    if match:
        return (match.group(1) or match.group(2)).upper(), 'plain_option'

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) > 1:
        match = re.fullmatch(option, lines[-1], flags=re.IGNORECASE)
        if match:
            return (match.group(1) or match.group(2)).upper(), 'final_line'

    # A sentence *at the very end* explicitly committing to a choice is
    # acceptable. "Should be A ... Wait, no" is not, because it is not final.
    match = re.search(
        r'(?:^|[.!?]\s+|\n)\s*(?:so\s+)?(?:the\s+)?'
        r'(?:final\s+(?:answer|result|choice)|correct\s+(?:answer|choice)|answer)'
        r'\s+(?:is|:)\s*(?:\*\*)?([A-D])(?:\*\*)?\s*\.?\s*$',
        raw, flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).upper(), 'final_statement'
    raise ParseError('No explicit final option A/B/C/D')


def parse_answer_response(text):
    """Return the final A/B/C/D option for ordinary QA evaluation."""
    return parse_answer_with_method(text)[0]
