"""Offline recovery of historical grounding parsing errors; never invokes a model."""
from copy import deepcopy
from pathlib import Path
import json
import re
from types import SimpleNamespace

from .dataset import read_jsonl, validate_window, validate_box
from .metrics import score_prediction, temporal_gate
from .parsing import parse_video_response, parse_spatial_response
from .results import atomic_json, atomic_jsonl, summary, write_outputs
from .types import ParseError


def _extract_single_json(text):
    """Return one complete JSON object/array; ignore only non-answer wrappers.

    Never repair truncated JSON, infer a missing answer, or accept competing
    JSON objects / a trailing change-of-mind.
    """
    if not isinstance(text, str) or not text.strip():
        raise ParseError('No saved response text')
    raw = text.strip()
    if raw.startswith('```'):
        raw = re.sub(r'^```(?:json)?\s*', '', raw, count=1, flags=re.IGNORECASE)
        raw = re.sub(r'\s*```', '', raw, count=1)
    decoder = json.JSONDecoder()
    candidates = []
    # Never count nested objects inside one complete top-level JSON payload as
    # competing predictions. Nested evidence_windows/boxes are expected.
    consumed_until = 0
    for i, char in enumerate(raw):
        if i < consumed_until or char not in '{[':
            continue
        try:
            value, used = decoder.raw_decode(raw[i:])
        except (ValueError, TypeError):
            continue
        if isinstance(value, (dict, list)):
            candidates.append((i, i + used, value))
            consumed_until = i + used
    if len(candidates) != 1:
        raise ParseError('Missing or multiple complete JSON predictions')
    start, end, value = candidates[0]
    prefix, suffix = raw[:start].strip(), raw[end:].strip()
    # No extra structured content, thinking tokens or new answer at the end.
    if any(c in prefix + suffix for c in '{}[]'):
        raise ParseError('Extra structured content around JSON')
    if re.search(r'<(?:think|answer)\b|(?:wait|actually|rather|instead|correction)\b',
                 suffix, re.IGNORECASE):
        raise ParseError('Possible revised or unfinished prediction after JSON')
    if re.search(r'\b(?:answer|option)\s*(?:is|:)\s*[A-D]\b', suffix, re.IGNORECASE):
        raise ParseError('Another answer appears after JSON')
    if prefix and not re.fullmatch(
        r'(?is)(?:here\s+is\s+)?(?:the\s+)?(?:final\s+)?(?:json|response|prediction|answer)\s*[:：]?',
        prefix,
    ):
        raise ParseError('Unrecognized prefix before JSON')
    if suffix and not (
        suffix == '.'
        or re.fullmatch(r'\([^{}[\]]*\)', suffix, re.DOTALL)
        or re.fullmatch(r'```', suffix)
    ):
        raise ParseError('Unrecognized text after JSON')
    return value


def _video_offline(text, duration):
    value = _extract_single_json(text)
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict):
        raise ParseError('Expected one video JSON object')
    # Reuse the normal strict answer + temporal parser for validation.
    return parse_video_response(json.dumps(value), duration)


def _spatial_offline(text, frames, width, height):
    value = _extract_single_json(text)
    return parse_spatial_response(json.dumps(value), frames, width, height)


def _latest_saved_text(stage):
    for entry in reversed(stage.get('history', [])):
        if isinstance(entry.get('text'), str) and entry['text'].strip():
            return entry['text']
    return None


def _sample_from_saved_gt(record, qa_by_id, meta_by_id):
    qid, vid = record['question_id'], record['video_id']
    qa = qa_by_id.get(qid)
    meta = meta_by_id.get(vid)
    if not qa or not meta:
        raise ValueError(f'{qid}: missing current QA or metadata; cannot rescore')
    if qa.get('video_id') != vid or qa.get('category') != record['category']:
        raise ValueError(f'{qid}: QA identity changed')
    if qa.get('answer') not in ('A', 'B', 'C', 'D'):
        raise ValueError(f'{qid}: invalid QA answer')
    windows = qa['evidence_windows']
    if not isinstance(windows, list) or len(windows) != 1:
        raise ValueError(f'{qid}: expected one GT window')
    validate_window(windows[0], meta['duration'])
    boxes = qa['evidence_boxes']
    if not isinstance(boxes, list) or not boxes:
        raise ValueError(f'{qid}: missing GT boxes')
    for entry in boxes:
        validate_box(entry['box'], meta['width'], meta['height'])
    return SimpleNamespace(question_id=qid, video_id=vid, category=record['category'],
                           answer=qa['answer'], evidence_windows=windows,
                           evidence_boxes=boxes, metadata=meta)


def recover_grounding_parse_errors(config):
    """Offline reparse exported grounding results; keep all originals unchanged."""
    source = Path(config['output_dir'])
    if not source.is_dir():
        raise FileNotFoundError(f'Missing grounding result directory: {source}')
    if (source / '.run.lock').exists():
        raise RuntimeError(f'Run is locked: {source / ".run.lock"}')

    groups = []
    for path in sorted(source.glob('*/grounding/predictions.jsonl')):
        category = path.parent.parent.name
        if config.get('categories') and category not in config['categories']:
            continue
        score_path = path.with_name('scores.jsonl')
        if not score_path.is_file():
            raise FileNotFoundError(f'Missing original scores: {score_path}')
        predictions, scores = read_jsonl(path), read_jsonl(score_path)
        score_by_id = {row['question_id']: row for row in scores}
        if len(score_by_id) != len(scores) or len(predictions) != len(scores):
            raise ValueError(f'Duplicate/mismatched grounding scores: {path}')
        if {r['question_id'] for r in predictions} != set(score_by_id):
            raise ValueError(f'Prediction-score ID mismatch: {path}')
        for row in predictions:
            if row.get('category') != category or score_by_id[row['question_id']].get('category') != category:
                raise ValueError(f'Category mismatch: {path}')
        groups.append((predictions, score_by_id))
    if not groups:
        raise ValueError(f'No grounding predictions found in {source}')

    qa_by_id = {r['question_id']: r for r in read_jsonl(config['qa_path'])}
    meta_by_id = {r['video_id']: r for r in read_jsonl(config['metadata_path'])}
    records, scores, unresolved = [], [], []
    video_recovered = spatial_recovered = 0
    video_parse_errors = spatial_parse_errors = 0
    for predictions, original_score_by_id in groups:
        for original in predictions:
            record = deepcopy(original)
            score = deepcopy(original_score_by_id[record['question_id']])
            video = record.get('video', {})
            spatial = record.get('spatial', {})
            video_changed = spatial_changed = False

            if video.get('status') == 'parse_error':
                video_parse_errors += 1
                text = _latest_saved_text(video)
                try:
                    sample = _sample_from_saved_gt(record, qa_by_id, meta_by_id)
                    recovered = _video_offline(text, sample.metadata['duration'])
                    # Allow independently valid answer or temporal interval,
                    # but do not overwrite either with invalid data.
                    existing = video.get('prediction') or {}
                    new_prediction = dict(existing)
                    if recovered.answer is not None and new_prediction.get('answer') is None:
                        new_prediction['answer'] = recovered.answer
                        video_changed = True
                    if recovered.evidence_windows and not new_prediction.get('evidence_windows'):
                        new_prediction['evidence_windows'] = recovered.evidence_windows
                        video_changed = True
                    if video_changed:
                        new_prediction['errors'] = recovered.errors
                        video['prediction'] = new_prediction
                        video['status'] = 'ok' if not recovered.errors else 'parse_error'
                        video['recovered_from_saved_response'] = True
                        video_recovered += 1
                except (ParseError, ValueError, KeyError, TypeError) as exc:
                    unresolved.append({'question_id': record['question_id'],
                                       'stage': 'video', 'reason': str(exc)})

            sample = None
            if video_changed or spatial.get('status') == 'parse_error':
                sample = _sample_from_saved_gt(record, qa_by_id, meta_by_id)
            if spatial.get('status') == 'parse_error':
                spatial_parse_errors += 1
                frames = record.get('frames') or []
                try:
                    if not frames:
                        raise ParseError('No saved sampled frames')
                    text = _latest_saved_text(spatial)
                    boxes = _spatial_offline(text, frames, sample.metadata['width'],
                                             sample.metadata['height'])
                    spatial['prediction'] = boxes
                    spatial['status'] = 'ok'
                    spatial['recovered_from_saved_response'] = True
                    spatial_recovered += 1
                    spatial_changed = True
                except (ParseError, ValueError, KeyError, TypeError) as exc:
                    unresolved.append({'question_id': record['question_id'],
                                       'stage': 'spatial', 'reason': str(exc)})

            if video_changed or spatial_changed:
                if sample is None:
                    sample = _sample_from_saved_gt(record, qa_by_id, meta_by_id)
                gate, _ = temporal_gate(
                    video.get('prediction', {}).get('evidence_windows', []),
                    sample.evidence_windows, config['c'], config['tau'])
                if gate and spatial.get('status') == 'skipped_gate':
                    # The original run has no spatial inference to recover.
                    # Keep missing spatial evidence scored zero.
                    spatial['status'] = 'skipped_no_spatial_prediction'
                states = [video.get('status'), spatial.get('status')]
                record['status'] = next((s for s in states if s not in
                                         ('ok', 'skipped_gate', 'skipped_no_spatial_prediction')), 'ok')
                score = score_prediction(sample, record, config['c'], config['tau'])
                # A newly recovered video window may change the temporal gate,
                # but never fabricates bounding boxes.
            records.append(record)
            scores.append(score)

    dest = source / 'recovered'
    write_outputs(dest, records, scores, write_thoughts=False)
    recovered_summary = summary(scores, records)
    atomic_json(dest / 'recovery_summary.json', {
        'source_run': str(source),
        'method': 'offline_grounding_reparse_exported_results',
        'video_parse_errors': video_parse_errors,
        'video_recovered': video_recovered,
        'spatial_parse_errors': spatial_parse_errors,
        'spatial_recovered': spatial_recovered,
        'unresolved': unresolved,
        'recovered_scores': recovered_summary,
        'note': ('Original scores are preserved for untouched records. Updated records use current '
                 'QA/metadata ground truth and original metric implementation; verify historical '
                 'ground truth and scoring parameters c/tau match the original run. Missing '
                 'spatial predictions remain zero, even if a recovered video window passes the gate.'),
        'original_files_unchanged': True,
    })
    print(f'Grounding offline recovery: video {video_recovered}/{video_parse_errors}, '
          f'spatial {spatial_recovered}/{spatial_parse_errors}; '
          f'output: {dest}; no API calls.')
    return scores
