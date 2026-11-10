"""Two-stage native-video evaluation with stage-level checkpoints."""
from dataclasses import asdict
from pathlib import Path
import shutil
import subprocess
from glint.models import get_model
from .calling import call_model, response_record
from .dataset import load_samples
from .metrics import temporal_gate, score_prediction
from .parsing import parse_video_response, parse_spatial_response
from .prompts import PROMPT_VERSION, video_messages, spatial_messages
from .results import (run_identity, ensure_run, load_checkpoint, save_checkpoint,
                      checkpoint_path, write_outputs, atomic_json)
from .sampling import sample_frames
from .types import CallFailure, ParseError


def video_stage(model, sample, config):
    history, prediction = [], None
    messages = video_messages(sample)
    for _ in range(config['parse_retries'] + 1):
        try:
            response, attempts = call_model(model, messages)
        except CallFailure as exc:
            history.append({'attempts': exc.attempts, 'error_type': type(exc.cause).__name__,
                            'error': str(exc)})
            # Retain a valid answer/interval from the last response if a regeneration fails.
            return {'status': 'call_error', 'prediction': asdict(prediction) if prediction else {},
                    'history': history}
        prediction = parse_video_response(response.text, sample.metadata['duration'])
        history.append({**response_record(response), 'attempts': attempts, 'errors': prediction.errors})
        if not prediction.errors:
            break
    return {'status': 'parse_error' if prediction.errors else 'ok',
            'prediction': asdict(prediction), 'history': history}


def spatial_stage(model, sample, prediction, frames, config):
    history = []
    messages = spatial_messages(sample, prediction, frames)
    for _ in range(config['parse_retries'] + 1):
        try:
            response, attempts = call_model(model, messages)
        except CallFailure as exc:
            history.append({'attempts': exc.attempts, 'error_type': type(exc.cause).__name__,
                            'error': str(exc)})
            return {'status': 'call_error', 'prediction': [], 'history': history}
        entry = {**response_record(response), 'attempts': attempts}
        try:
            boxes = parse_spatial_response(response.text, frames,
                                           sample.metadata['width'], sample.metadata['height'])
        except ParseError as exc:
            entry['error'] = str(exc)
            history.append(entry)
            continue
        history.append(entry)
        return {'status': 'ok', 'prediction': boxes, 'history': history}
    return {'status': 'parse_error', 'prediction': [], 'history': history}


def run_sample(model, sample, config, state_dir, record=None):
    if record is None:
        record = {'question_id': sample.question_id, 'video_id': sample.video_id,
                  'category': sample.category, 'status': 'pending', 'frames': []}
    if 'video' not in record:
        record['video'] = video_stage(model, sample, config)
        save_checkpoint(state_dir, record)
    prediction = record['video']['prediction']
    passed, intersection = temporal_gate(prediction.get('evidence_windows', []),
                                          sample.evidence_windows, config['c'], config['tau'])
    if not passed:
        record['spatial'] = {'status': 'skipped_gate', 'prediction': []}
    elif record.get('spatial', {}).get('status') != 'ok':
        cache_key = checkpoint_path(state_dir, sample.question_id).stem
        try:
            record['frames'] = sample_frames(sample, intersection, config['k'],
                                              state_dir / 'frames' / cache_key)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            record['spatial'] = {'status': 'sampling_error', 'prediction': [],
                                 'error_type': type(exc).__name__, 'error': str(exc)}
            print(f'{sample.question_id}: sampling_error: {exc}', flush=True)
        else:
            save_checkpoint(state_dir, record)
            record['spatial'] = spatial_stage(model, sample, prediction, record['frames'], config)
    states = [record['video']['status'], record['spatial']['status']]
    record['status'] = next((state for state in states if state not in ('ok', 'skipped_gate')), 'ok')
    save_checkpoint(state_dir, record)
    return record


def evaluate_grounding(config, *, retry_failed=False, model=None, score_only=False):
    samples = load_samples(config)
    identity = run_identity(config, samples, PROMPT_VERSION)
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    # mkdir is atomic; concurrent writers to the same run are rejected.
    lock = output / '.run.lock'
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f'Run is locked: {lock}. Remove a stale lock after confirming no writer is active.') from exc
    records, scores = [], []
    try:
        # Keep existing runs resumable in their original checkpoint location.
        legacy = (output / 'run_config.json').exists() or (output / 'checkpoints').exists()
        state_dir = output if legacy else output / '.run_state'
        ensure_run(state_dir, identity, score_only=score_only)
        for sample in samples:
            record = load_checkpoint(state_dir, sample.question_id)
            if score_only:
                if record is None or record.get('status') == 'pending':
                    raise ValueError(f'{sample.question_id}: missing completed prediction')
            elif record is None or record.get('status') == 'pending' or (retry_failed and record['status'] != 'ok'):
                if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:
                    raise RuntimeError('Install ffmpeg and ffprobe for spatial frame extraction')
                if model is None:
                    model = get_model(config['model'], config_path=config['model_config'])
                if retry_failed and record and record['video']['status'] != 'ok':
                    record = None
                record = run_sample(model, sample, config, state_dir, record)
            if score_only:
                passed, _ = temporal_gate(record['video']['prediction'].get('evidence_windows', []),
                                           sample.evidence_windows, config['c'], config['tau'])
                if passed and record.get('spatial', {}).get('status') == 'skipped_gate':
                    raise ValueError(f'{sample.question_id}: new gate requires spatial inference; use a new run')
            records.append(record)
            scores.append(score_prediction(sample, record, config['c'], config['tau']))
            print(f'{len(records)}/{len(samples)} {sample.question_id}: {record["status"]}', flush=True)
        write_outputs(output, records, scores)
        atomic_json(state_dir / 'score_config.json', {'c': config['c'], 'tau': config['tau']})
    finally:
        lock.rmdir()
    return scores


def evaluate(config, *, retry_failed=False, model=None, score_only=False):
    if config['task'] == 'progressive_qa':
        from .progressive import evaluate_progressive
        return evaluate_progressive(config, retry_failed=retry_failed, model=model, score_only=score_only)
    return evaluate_grounding(config, retry_failed=retry_failed, model=model, score_only=score_only)
