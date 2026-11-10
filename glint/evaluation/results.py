"""Atomic checkpoints and reproducible run identity."""
import hashlib
import json
import os
from pathlib import Path
from .config import normalize_selection
from glint.models import model_fingerprint


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def checkpoint_path(directory, question_id, condition=None):
    # Question IDs never become filesystem paths.
    key = hashlib.sha256(question_id.encode()).hexdigest()
    parent = Path(directory) / 'checkpoints'
    if condition is not None:
        from .prompts import QA_CONDITIONS
        if condition not in QA_CONDITIONS:
            raise ValueError(f'Unknown QA condition: {condition}')
        parent = parent / condition
    return parent / f'{key}.json'


def load_checkpoint(directory, question_id, condition=None):
    path = checkpoint_path(directory, question_id, condition)
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('question_id') != question_id or record.get('condition') != condition:
        raise ValueError('Checkpoint question ID mismatch')
    return record


def save_checkpoint(directory, record):
    atomic_json(checkpoint_path(directory, record['question_id'], record.get('condition')), record)


def run_identity(config, samples, prompt_version):
    # Model identity is an opaque fingerprint provided by the Adapter.
    keys = ['task', 'model', 'parse_retries']
    if config['task'] == 'grounding':
        keys.extend(['k', 'c', 'tau'])
    public = {key: config[key] for key in keys}
    public.update(normalize_selection(config))
    public.update({'prompt_version': prompt_version,
                   'protocol': ('single-window-gt-intersection-v1' if config['task'] == 'grounding'
                                else 'full-video-independent-evidence-hints-v1'),
                   'failure_scoring': 'zero; denominator=selected_questions'})
    public['model_fingerprint'] = model_fingerprint(config['model'], config_path=config['model_config'])
    public['data_hashes'] = {key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
                            for key in ('qa_path', 'metadata_path')}
    public['videos'] = [{'video_id': sample.video_id, 'path': str(sample.video_path),
                         'size': sample.video_path.stat().st_size,
                         'mtime_ns': sample.video_path.stat().st_mtime_ns} for sample in samples]
    fingerprint = hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
    return {'fingerprint': fingerprint, 'settings': public}


def ensure_run(directory, identity, *, score_only=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / 'run_config.json'
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding='utf-8'))
        if existing['fingerprint'] != identity['fingerprint']:
            old = dict(existing['settings'])
            new = dict(identity['settings'])
            # Older runs implicitly selected every category.
            old.update(normalize_selection(old))
            old['failure_scoring'] = old['failure_scoring'].replace('denominator=all_questions', 'denominator=selected_questions')
            if score_only:
                for key in ('c', 'tau'):
                    old.pop(key, None)
                    new.pop(key, None)
            if old != new:
                raise ValueError('Run configuration or dataset changed; use another output_dir')
    elif (directory / '.run_state').exists():
        raise ValueError('Output directory contains grounding state; use another output_dir')
    elif (directory / 'checkpoints').exists():
        raise ValueError('Checkpoints exist without run_config.json; use another output_dir')
    else:
        atomic_json(config_path, identity)


def summary(scores, records):
    def aggregate(rows):
        count = len(rows)
        return {'count': count,
                **{metric: sum(row[metric] for row in rows) / count if count else None
                   for metric in ('accqa', 'tnwd', 'tg_siou')},
                'gate_passed': sum(row['gate_passed'] for row in rows),
                'status_counts': {state: sum(row['status'] == state for row in rows)
                                  for state in sorted({row['status'] for row in rows})}}
    return {'overall': aggregate(scores), 'by_category': {
        category: aggregate([row for row in scores if row['category'] == category])
        for category in sorted({row['category'] for row in scores})},
        'stage_failures': {stage: {state: sum(record.get(stage, {}).get('status') == state for record in records)
                                   for state in ('call_error', 'parse_error', 'sampling_error')}
                           for stage in ('video', 'spatial')},
        'denominator': 'selected_questions; call/parse failures score zero for affected metrics'}


def atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_outputs(directory, records, scores):
    directory = Path(directory)
    for category in sorted({record['category'] for record in records}):
        result_dir = directory / category / 'grounding'
        category_records = [row for row in records if row['category'] == category]
        category_scores = [row for row in scores if row['category'] == category]
        atomic_jsonl(result_dir / 'predictions.jsonl', category_records)
        atomic_jsonl(result_dir / 'scores.jsonl', category_scores)
        atomic_json(result_dir / 'summary.json', summary(category_scores, category_records))
