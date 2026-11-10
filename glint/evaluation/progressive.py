"""Three independent answer-only requests per question, one record per condition."""
from pathlib import Path
from glint.models import get_model
from .calling import call_model, response_record
from .dataset import load_samples
from .parsing import parse_answer_response
from .prompts import QA_CONDITIONS, qa_messages
from .results import (run_identity, ensure_run, load_checkpoint, save_checkpoint,
                      atomic_json, atomic_jsonl)
from .types import CallFailure, ParseError

QA_PROMPT_VERSION = 'glint-progressive-qa-v1'


def predict_answer(model, sample, condition, config):
    record = {'question_id': sample.question_id, 'video_id': sample.video_id,
              'category': sample.category, 'model_name': config['model'],
              'condition': condition, 'status': 'pending', 'raw_response': None,
              'predicted_answer': None, 'correct': False, 'history': []}
    messages = qa_messages(sample, condition)
    for _ in range(config['parse_retries'] + 1):
        try:
            response, attempts = call_model(model, messages)
        except CallFailure as exc:
            record['history'].append({'attempts': exc.attempts, 'error_type': type(exc.cause).__name__,
                                      'error': str(exc)})
            record['status'] = 'call_error'
            return record
        record['raw_response'] = response.text
        entry = {**response_record(response), 'attempts': attempts}
        try:
            answer = parse_answer_response(response.text)
        except ParseError as exc:
            entry['error'] = str(exc)
            record['history'].append(entry)
            record['status'] = 'parse_error'
            continue
        record['history'].append(entry)
        record['status'] = 'ok'
        record['predicted_answer'] = answer
        record['correct'] = answer == sample.answer
        return record
    return record


def qa_score(sample, record):
    return {'question_id': sample.question_id, 'category': sample.category,
            'condition': record['condition'], 'status': record['status'],
            'accqa': float(record['status'] == 'ok' and record['predicted_answer'] == sample.answer)}


def aggregate_qa(rows):
    count = len(rows)
    return {'count': count, 'correct': sum(row['accqa'] == 1.0 for row in rows),
            'accqa': sum(row['accqa'] for row in rows) / count if count else None,
            'status_counts': {state: sum(row['status'] == state for row in rows)
                              for state in ('ok', 'call_error', 'parse_error')}}


def summarize_qa(scores):
    conditions = {condition: aggregate_qa([row for row in scores if row['condition'] == condition])
                  for condition in QA_CONDITIONS}
    accuracies = {condition: result['accqa'] for condition, result in conditions.items()}
    return {'by_condition': conditions,
            'by_category': {category: {condition: aggregate_qa([
                row for row in scores if row['category'] == category and row['condition'] == condition])
                for condition in QA_CONDITIONS}
                for category in sorted({row['category'] for row in scores})},
            'accuracy_deltas': {
                'temporal_minus_baseline': accuracies['temporal_gt'] - accuracies['baseline'],
                'spatiotemporal_minus_temporal': accuracies['spatiotemporal_gt'] - accuracies['temporal_gt']},
            'denominator': 'selected_questions separately for each condition; call/parse failures score zero',
            'hint_protocol': 'same full native video; timestamp and pixel-xywh hints supplied as text'}


def write_qa_outputs(output, records, scores):
    atomic_jsonl(output / 'scores.jsonl', scores)

    groups = []
    for category in sorted({record['category'] for record in records}):
        category_scores = [row for row in scores if row['category'] == category]
        atomic_json(output / category / 'summary.json', summarize_qa(category_scores))

        for condition in QA_CONDITIONS:
            relative = Path(category) / condition
            directory = output / relative
            predictions = [
                row for row in records
                if row['category'] == category and row['condition'] == condition
            ]
            condition_scores = [
                row for row in category_scores if row['condition'] == condition
            ]

            atomic_jsonl(directory / 'predictions.jsonl', predictions)
            atomic_jsonl(directory / 'scores.jsonl', condition_scores)
            atomic_json(directory / 'summary.json', {
                'category': category,
                'condition': condition,
                **aggregate_qa(condition_scores),
            })

            groups.append({
                'category': category,
                'condition': condition,
                'count': len(predictions),
                'predictions': (relative / 'predictions.jsonl').as_posix(),
                'scores': (relative / 'scores.jsonl').as_posix(),
                'summary': (relative / 'summary.json').as_posix(),
            })

    atomic_json(output / 'runtime' / 'result_index.json', {
        'layout_version': 1,
        'task': 'progressive_qa',
        'groups': groups,
    })


def evaluate_progressive(config, *, retry_failed=False, model=None, score_only=False):
    samples = load_samples(config)
    identity = run_identity(config, samples, QA_PROMPT_VERSION)

    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    state_dir = output / 'runtime'

    lock = output / '.run.lock'
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(
            f'Run is locked: {lock}. Remove a stale lock after confirming no writer is active.'
        ) from exc

    records, scores = [], []
    try:
        ensure_run(state_dir, identity, score_only=score_only)

        for sample in samples:
            for condition in QA_CONDITIONS:
                record = load_checkpoint(state_dir, sample.question_id, condition)

                if score_only:
                    if record is None or record.get('status') == 'pending':
                        raise ValueError(
                            f'{sample.question_id}/{condition}: missing completed prediction'
                        )
                elif record is None or (retry_failed and record['status'] != 'ok'):
                    if model is None:
                        model = get_model(config['model'], config_path=config['model_config'])
                    record = predict_answer(model, sample, condition, config)
                    save_checkpoint(state_dir, record)

                record['correct'] = (
                    record['status'] == 'ok'
                    and record['predicted_answer'] == sample.answer
                )
                records.append(record)
                scores.append(qa_score(sample, record))
                print(
                    f'{len(records)}/{len(samples) * len(QA_CONDITIONS)} '
                    f'{sample.question_id}/{condition}: {record["status"]}',
                    flush=True,
                )

        write_qa_outputs(output, records, scores)
    finally:
        lock.rmdir()

    return scores
