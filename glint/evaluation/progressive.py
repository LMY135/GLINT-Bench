"""Three independent answer-only requests per question, one record per condition."""
from pathlib import Path
from glint.models import get_model
from .calling import call_model, response_record
from .dataset import load_samples
from .parsing import parse_answer_response
from .prompts import QA_CONDITIONS, qa_messages
from .results import (run_identity, ensure_run, load_checkpoint, save_checkpoint,
                      atomic_json, atomic_jsonl, save_thought, export_thoughts)
from .types import CallFailure, ParseError

QA_PROMPT_VERSION = 'glint-progressive-qa-structured-v2'


def predict_answer(model, sample, condition, config, state_dir=None):
    record = {'question_id': sample.question_id, 'video_id': sample.video_id,
              'category': sample.category, 'model_name': config['model'],
              'condition': condition, 'status': 'pending', 'raw_response': None,
              'predicted_answer': None, 'correct': False, 'history': []}
    messages = qa_messages(sample, condition)
    # Short answer-only outputs may be regenerated after a parse failure.
    # Always allow one retry, even when general parse_retries is zero.
    for _ in range(max(1, config['parse_retries']) + 1):
        try:
            response, attempts = call_model(model, messages, stage='qa')
        except CallFailure as exc:
            record['history'].append({'attempts': exc.attempts, 'error_type': type(exc.cause).__name__,
                                      'error': str(exc)})
            record['status'] = 'call_error'
            return record
        if state_dir is not None:
            save_thought(state_dir, sample.question_id, condition, response)
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


def write_qa_outputs(output, records, scores, *, state_dir=None, write_thoughts=True):
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
            if write_thoughts and state_dir is not None:
                export_thoughts(state_dir, [r['question_id'] for r in predictions],
                                (condition,), directory / 'thoughts.jsonl')
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
                    record = predict_answer(model, sample, condition, config, state_dir)
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

        write_qa_outputs(output, records, scores, state_dir=state_dir,
                         write_thoughts=not score_only)
    finally:
        lock.rmdir()

    return scores


def recover_qa_parse_errors(config):
    """Reparse existing exported QA JSONL without invoking evaluation or model APIs.

    Preserve original outputs/checkpoints. Only a parse_error with an unambiguous
    saved final answer is recovered. Existing scores remain the source of truth
    for every other record.
    """
    import json
    from .dataset import read_jsonl

    output = Path(config['output_dir'])
    if not output.is_dir():
        raise FileNotFoundError(f'Missing original results directory: {output}')
    if (output / '.run.lock').exists():
        raise RuntimeError(f'Run is locked: {output / ".run.lock"}')

    # No dataset loading, video access, prompt version or run-identity checks:
    # the source of truth is the exported predictions and scores.
    groups = []
    for prediction_path in sorted(output.glob('*/**/predictions.jsonl')):
        relative = prediction_path.relative_to(output)
        if len(relative.parts) != 3:
            continue
        category, condition, _ = relative.parts
        if condition not in QA_CONDITIONS:
            continue
        if config.get('categories') and category not in config['categories']:
            continue
        score_path = prediction_path.with_name('scores.jsonl')
        if not score_path.is_file():
            raise FileNotFoundError(f'Missing original QA scores: {score_path}')
        predictions = read_jsonl(prediction_path)
        scores = read_jsonl(score_path)
        if len(predictions) != len(scores):
            raise ValueError(f'Prediction/score count mismatch: {prediction_path}')
        by_id = {row['question_id']: row for row in scores}
        if len(by_id) != len(scores):
            raise ValueError(f'Duplicate score question IDs: {score_path}')
        if {row['question_id'] for row in predictions} != set(by_id):
            raise ValueError(f'Prediction/score question IDs mismatch: {prediction_path}')
        for row in predictions:
            if row.get('category') != category or row.get('condition') != condition:
                raise ValueError(f'Unexpected category or condition: {prediction_path}')
            score = by_id[row['question_id']]
            if score.get('category') != category or score.get('condition') != condition:
                raise ValueError(f'Unexpected score category or condition: {score_path}')
        groups.append((category, condition, predictions, by_id))

    if not groups:
        raise ValueError(f'No original Progressive QA predictions found in {output}')

    # Current QA is consulted *only* for answers to newly recovered records.
    # Existing exported scores and their correctness are preserved as-is.
    qa_records = read_jsonl(config['qa_path'])
    qa_by_id = {}
    for row in qa_records:
        qid = row.get('question_id')
        if qid in qa_by_id:
            raise ValueError(f'Duplicate question ID in QA file: {qid}')
        qa_by_id[qid] = row

    recovered = 0
    pending = []
    original_scores, recovered_scores, recovered_records = [], [], []
    for category, condition, predictions, by_id in groups:
        for source in predictions:
            record = dict(source)
            score = dict(by_id[source['question_id']])
            original_scores.append(dict(score))
            if score.get('status') != record.get('status'):
                raise ValueError(f'Original prediction/score status mismatch: {source["question_id"]}/{condition}')
            if record['status'] == 'parse_error' and isinstance(record.get('raw_response'), str):
                try:
                    answer = parse_answer_response(record['raw_response'])
                except ParseError:
                    pass
                else:
                    qa = qa_by_id.get(record['question_id'])
                    # Only score a recoverable answer with a valid matching QA entry.
                    if qa is None or qa.get('category') != category or qa.get('answer') not in ('A', 'B', 'C', 'D'):
                        pending.append(f'{category}/{record["question_id"]}')
                    else:
                        record['status'] = 'ok'
                        record['predicted_answer'] = answer
                        record['correct'] = answer == qa['answer']
                        record['recovered_from_saved_response'] = True
                        score['status'] = 'ok'
                        score['accqa'] = float(record['correct'])
                        recovered += 1
            recovered_records.append(record)
            recovered_scores.append(score)

    if pending:
        raise ValueError(
            'Cannot safely score recoverable answers without matching valid QA ground truth: '
            + ', '.join(sorted(set(pending))[:20])
        )

    destination = output / 'recovered'
    write_qa_outputs(destination, recovered_records, recovered_scores, write_thoughts=False)
    atomic_json(destination / 'recovery_summary.json', {
        'source_run': str(output),
        'method': 'offline_reparse_exported_predictions',
        'recovered_count': recovered,
        'original': summarize_qa(original_scores),
        'recovered': summarize_qa(recovered_scores),
        'scoring_note': (
            'Existing scores are preserved. Newly recovered answers are graded '
            'against matching question IDs in the current QA file; confirm the '
            'ground-truth answers are unchanged from the original run.'
        ),
        'original_files_unchanged': True,
    })
    print(f'Recovered {recovered} QA parse_error records into {destination}; no API calls.')
    return recovered_scores
