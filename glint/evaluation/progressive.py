"""Three independent answer-only requests per question, one record per condition."""
from pathlib import Path
from glint.models import get_model
from .calling import call_model, response_record
from .dataset import load_samples
from .parsing import parse_answer_response, parse_answer_with_method
from .prompts import QA_CONDITIONS, qa_messages
from .results import (run_identity, ensure_run, load_checkpoint, save_checkpoint,
                      atomic_json, atomic_jsonl, save_thought, export_thoughts)
from .types import CallFailure, ParseError

QA_PROMPT_VERSION = 'glint-progressive-qa-system-user-v3'


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
    parse_errors_before = 0
    remaining_parse_errors = 0
    recovery_methods = {}
    unrecovered = []
    pending = []
    original_scores, recovered_scores, recovered_records = [], [], []
    for category, condition, predictions, by_id in groups:
        for source in predictions:
            record = dict(source)
            score = dict(by_id[source['question_id']])
            original_scores.append(dict(score))
            if score.get('status') != record.get('status'):
                raise ValueError(f'Original prediction/score status mismatch: {source["question_id"]}/{condition}')
            if record['status'] == 'parse_error':
                parse_errors_before += 1
                try:
                    answer, method = parse_answer_with_method(record.get('raw_response'))
                except ParseError as exc:
                    remaining_parse_errors += 1
                    unrecovered.append({
                        'question_id': record['question_id'],
                        'category': category,
                        'condition': condition,
                        'reason': str(exc),
                    })
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
                        record['recovery_method'] = method
                        recovery_methods[method] = recovery_methods.get(method, 0) + 1
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
        'original_parse_errors': parse_errors_before,
        'remaining_parse_errors': remaining_parse_errors,
        'recovery_methods': recovery_methods,
        'unrecovered': unrecovered,
        'original': summarize_qa(original_scores),
        'recovered': summarize_qa(recovered_scores),
        'scoring_note': (
            'Existing scores are preserved. Newly recovered answers are graded '
            'against matching question IDs in the current QA file; confirm the '
            'ground-truth answers are unchanged from the original run.'
        ),
        'original_files_unchanged': True,
    })
    print(
        f'QA parse errors: {parse_errors_before}; recovered: {recovered}; '
        f'remaining: {remaining_parse_errors}. '
        f'Output: {destination}; no API calls.'
    )
    return recovered_scores


def retry_historical_call_errors(config, *, model=None):
    """Retry only saved call_error QA predictions into a separate result tree.

    Reads the original exported predictions/scores and optional checkpoints, never
    calls ensure_run, and never changes the original run. New requests are
    checkpointed in <original>/retried_call_errors/runtime/ for safe resume.
    """
    from .dataset import read_jsonl

    source = Path(config['output_dir'])
    destination = source / 'retried_call_errors'
    state_dir = destination / 'runtime'
    if not source.is_dir():
        raise FileNotFoundError(f'Original QA result directory not found: {source}')
    samples = load_samples(config)
    sample_by_id = {sample.question_id: sample for sample in samples}
    original_records, original_scores, retry_targets = [], [], []
    for category in sorted({sample.category for sample in samples}):
        for condition in QA_CONDITIONS:
            directory = source / category / condition
            prediction_path = directory / 'predictions.jsonl'
            score_path = directory / 'scores.jsonl'
            if not prediction_path.is_file() or not score_path.is_file():
                raise FileNotFoundError(f'Missing historical QA exports: {directory}')
            records = read_jsonl(prediction_path)
            old_scores = read_jsonl(score_path)
            scores_by_id = {row['question_id']: row for row in old_scores}
            if len(scores_by_id) != len(old_scores) or len(records) != len(old_scores):
                raise ValueError(f'Invalid historical QA scores in {directory}')
            seen = set()
            for record in records:
                qid = record['question_id']
                if qid in seen or qid not in scores_by_id:
                    raise ValueError(f'Duplicate or missing historical score: {qid}/{condition}')
                seen.add(qid)
                if qid not in sample_by_id:
                    raise ValueError(f'{qid}: missing from current QA; restore historical question before retry')
                sample = sample_by_id[qid]
                if record.get('category') != category or record.get('condition') != condition or sample.category != category:
                    raise ValueError(f'{qid}: category/condition mismatch')
                score = scores_by_id[qid]
                if score.get('category') != category or score.get('condition') != condition or score.get('status') != record.get('status'):
                    raise ValueError(f'{qid}: historical score mismatch')
                original_records.append(record)
                original_scores.append(score)
                if record['status'] == 'call_error':
                    retry_targets.append((qid, condition))
            if seen != set(scores_by_id):
                raise ValueError(f'Unmatched historical scores in {directory}')

    # The full original selection must match the current QA selection. Global
    # file hashes may change for unrelated added questions, but we must never
    # silently drop an old result or add a new question to the historical run.
    expected = {(sample.question_id, condition) for sample in samples for condition in QA_CONDITIONS}
    actual = {(record['question_id'], record['condition']) for record in original_records}
    if expected != actual:
        raise ValueError(
            'Historical QA question IDs differ from the selected current dataset; '
            'use the original question selection before retrying'
        )
    if not retry_targets:
        print('No historical call_error records to retry.')
        return original_scores

    if (source / '.run.lock').exists():
        raise RuntimeError(f'Original run is locked: {source / ".run.lock"}')
    if destination.exists() and not (state_dir / 'retry_manifest.json').exists():
        raise RuntimeError(
            f'{destination} exists without retry_manifest.json; '
            'refusing to overwrite unrelated files'
        )
    import json
    from .results import atomic_json
    manifest_path = state_dir / 'retry_manifest.json'
    manifest = {
        'source_run': str(source.resolve()),
        'model': config['model'],
        'retry_keys': [list(key) for key in retry_targets],
        'method': 'rerun_historical_call_errors_only',
    }
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Existing retry run manifest does not match; use another source directory')
    lock = destination / '.run.lock'
    destination.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f'Retry run is locked: {lock}') from exc

    try:
        if not manifest_path.is_file():
            atomic_json(manifest_path, manifest)
        latest_records, latest_scores = [], []
        count = 0
        for record, score in zip(original_records, original_scores):
            qid, condition = record['question_id'], record['condition']
            if record['status'] != 'call_error':
                latest_records.append(dict(record))
                latest_scores.append(dict(score))
                continue

            sample = sample_by_id[qid]
            retried = load_checkpoint(state_dir, qid, condition)
            if retried is None:
                if model is None:
                    model = get_model(config['model'], config_path=config['model_config'])
                retried = predict_answer(model, sample, condition, config, state_dir)
                save_checkpoint(state_dir, retried)
                count += 1
                print(f'Retried {count}/{len(retry_targets)} {qid}/{condition}: '
                      f'{retried["status"]}', flush=True)
            # Do not replay completed calls, including those that still failed.
            retry_score = qa_score(sample, retried)
            latest_records.append(retried)
            latest_scores.append(retry_score)

        write_qa_outputs(destination, latest_records, latest_scores,
                         state_dir=state_dir, write_thoughts=False)
        atomic_json(destination / 'retry_summary.json', {
            'source_run': str(source),
            'retry_targets': len(retry_targets),
            'new_calls': count,
            'original': summarize_qa(original_scores),
            'retried': summarize_qa(latest_scores),
            'note': 'Unretried records and their scores are unchanged. New scores use the current QA answer key; verify that historical questions and ground-truth answers have not changed.',
            'original_files_unchanged': True,
        })
    finally:
        lock.rmdir()

    print(f'Historical QA call errors retried into {destination}; '
          'original run unchanged.')
    return latest_scores
