"""Offline CS-IoU aggregation from saved grounding scores. No model or video needed."""
import argparse
import json
from pathlib import Path

from .metrics import conditional_spatial_iou
from .results import atomic_json


def read_scores(path):
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{path}:{number}: invalid JSON') from exc
        if not isinstance(row, dict) or 'gate_passed' not in row or 'tg_siou' not in row:
            raise ValueError(f'{path}:{number}: expected grounding scores with gate_passed and tg_siou')
        if type(row['gate_passed']) is not bool:
            raise ValueError(f'{path}:{number}: gate_passed must be a boolean')
        value = row['tg_siou']
        if type(value) not in (int, float) or not 0 <= value <= 1:
            raise ValueError(f'{path}:{number}: invalid TG-sIoU')
        rows.append(row)
    return rows


def calculate(input_path):
    """Read a grounding scores file or discover category score files in a run root."""
    input_path = Path(input_path)
    if input_path.is_file():
        paths = [input_path]
    elif input_path.is_dir():
        paths = sorted(input_path.rglob('grounding/scores.jsonl'))
        if not paths:
            raise FileNotFoundError(f'No grounding/scores.jsonl found under {input_path}')
    else:
        raise FileNotFoundError(input_path)
    rows = [row for path in paths for row in read_scores(path)]
    if not rows:
        raise ValueError('No grounding scores found')
    identifiers = [(row.get('category'), row.get('question_id')) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate category/question_id in grounding scores')

    def aggregate(group):
        passed = sum(row['gate_passed'] for row in group)
        return {'count': len(group), 'gate_passed': passed,
                'cs_iou': conditional_spatial_iou(group)}

    return {'overall': aggregate(rows),
            'by_category': {category: aggregate([r for r in rows if r.get('category') == category])
                            for category in sorted({r.get('category') for r in rows})},
            'definition': 'Mean TG-sIoU among gate-passed samples; spatial failures score zero',
            'source_files': [str(path) for path in paths]}


def main():
    parser = argparse.ArgumentParser(description='Calculate CS-IoU offline from saved grounding scores')
    parser.add_argument('--input', required=True, help='scores.jsonl or run directory containing category/grounding/scores.jsonl')
    parser.add_argument('--output', help='Optional JSON summary path (source results remain unchanged)')
    args = parser.parse_args()
    result = calculate(args.input)
    if args.output:
        atomic_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
