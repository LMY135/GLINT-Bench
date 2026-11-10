"""Run with python -m glint.evaluation."""
import argparse
from .config import DEFAULT_CONFIG, VALID_CATEGORIES, load_config, normalize_selection
from .dataset import load_samples
from .runner import evaluate


def main():
    parser = argparse.ArgumentParser(description='GLINT-Bench native-video evaluation')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--categories', nargs='+', choices=VALID_CATEGORIES,
                        help='Run only selected categories, e.g. magic or magic game')
    parser.add_argument('--video-ids', nargs='+', help='Select videos by video_id in qa.jsonl')
    parser.add_argument('--question-ids', nargs='+', help='Select questions by question_id')
    parser.add_argument('--limit', type=int, help='First N selected QA records in original file order')
    parser.add_argument('--video-limit', type=int, help='First N distinct selected videos in QA file order')
    parser.add_argument('--task', choices=['progressive_qa', 'grounding'])
    parser.add_argument('--model', help='Model name in configs/models.yaml')
    parser.add_argument('--videos-dir', help='Override benchmark clips root directory')
    parser.add_argument('--output-dir', help='Override result directory')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--validate-only', action='store_true')
    mode.add_argument('--score-only', action='store_true')
    mode.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    from pathlib import Path
    for argument in ('videos_dir', 'output_dir'):
        value = getattr(args, argument)
        if value:
            config[argument] = str(Path(value).resolve())
    for key in ('categories', 'video_ids', 'question_ids', 'limit', 'video_limit'):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    config.update(normalize_selection(config))
    if args.task:
        config['task'] = args.task
    if args.model:
        config['model'] = args.model
    if args.validate_only:
        samples = load_samples(config)
        print(f'Validated {len(samples)} questions, {len({s.video_id for s in samples})} local videos')
        for sample in samples:
            print(f'{sample.question_id} -> {sample.video_path}')
        return
    evaluate(config, retry_failed=args.retry_failed, score_only=args.score_only)


if __name__ == '__main__':
    main()
