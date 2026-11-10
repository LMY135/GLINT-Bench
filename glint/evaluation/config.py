"""Evaluation paths are relative to project_root, never the current directory."""
import math
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs' / 'evaluation.yaml'
VALID_CATEGORIES = ('sports', 'traffic', 'crowd', 'game', 'magic')


def normalize_categories(value):
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError('categories must be a list, for example [magic, game]')
    unknown = set(value) - set(VALID_CATEGORIES)
    if unknown:
        raise ValueError(f'Unknown categories: {sorted(unknown)}')
    return sorted(set(value))


def normalize_selection(config):
    selection = {'categories': normalize_categories(config.get('categories', []))}
    for key in ('video_ids', 'question_ids'):
        value = config.get(key, [])
        if value is None:
            value = []
        if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError(f'{key} must be a list of nonempty IDs')
        selection[key] = sorted({item.strip() for item in value})
    for key in ('limit', 'video_limit'):
        value = config.get(key)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f'{key} must be a positive integer or null')
        selection[key] = value
    return selection


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('Evaluation config must be a mapping')
    allowed = {'project_root', 'task', 'model', 'model_config', 'qa_path', 'metadata_path',
               'videos_dir', 'output_dir', 'categories', 'video_ids', 'question_ids',
               'limit', 'video_limit', 'k', 'c', 'tau', 'parse_retries'}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f'Unexpected evaluation settings: {sorted(unknown)}; model settings belong in models.yaml')
    data = dict(data)
    data.update(normalize_selection(data))
    root = Path(data.get('project_root', ROOT))
    if not root.is_absolute():
        root = path.parent / root
    root = root.resolve()
    defaults = {'qa_path': 'data/qa.jsonl', 'metadata_path': 'data/video_metadata.jsonl',
                'videos_dir': 'videos', 'model_config': 'configs/models.yaml',
                'output_dir': 'results/evaluation'}
    for key, default in defaults.items():
        value = Path(data.get(key, default))
        data[key] = str((root / value).resolve())
    if not isinstance(data.get('model'), str) or not data['model']:
        raise ValueError('Set model to an entry in configs/models.yaml')
    data.setdefault('task', 'progressive_qa')
    if data['task'] not in ('progressive_qa', 'grounding'):
        raise ValueError('task must be progressive_qa or grounding')
    data.setdefault('k', 5)
    data.setdefault('c', -0.5 / math.log(0.5))
    data.setdefault('tau', 0.5)
    if type(data['k']) is not int or data['k'] < 1:
        raise ValueError('k must be a positive integer')
    for key in ('c', 'tau'):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite')
    if data['c'] <= 0 or not 0 <= data['tau'] <= 1:
        raise ValueError('Require c > 0 and 0 <= tau <= 1')
    data.setdefault('parse_retries', 0)
    if type(data['parse_retries']) is not int or data['parse_retries'] < 0:
        raise ValueError('parse_retries must be a nonnegative integer')
    data['project_root'] = str(root)
    return data
