"""Join current QA schema with local benchmark clips and metadata."""
import json
import math
from pathlib import Path
from .types import Sample
from .config import VALID_CATEGORIES, normalize_selection


def read_jsonl(path):
    records = []
    with Path(path).open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_no}: invalid JSON') from exc
            if not isinstance(item, dict):
                raise ValueError(f'{path}:{line_no}: expected an object')
            records.append(item)
    return records


def number(value):
    return type(value) in (float, int) and math.isfinite(value)


def validate_window(window, duration):
    if not isinstance(window, dict):
        raise ValueError('Expected a temporal window object')
    start, end = window.get('start'), window.get('end')
    if not number(start) or not number(end) or not 0 <= start < end <= duration:
        raise ValueError(f'Invalid temporal window: {window}')


def validate_box(box, width, height):
    if not isinstance(box, list) or len(box) != 4 or not all(number(v) for v in box):
        raise ValueError('Expected four finite xywh coordinates')
    x, y, w, h = box
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width + 0.01 or y + h > height + 0.01:
        raise ValueError('Bounding box is outside original image dimensions')


def load_samples(config):
    selection = normalize_selection(config)
    categories = selection['categories']
    records = read_jsonl(config['qa_path'])
    for option, field in (('video_ids', 'video_id'), ('question_ids', 'question_id')):
        available = {item.get(field) for item in records}
        missing = set(selection[option]) - available
        if missing:
            raise ValueError(f'Unknown {option} in qa.jsonl: {sorted(missing)}')
    if categories:
        records = [item for item in records if item.get('category') in categories]
    for option, field in (('video_ids', 'video_id'), ('question_ids', 'question_id')):
        if selection[option]:
            records = [item for item in records if item.get(field) in selection[option]]
    if selection['video_limit'] is not None:
        # dict preserves first-occurrence order, including nonadjacent questions for a video.
        first_videos = list(dict.fromkeys(item['video_id'] for item in records))[:selection['video_limit']]
        records = [item for item in records if item['video_id'] in first_videos]
    if selection['limit'] is not None:
        records = records[:selection['limit']]
    if not records:
        raise ValueError(f'No QA samples matched categories/IDs/limits: {selection}')
    selected_videos = {item['video_id'] for item in records}
    metadata = {}
    for item in read_jsonl(config['metadata_path']):
        video_id = item.get('video_id')
        if video_id not in selected_videos:
            continue
        if video_id in metadata:
            raise ValueError(f'Duplicate video metadata: {video_id}')
        for key in ('width', 'height', 'fps', 'duration', 'num_frames'):
            if not number(item.get(key)) or item[key] <= 0:
                raise ValueError(f'Invalid metadata {key} for {video_id}')
        metadata[video_id] = item
    root = Path(config['videos_dir']).resolve()
    samples, seen = [], set()
    for item in records:
        qid, vid, category = item['question_id'], item['video_id'], item['category']
        if not all(isinstance(v, str) and v for v in (qid, vid, category)):
            raise ValueError('Question ID, video ID and category must be nonempty strings')
        if category not in VALID_CATEGORIES:
            raise ValueError(f'{qid}: unsupported category: {category}')
        if qid in seen:
            raise ValueError(f'Duplicate question_id: {qid}')
        seen.add(qid)
        if item.get('question_type') != 'mcq' or item.get('answer') not in ('A', 'B', 'C', 'D'):
            raise ValueError(f'{qid}: expected mcq with answer A/B/C/D')
        if not isinstance(item.get('question'), str) or not item['question'].strip():
            raise ValueError(f'{qid}: missing question')
        if vid not in metadata:
            raise ValueError(f'{qid}: missing metadata for {vid}')
        # Same layout as scripts/prepare_videos.py. Never select a raw source clip.
        video = (root / category / f'{vid}.mp4').resolve()
        if not video.is_relative_to(root):
            raise ValueError(f'{qid}: video path escapes videos_dir')
        if not video.is_file():
            raise FileNotFoundError(f'{qid}: benchmark video not found: {video}')
        meta = metadata[vid]
        windows, boxes = item['evidence_windows'], item['evidence_boxes']
        if not isinstance(windows, list) or len(windows) != 1:
            raise ValueError(f'{qid}: this protocol requires exactly one evidence window')
        validate_window(windows[0], meta['duration'])
        if not isinstance(boxes, list) or not boxes:
            raise ValueError(f'{qid}: missing evidence_boxes')
        frames = set()
        for entry in boxes:
            frame, time = entry['frame'], entry['time']
            if type(frame) is not int or not 0 <= frame < meta['num_frames'] or frame in frames:
                raise ValueError(f'{qid}: invalid or duplicate frame {frame}')
            frames.add(frame)
            if not number(time) or not windows[0]['start'] <= time <= windows[0]['end']:
                raise ValueError(f'{qid}: evidence frame time outside GT interval')
            validate_box(entry['box'], meta['width'], meta['height'])
        samples.append(Sample(qid, vid, category, item['question'], item['answer'],
                              windows, boxes, meta, video))
    if not samples:
        raise ValueError('QA dataset is empty')
    return samples
