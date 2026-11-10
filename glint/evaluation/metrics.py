"""Metrics from results/README.md; all temporal values use seconds."""
import math


def tnwd(predicted, truth, c):
    pc = (predicted['start'] + predicted['end']) / 2
    gc = (truth['start'] + truth['end']) / 2
    pl = predicted['end'] - predicted['start']
    gl = truth['end'] - truth['start']
    distance = math.hypot(pc - gc, (pl - gl) / 2)
    return math.exp(-distance / c)


def temporal_gate(predicted, truth, c, tau):
    if not predicted:
        return False, None
    score = tnwd(predicted[0], truth[0], c)
    start = max(predicted[0]['start'], truth[0]['start'])
    end = min(predicted[0]['end'], truth[0]['end'])
    return score >= tau and start < end, (start, end) if start < end else None


def bbox_iou(predicted, truth):
    x, y, w, h = predicted
    gx, gy, gw, gh = truth
    intersection = max(0, min(x + w, gx + gw) - max(x, gx)) * max(0, min(y + h, gy + gh) - max(y, gy))
    union = w * h + gw * gh - intersection
    return min(1.0, max(0.0, intersection / union)) if union > 0 else 0.0


def score_prediction(sample, record, c, tau):
    prediction = record.get('video', {}).get('prediction', {})
    windows = prediction.get('evidence_windows', [])
    temporal = tnwd(windows[0], sample.evidence_windows[0], c) if windows else 0.0
    gate, _ = temporal_gate(windows, sample.evidence_windows, c, tau)
    frames = record.get('frames', [])
    gt = {entry['frame']: entry['box'] for entry in sample.evidence_boxes}
    boxes = {entry['frame']: entry['box'] for entry in record.get('spatial', {}).get('prediction', [])}
    siou = None
    if gate and frames and record.get('spatial', {}).get('status') == 'ok':
        siou = sum(bbox_iou(boxes[f['frame']], gt[f['frame']]) if f['frame'] in boxes else 0.0
                   for f in frames) / len(frames)
    return {'question_id': sample.question_id, 'category': sample.category,
            'accqa': float(prediction.get('answer') == sample.answer), 'tnwd': temporal,
            'gate_passed': gate, 'siou': siou, 'tg_siou': siou if siou is not None else 0.0,
            'sampled_frames': len(frames), 'status': record.get('status', 'missing')}
