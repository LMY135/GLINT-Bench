"""Task instructions and messages owned by Evaluation."""
import json
from glint.models import Text, Video, Image

PROMPT_VERSION = 'glint-eval-v1'


def video_messages(sample):
    prompt = (
        'Watch the attached benchmark video. Answer the multiple-choice question and '
        'locate the single temporal interval containing the visual evidence. '
        'Use seconds relative to the beginning of this attached clip. '
        f'The clip duration is {sample.metadata["duration"]} seconds.\n'
        f'Question: {sample.question}\n'
        'Return only JSON with this schema: '
        '{"answer":"A", "evidence_windows":[{"start":0.1,"end":0.5}]}. '
        'Use the selected option letter and your predicted interval.'
    )
    return [{'role': 'user', 'content': [Text(prompt), Video(str(sample.video_path))]}]


def spatial_messages(sample, prediction, frames):
    prompt = (
        f'Question: {sample.question}\n'
        f'Previous video prediction: {json.dumps(prediction, ensure_ascii=False)}\n'
        'For each attached frame, locate the visual target supporting the answer. '
        'Return only JSON: {"evidence_boxes":[{"frame":0,"box":[x,y,width,height]}]}. '
        'Use the provided frame identifiers and absolute pixel coordinates, '
        'with origin at the top left of the original image. '
        f'Image dimensions: {sample.metadata["width"]} by {sample.metadata["height"]} pixels.'
    )
    content = [Text(prompt)]
    for entry in frames:
        content.extend([Text(f'Frame {entry["frame"]}, time {entry["time"]} seconds'),
                        Image(entry['path'])])
    return [{'role': 'user', 'content': content}]


QA_CONDITIONS = ('baseline', 'temporal_gt', 'spatiotemporal_gt')


def qa_messages(sample, condition):
    """Independent full-video QA requests with progressively supplied evidence hints."""
    if condition not in QA_CONDITIONS:
        raise ValueError(f'Unknown QA condition: {condition}')
    prompt = f'Watch the attached video and answer the question.\nQuestion: {sample.question}\n'
    if condition in ('temporal_gt', 'spatiotemporal_gt'):
        prompt += (
            'Supporting visual evidence is located in these time intervals, '
            'in seconds from the beginning of the attached video:\n'
            f'{json.dumps(sample.evidence_windows)}\n'
        )
    if condition == 'spatiotemporal_gt':
        prompt += (
            'The supporting visual regions at annotated frames are listed below. '
            'Frame numbers are zero-based. Each box is [x, y, width, height] '
            'in original-image pixels, with origin at the top left. '
            f'The video dimensions are {sample.metadata["width"]} by '
            f'{sample.metadata["height"]} pixels.\n'
            f'{json.dumps(sample.evidence_boxes)}\n'
        )
    prompt += 'Return only the selected option letter (A, B, C, or D).'
    return [{'role': 'user', 'content': [Text(prompt), Video(str(sample.video_path))]}]
