"""Task instructions and messages owned by Evaluation."""
import json
from glint.models import Text, Video, Image

PROMPT_VERSION = 'glint-eval-system-user-v3'


def _system_message(instruction):
    return {'role': 'system', 'content': [Text(instruction)]}



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
    return [
        _system_message('You are a video evidence grounding assistant. Answer the multiple-choice question and locate one temporal evidence interval. Use seconds relative to the beginning of the clip. Return only valid JSON with the requested answer and evidence_windows fields; do not add explanations.'),
        {'role': 'user', 'content': [Text(prompt), Video(str(sample.video_path))]},
    ]


def spatial_messages(sample, prediction, frames):
    frame_ids = [entry['frame'] for entry in frames]
    prompt = (
        f'Question: {sample.question}\n'
        f'Previous video prediction: {json.dumps(prediction, ensure_ascii=False)}\n'
        'Locate the ONE visual target supporting the answer in EACH provided frame. '
        'Return exactly one complete JSON object, without Markdown fences or explanations, '
        'using this schema: {"evidence_boxes":[{"frame":0,"box":[x,y,width,height]}]}. '
        f'Use exactly these supplied frame IDs: {json.dumps(frame_ids)}. '
        'Each frame ID must appear once, with exactly one bounding box. '
        'Coordinates are absolute pixels in the ORIGINAL image, with top-left origin. '
        'The box format is [x, y, width, height], NOT [x1, y1, x2, y2]; '
        'width and height are positive extents. Do not output labels or other fields. '
        f'Image dimensions: {sample.metadata["width"]} by {sample.metadata["height"]} pixels.'
    )
    content = [Text(prompt)]
    for entry in frames:
        content.extend([Text(f'Frame {entry["frame"]}, time {entry["time"]} seconds'),
                        Image(entry['path'])])
    return [
        _system_message('You are a visual evidence grounding assistant. Locate exactly one visual target in each supplied frame. Return one bounding box per frame, in original-image absolute pixel coordinates using [x, y, width, height] with a top-left origin and positive extents. Return only valid JSON with the evidence_boxes field; do not add Markdown, explanations, labels, or extra fields.'),
        {'role': 'user', 'content': content},
    ]


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
    return [
        _system_message('You are a video question-answering assistant. Answer the multiple-choice question using the provided video and any supplied evidence hints. Return only one selected option letter (A, B, C, or D), with no explanation.'),
        {'role': 'user', 'content': [Text(prompt), Video(str(sample.video_path))]},
    ]
