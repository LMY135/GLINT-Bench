"""Extract original-resolution frames by zero-based annotation frame index."""
import json
import subprocess
from pathlib import Path


def run_media(command, *, timeout, input_bytes=None):
    """Keep media bytes intact and decode diagnostic output explicitly."""
    try:
        result = subprocess.run(command, input=input_bytes, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stderr = (exc.stderr or b'').decode('utf-8', errors='replace')
        raise ValueError(f'{command[0]} timed out after {timeout}s: {stderr[-4000:]}') from exc
    if result.returncode:
        stderr = result.stderr.decode('utf-8', errors='replace')
        raise ValueError(f'{command[0]} failed (exit {result.returncode}): {stderr[-4000:]}')
    return result.stdout


def sample_frames(sample, intersection, k, output_dir):
    start, end = intersection
    candidates = sorted((entry for entry in sample.evidence_boxes if start <= entry['time'] <= end),
                        key=lambda entry: entry['frame'])
    if not candidates:
        raise ValueError('Temporal intersection contains no annotated frames')
    count = min(k, len(candidates))
    indices = [len(candidates) // 2] if count == 1 else [round(i * (len(candidates) - 1) / (count - 1)) for i in range(count)]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in indices:
        entry = candidates[index]
        path = output / f'frame_{entry["frame"]:08d}.jpg'
        # Output JPEG through stdout so FFmpeg never opens a Windows image path.
        image = run_media([
            'ffmpeg', '-nostdin', '-v', 'error', '-filter_threads', '1',
            '-noautorotate', '-i', str(sample.video_path),
            '-vf', f"select='eq(n,{entry['frame']})'", '-frames:v', '1',
            '-c:v', 'mjpeg', '-threads:v', '1', '-q:v', '2',
            '-f', 'image2pipe', 'pipe:1',
        ], timeout=120)
        if not image:
            raise ValueError(f'No decoded image for frame {entry["frame"]}')
        probe = run_media([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height', '-of', 'json', '-i', 'pipe:0',
        ], timeout=30, input_bytes=image)
        streams = json.loads(probe).get('streams', [])
        if not streams:
            raise ValueError(f'No decoded image stream for frame {entry["frame"]}')
        size = streams[0]
        if (size['width'], size['height']) != (sample.metadata['width'], sample.metadata['height']):
            raise ValueError('Decoded image dimensions disagree with video metadata')
        path.write_bytes(image)
        frames.append({'frame': entry['frame'], 'time': entry['time'], 'path': str(path)})
    return frames
