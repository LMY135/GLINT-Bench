#!/usr/bin/env python3
"""Convert COCO annotations into validated GLINT-Bench QA JSONL samples.

The script reads qa.jsonl as the authoritative sample list and enriches it with
spatial evidence from the three Annotix COCO exports. It never changes source
files. Output is written only after every quality-control check passes.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# Required directory layout (DATASET_ROOT is the ball_video directory):
#
# ball_video/
# ├── convert_ball_annotations.py
# ├── qa.jsonl                         # One JSON object per QA; must contain
# │                                    # question_id, video_id, question, answer.
# │                                    # category and question_type are optional.
# ├── sport_qa.jsonl                    # Created by this script after QC passes.
# ├── video_id_mapping.json            # Created by this script: old ID -> new ID.
# ├── exported_annotations/
# │   ├── basketball_annotations.json  # COCO export: images + annotations
# │   ├── basketball_project.json      # Annotix export: videos[] UUID -> name
# │   ├── football_annotations.json
# │   ├── football_project.json
# │   ├── volleyball_annotations.json
# │   └── volleyball_project.json
# ├── basketball_1/                    # Folder name must equal a qa.jsonl video_id.
# │   └── *.meta.json                  # Exactly one file; it must contain valid fps.
# ├── football_1/
# │   └── *.meta.json
# └── volleyball_1/
#     └── *.meta.json
#
# The same <sport>_<number>/ + *.meta.json convention is required for every
# video_id in qa.jsonl. Project video names must use the matching video_id,
# for example basketball_1.mp4 <-> basketball_1/.
# Edit these three settings directly when running the conversion elsewhere.
SUBSETS = ("basketball", "football", "volleyball")
DATASET_ROOT = Path(__file__).resolve().parent
CUSTOMIZED_QA_FILE = "qa.jsonl"
OUTPUT_PATH = DATASET_ROOT / "sport_qa.jsonl"
VIDEO_ID_MAPPING_PATH = DATASET_ROOT / "video_id_mapping.json"
OVERWRITE_OUTPUT = False
REQUIRED_TEXT_FIELDS = (
    "question_id",
    "video_id",
    "category",
    "question_type",
    "question",
    "answer",
)
FRAME_NAME_RE = re.compile(
    r"_(?P<video_uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12})_frame_(?P<frame>\d+)\.jpg$",
    re.IGNORECASE,
)
VIDEO_ID_RE = re.compile(
    r"^(?P<sport>basketball|football|volleyball)_(?P<index>[1-9]\d*)$"
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read QA rows and fill the two permitted default fields."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object on {path}:{number}")
            row.setdefault("category", "ball")
            row.setdefault("question_type", "mcq")
            rows.append(row)
    return rows


def find_meta_fps(root: Path, video_id: str) -> float:
    meta_files = sorted((root / video_id).glob("*.meta.json"))
    if len(meta_files) != 1:
        raise ValueError(
            f"{video_id}: expected exactly one *.meta.json, found {len(meta_files)}"
        )
    meta = read_json(meta_files[0])
    fps = meta.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError(f"{meta_files[0]}: invalid fps value {fps!r}")
    return float(fps)


def parse_frame_name(file_name: str) -> tuple[str, int]:
    match = FRAME_NAME_RE.search(file_name)
    if not match:
        raise ValueError(
            "Cannot extract video UUID and frame index from COCO file_name: "
            f"{file_name!r}"
        )
    return match.group("video_uuid").lower(), int(match.group("frame"))


def load_coco_evidence(
    root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float], set[str]]:
    """Return COCO records, project FPS values, and video IDs seen in Annotix."""
    annotation_dir = root / "exported_annotations"
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    project_fps: dict[str, float] = {}
    annotix_video_ids: set[str] = set()

    for subset in SUBSETS:
        project_path = annotation_dir / f"{subset}_project.json"
        coco_path = annotation_dir / f"{subset}_annotations.json"
        project = read_json(project_path)
        coco = read_json(coco_path)

        uuid_to_video_id: dict[str, str] = {}
        for video in project.get("videos", []):
            video_uuid = video.get("id")
            video_name = video.get("name")
            fps = video.get("fpsOriginal")
            if not isinstance(video_uuid, str) or not isinstance(video_name, str):
                raise ValueError(f"{project_path}: malformed video entry {video!r}")
            if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
                raise ValueError(f"{project_path}: invalid fpsOriginal for {video_name}")
            video_id = Path(video_name).stem
            if video_uuid.lower() in uuid_to_video_id or video_id in project_fps:
                raise ValueError(f"{project_path}: duplicate project video {video_name}")
            uuid_to_video_id[video_uuid.lower()] = video_id
            project_fps[video_id] = float(fps)

        images_by_id: dict[Any, dict[str, Any]] = {}
        for image in coco.get("images", []):
            image_id = image.get("id")
            if image_id in images_by_id:
                raise ValueError(f"{coco_path}: duplicate COCO image id {image_id!r}")
            video_uuid, frame = parse_frame_name(str(image.get("file_name", "")))
            video_id = uuid_to_video_id.get(video_uuid)
            if video_id is None:
                raise ValueError(
                    f"{coco_path}: {image['file_name']!r} refers to a video UUID "
                    "absent from the project export"
                )
            width, height = image.get("width"), image.get("height")
            if (
                isinstance(width, bool)
                or isinstance(height, bool)
                or not isinstance(width, (int, float))
                or not isinstance(height, (int, float))
                or width <= 0
                or height <= 0
            ):
                raise ValueError(f"{coco_path}: invalid image dimensions for {image_id!r}")
            images_by_id[image_id] = {
                "video_id": video_id,
                "frame": frame,
                "width": float(width),
                "height": float(height),
            }
            annotix_video_ids.add(video_id)

        for annotation in coco.get("annotations", []):
            image = images_by_id.get(annotation.get("image_id"))
            if image is None:
                raise ValueError(
                    f"{coco_path}: annotation {annotation.get('id')!r} references "
                    f"unknown image {annotation.get('image_id')!r}"
                )
            bbox = annotation.get("bbox")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox)
            ):
                raise ValueError(f"{coco_path}: invalid bbox for image {annotation.get('image_id')!r}")
            by_video[image["video_id"]].append(
                {
                    "frame": image["frame"],
                    "box": [float(value) for value in bbox],
                    "width": image["width"],
                    "height": image["height"],
                }
            )
    return dict(by_video), project_fps, annotix_video_ids


def make_windows(records: list[dict[str, Any]], fps: float) -> list[dict[str, float]]:
    """Make inclusive time windows for each contiguous run of annotated frames."""
    frames = [record["frame"] for record in records]
    if not frames:
        return []
    windows: list[dict[str, float]] = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame != previous + 1:
            windows.append({"start": round(start / fps, 3), "end": round(previous / fps, 3)})
            start = frame
        previous = frame
    windows.append({"start": round(start / fps, 3), "end": round(previous / fps, 3)})
    return windows


def clip_box_to_image(
    box: list[float], image_width: float, image_height: float
) -> list[float]:
    """Clip a COCO [x, y, w, h] box to its image and recompute w/h."""
    x, y, width, height = box
    left = min(max(x, 0.0), image_width)
    top = min(max(y, 0.0), image_height)
    right = min(max(x + width, 0.0), image_width)
    bottom = min(max(y + height, 0.0), image_height)
    return [left, top, right - left, bottom - top]


def enrich_samples(
    samples: list[dict[str, Any]],
    evidence: dict[str, list[dict[str, Any]]],
    project_fps: dict[str, float],
    meta_fps: dict[str, float],
) -> list[dict[str, Any]]:
    """Construct output rows in memory; QC decides whether they may be written."""
    output: list[dict[str, Any]] = []

    for sample in samples:
        video_id = sample.get("video_id")
        records = sorted(evidence.get(video_id, []), key=lambda record: record["frame"])
        result = dict(sample)
        if not records:
            result["evidence_windows"] = []
            result["evidence_boxes"] = []
            output.append(result)
            continue

        fps = meta_fps[video_id]
        export_fps = project_fps.get(video_id)
        if export_fps is None:
            raise ValueError(f"{video_id}: absent from all project exports")
        if abs(fps - export_fps) > 1e-9:
            raise ValueError(
                f"{video_id}: meta FPS ({fps}) differs from project FPS ({export_fps})"
            )

        result["evidence_windows"] = make_windows(records, fps)
        result["evidence_boxes"] = [
            {
                "frame": record["frame"],
                "time": round(record["frame"] / fps, 3),
                "box": [
                    round(value, 2)
                    for value in clip_box_to_image(
                        record["box"], record["width"], record["height"]
                    )
                ],
            }
            for record in records
        ]
        output.append(result)
    return output


def run_qc(
    samples: list[dict[str, Any]],
    annotix_video_ids: set[str],
    source_evidence: dict[str, list[dict[str, Any]]],
    meta_fps: dict[str, float],
) -> list[str]:
    """Return every QC failure; an empty list means the output is safe to write."""
    errors: list[str] = []
    qa_by_video: dict[str, dict[str, Any]] = {}

    for index, sample in enumerate(samples, start=1):
        prefix = f"sample {index}"
        for field in REQUIRED_TEXT_FIELDS:
            value = sample.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix}: required non-empty text field {field!r} is missing or invalid")
        video_id = sample.get("video_id")
        if isinstance(video_id, str) and video_id.strip():
            if video_id in qa_by_video:
                errors.append(f"{prefix}: duplicate video_id {video_id!r} in {CUSTOMIZED_QA_FILE}")
            else:
                qa_by_video[video_id] = sample

    qa_video_ids = set(qa_by_video)
    for video_id in sorted(annotix_video_ids - qa_video_ids):
        errors.append(f"Annotix video_id {video_id!r} does not exist in {CUSTOMIZED_QA_FILE}")

    for video_id, sample in qa_by_video.items():
        prefix = video_id
        fps = meta_fps.get(video_id)
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
            errors.append(f"{prefix}: missing or invalid FPS in meta.json")

        boxes = sample.get("evidence_boxes")
        if not isinstance(boxes, list) or not boxes:
            errors.append(f"{prefix}: sample has no bbox")
            continue
        windows = sample.get("evidence_windows")
        if not isinstance(windows, list) or not windows:
            errors.append(f"{prefix}: sample has no evidence window")

        source_by_frame = {
            record["frame"]: record for record in source_evidence.get(video_id, [])
        }
        frames: list[int] = []
        previous_time: float | None = None
        for position, box_entry in enumerate(boxes, start=1):
            item_prefix = f"{prefix}: bbox {position}"
            if not isinstance(box_entry, dict):
                errors.append(f"{item_prefix} is not an object")
                continue
            frame = box_entry.get("frame")
            time = box_entry.get("time")
            box = box_entry.get("box")
            if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
                errors.append(f"{item_prefix} has an invalid frame")
                continue
            frames.append(frame)
            if isinstance(time, bool) or not isinstance(time, (int, float)):
                errors.append(f"{item_prefix} has an invalid timestamp")
            else:
                if previous_time is not None and time <= previous_time:
                    errors.append(f"{item_prefix} timestamp is not strictly increasing")
                previous_time = float(time)
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in box)
            ):
                errors.append(f"{item_prefix} has an invalid box")
                continue
            x, y, width, height = box
            if width <= 0 or height <= 0:
                errors.append(f"{item_prefix} must have w > 0 and h > 0")
            source = source_by_frame.get(frame)
            if source is None:
                errors.append(f"{item_prefix} has no corresponding source annotation")
            elif (
                x < 0
                or y < 0
                or x + width > source["width"] + 1e-9
                or y + height > source["height"] + 1e-9
            ):
                errors.append(
                    f"{item_prefix} is outside image bounds "
                    f"[0, {source['width']}] x [0, {source['height']}]"
                )

        if len(frames) != len(set(frames)):
            errors.append(f"{prefix}: more than one bbox exists for a frame")
        if frames:
            if frames[0] > frames[-1]:
                errors.append(f"{prefix}: start_frame is greater than end_frame")
            expected_frames = list(range(frames[0], frames[-1] + 1))
            if frames != expected_frames:
                errors.append(f"{prefix}: bbox frames are not continuous and strictly ordered")

        if isinstance(windows, list):
            previous_end: float | None = None
            for position, window in enumerate(windows, start=1):
                if not isinstance(window, dict):
                    errors.append(f"{prefix}: evidence window {position} is not an object")
                    continue
                start, end = window.get("start"), window.get("end")
                if (
                    isinstance(start, bool)
                    or isinstance(end, bool)
                    or not isinstance(start, (int, float))
                    or not isinstance(end, (int, float))
                ):
                    errors.append(f"{prefix}: evidence window {position} has invalid timestamps")
                elif start > end:
                    errors.append(f"{prefix}: evidence window {position} has start > end")
                elif previous_end is not None and start <= previous_end:
                    errors.append(f"{prefix}: evidence window {position} is not time-monotonic")
                else:
                    previous_end = float(end)
    return errors


def build_video_id_mapping(samples: list[dict[str, Any]]) -> dict[str, str]:
    """Convert all source IDs into one deterministic sport_0001-style sequence."""
    mapping: dict[str, str] = {}
    parsed_ids: list[tuple[int, int, str]] = []
    subset_order = {subset: position for position, subset in enumerate(SUBSETS)}
    for sample in samples:
        old_video_id = sample["video_id"]
        match = VIDEO_ID_RE.fullmatch(old_video_id)
        if match is None:
            raise ValueError(
                f"Cannot convert video_id {old_video_id!r}; expected sport_<number>"
            )
        if old_video_id in mapping:
            raise ValueError(f"Duplicate video_id while building mapping: {old_video_id!r}")
        mapping[old_video_id] = ""  # Marks this source ID as seen before sorting.
        parsed_ids.append(
            (
                subset_order[match.group("sport")],
                int(match.group("index")),
                old_video_id,
            )
        )
    for sequence, (_, _, old_video_id) in enumerate(sorted(parsed_ids), start=1):
        mapping[old_video_id] = f"sport_{sequence:04d}"
    return mapping


def apply_video_id_mapping(
    samples: list[dict[str, Any]], mapping: dict[str, str]
) -> list[dict[str, Any]]:
    """Return output samples with only video_id replaced by its normalized value."""
    converted: list[dict[str, Any]] = []
    for sample in samples:
        result = dict(sample)
        result["video_id"] = mapping[sample["video_id"]]
        converted.append(result)
    return converted


def main() -> int:
    root = DATASET_ROOT.resolve()
    qa_path = root / CUSTOMIZED_QA_FILE
    output_path = OUTPUT_PATH.resolve()
    mapping_path = VIDEO_ID_MAPPING_PATH.resolve()
    if not qa_path.is_file():
        print(f"Conversion aborted: QA file not found: {qa_path}", file=sys.stderr)
        return 1
    if output_path == mapping_path:
        print("Conversion aborted: output and mapping paths must be different", file=sys.stderr)
        return 1
    existing_outputs = [
        path for path in (output_path, mapping_path) if path.exists()
    ]
    if existing_outputs and not OVERWRITE_OUTPUT:
        print(
            "Conversion aborted: output already exists: "
            f"{', '.join(str(path) for path in existing_outputs)}; "
            "set OVERWRITE_OUTPUT = True to replace it",
            file=sys.stderr,
        )
        return 1

    try:
        samples = read_jsonl(qa_path)
        video_ids = [sample.get("video_id") for sample in samples]
        if any(not isinstance(video_id, str) or not video_id.strip() for video_id in video_ids):
            raise ValueError(f"{CUSTOMIZED_QA_FILE} contains a sample without a valid video_id")
        meta_fps = {video_id: find_meta_fps(root, video_id) for video_id in video_ids}
        evidence, project_fps, annotix_video_ids = load_coco_evidence(root)
        enriched = enrich_samples(samples, evidence, project_fps, meta_fps)
        qc_errors = run_qc(enriched, annotix_video_ids, evidence, meta_fps)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Conversion aborted: {exc}", file=sys.stderr)
        return 1

    if qc_errors:
        print(f"QC failed with {len(qc_errors)} issue(s); no output was written:", file=sys.stderr)
        for error in qc_errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    try:
        video_id_mapping = build_video_id_mapping(enriched)
        converted = apply_video_id_mapping(enriched, video_id_mapping)
        converted.sort(key=lambda sample: sample["video_id"])
    except ValueError as exc:
        print(f"Conversion aborted: {exc}", file=sys.stderr)
        return 1

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in converted:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with mapping_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(video_id_mapping, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"QC passed for {len(enriched)} samples.")
    print(f"Wrote {len(converted)} samples to {output_path}")
    print(f"Wrote {len(video_id_mapping)} video ID mappings to {mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
