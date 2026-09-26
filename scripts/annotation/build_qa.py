import argparse
import sys
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

ANNOTATION_SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(ANNOTATION_SCRIPT_DIR),
)

from cvat import parse_cvat_xml
from utils import load_jsonl, save_jsonl


ANNOTATIONS_DIR = (
    ROOT
    / "data"
    / "annotations"
)

METADATA_FILE = (
    ROOT
    / "data"
    / "video_metadata.jsonl"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "qa.jsonl"
)


# Human-authored fields must never be overwritten
# when evidence is regenerated.
HUMAN_QA_FIELDS = [
    "question_type",
    "question",
    "answer",
]


# --------------------------------------------------
# Arguments
# --------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build GLINT-Bench QA annotations "
            "from CVAT XML files."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Update metadata and evidence for existing "
            "questions while preserving question_type, "
            "question, and answer."
        ),
    )

    return parser.parse_args()


# --------------------------------------------------
# Build one QA sample
# --------------------------------------------------

def build_sample(
    cvat_data,
    track,
    metadata,
):

    video_id = cvat_data["video_id"]
    category = cvat_data["category"]

    fps = float(metadata["fps"])

    if fps <= 0:
        raise ValueError(
            f"Invalid FPS for {video_id}: {fps}"
        )

    evidence_boxes = []

    for item in track["frames"]:

        frame = item["frame"]

        # Convert frame index to seconds.
        time = frame / fps

        evidence_boxes.append(
            {
                "frame": frame,
                "time": round(time, 3),
                "box": item["box"],
            }
        )

    start_time = (
        track["start_frame"] / fps
    )

    end_time = (
        track["end_frame"] / fps
    )

    return {
        "question_id": track["question_id"],
        "video_id": video_id,
        "category": category,
        "question_type": "",
        "question": "",
        "answer": "",
        "evidence_windows": [
            {
                "start": round(
                    start_time,
                    3,
                ),
                "end": round(
                    end_time,
                    3,
                ),
            }
        ],
        "evidence_boxes": evidence_boxes,
    }


# --------------------------------------------------
# Preserve manually written QA
# --------------------------------------------------

def preserve_human_qa(
    new_sample,
    old_sample,
):

    for field in HUMAN_QA_FIELDS:

        new_sample[field] = (
            old_sample.get(
                field,
                "",
            )
        )

    return new_sample


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    args = parse_args()

    # ------------------------------
    # Load video metadata
    # ------------------------------

    metadata = load_jsonl(
        METADATA_FILE,
        key="video_id",
    )

    print(
        f"Video metadata: "
        f"{len(metadata)}"
    )

    # ------------------------------
    # Load existing QA
    # ------------------------------

    existing = load_jsonl(
        OUTPUT_FILE,
        key="question_id",
    )

    print(
        f"Existing QA: "
        f"{len(existing)}"
    )

    # ------------------------------
    # Find CVAT XML
    # ------------------------------

    xml_paths = sorted(
        ANNOTATIONS_DIR.rglob("*.xml")
    )

    print(
        f"CVAT XML files: "
        f"{len(xml_paths)}"
    )

    print()

    added = 0
    updated = 0
    skipped = 0
    failed = 0

    # ------------------------------
    # Process XML files
    # ------------------------------

    for xml_path in xml_paths:

        relative_path = (
            xml_path.relative_to(ROOT)
        )

        print(
            f"[PARSE] {relative_path}"
        )

        try:

            cvat_data = parse_cvat_xml(
                xml_path
            )

            video_id = (
                cvat_data["video_id"]
            )

            # --------------------------
            # Check metadata
            # --------------------------

            if video_id not in metadata:

                raise ValueError(
                    f"Missing video metadata "
                    f"for {video_id}."
                )

            video_metadata = (
                metadata[video_id]
            )

            # --------------------------
            # Check resolution
            # --------------------------

            metadata_width = int(
                video_metadata["width"]
            )

            metadata_height = int(
                video_metadata["height"]
            )

            cvat_width = (
                cvat_data["width"]
            )

            cvat_height = (
                cvat_data["height"]
            )

            if (
                metadata_width != cvat_width
                or
                metadata_height != cvat_height
            ):

                raise ValueError(
                    f"Resolution mismatch "
                    f"for {video_id}: "
                    f"CVAT="
                    f"{cvat_width}x"
                    f"{cvat_height}, "
                    f"metadata="
                    f"{metadata_width}x"
                    f"{metadata_height}"
                )

            # --------------------------
            # Process tracks
            # --------------------------

            for track in cvat_data["tracks"]:

                question_id = (
                    track["question_id"]
                )

                already_exists = (
                    question_id in existing
                )

                # ----------------------
                # Skip existing sample
                # ----------------------

                if (
                    already_exists
                    and not args.overwrite
                ):

                    print(
                        f"  [SKIP] "
                        f"{question_id}"
                    )

                    skipped += 1

                    continue

                # ----------------------
                # Build evidence
                # ----------------------

                sample = build_sample(
                    cvat_data=cvat_data,
                    track=track,
                    metadata=video_metadata,
                )

                # ----------------------
                # Update existing sample
                # ----------------------

                if already_exists:

                    old_sample = (
                        existing[
                            question_id
                        ]
                    )

                    # Preserve manually written:
                    #
                    # question_type
                    # question
                    # answer
                    sample = preserve_human_qa(
                        new_sample=sample,
                        old_sample=old_sample,
                    )

                    existing[
                        question_id
                    ] = sample

                    updated += 1

                    print(
                        f"  [UPDATE] "
                        f"{question_id}: "
                        f"{len(sample['evidence_boxes'])} "
                        f"boxes"
                    )

                # ----------------------
                # Add new sample
                # ----------------------

                else:

                    existing[
                        question_id
                    ] = sample

                    added += 1

                    print(
                        f"  [ADD] "
                        f"{question_id}: "
                        f"{len(sample['evidence_boxes'])} "
                        f"boxes"
                    )

        except Exception as e:

            print(
                f"  [FAILED] "
                f"{relative_path}: {e}"
            )

            failed += 1

    # ------------------------------
    # Sort QA
    # ------------------------------

    records = sorted(
        existing.values(),
        key=lambda record: (
            record["question_id"]
        ),
    )

    # ------------------------------
    # Save only when changed
    # ------------------------------

    changed = (
        added > 0
        or
        updated > 0
    )

    if changed:

        save_jsonl(
            OUTPUT_FILE,
            records,
        )

        print()
        print(
            f"[SAVED] {OUTPUT_FILE}"
        )

    else:

        print()
        print(
            "[NO CHANGE] "
            "qa.jsonl was not rewritten."
        )

    # ------------------------------
    # Summary
    # ------------------------------

    print()
    print("=" * 60)

    print(
        f"Added   : {added}"
    )

    print(
        f"Updated : {updated}"
    )

    print(
        f"Skipped : {skipped}"
    )

    print(
        f"Failed  : {failed}"
    )

    print(
        f"Total   : {len(records)}"
    )

    print(
        f"Output  : {OUTPUT_FILE}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()