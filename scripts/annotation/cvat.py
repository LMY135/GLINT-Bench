import xml.etree.ElementTree as ET


VALID_CATEGORIES = {
    "sports",
    "traffic",
    "crowd",
    "game",
    "magic",
}


def parse_track(track, video_id):
    """
    Parse one CVAT track.

    CVAT track label:
        q01, q02, q03, ...

    Bounding box format:
        pixel xyxy
        [x1, y1, x2, y2]
    """

    track_label = track.attrib.get("label")

    if not track_label:
        raise ValueError(
            f"Track in {video_id} has no label."
        )

    if (
        not track_label.startswith("q")
        or not track_label[1:].isdigit()
    ):
        raise ValueError(
            f"Invalid track label '{track_label}' "
            f"in {video_id}. "
            f"Expected q01, q02, q03, ..."
        )

    question_id = f"{video_id}_{track_label}"

    frames = []

    for box in track.findall("box"):

        # outside=1 means the target is no longer
        # part of the valid evidence track.
        if box.attrib.get("outside", "0") == "1":
            continue

        frame = int(box.attrib["frame"])

        x1 = float(box.attrib["xtl"])
        y1 = float(box.attrib["ytl"])
        x2 = float(box.attrib["xbr"])
        y2 = float(box.attrib["ybr"])

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid bbox in {question_id}, "
                f"frame {frame}: "
                f"[{x1}, {y1}, {x2}, {y2}]"
            )
        
        width = x2 - x1
        height = y2 - y1

        frames.append(
            {
                "frame": frame,
                "box": [
                    round(x1, 2),
                    round(y1, 2),
                    round(width, 2),
                    round(height, 2),
                ],
            }
        )

    if not frames:
        raise ValueError(
            f"No valid boxes found for {question_id}."
        )

    frames.sort(
        key=lambda item: item["frame"]
    )

    frame_ids = [
        item["frame"]
        for item in frames
    ]

    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError(
            f"Duplicate frame annotations "
            f"in {question_id}."
        )

    return {
        "question_id": question_id,
        "start_frame": frames[0]["frame"],
        "end_frame": frames[-1]["frame"],
        "frames": frames,
    }


def parse_cvat_xml(xml_path):
    """
    Parse one CVAT XML file.

    Expected path:
        data/annotations/<category>/<video_id>.xml
    """

    category = xml_path.parent.name

    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. "
            f"Expected one of: "
            f"{', '.join(sorted(VALID_CATEGORIES))}"
        )

    video_id = xml_path.stem

    tree = ET.parse(xml_path)
    root = tree.getroot()

    original_size = root.find(
        "./meta/original_size"
    )

    if original_size is None:
        raise ValueError(
            f"Missing original_size in {xml_path}."
        )

    width_text = original_size.findtext("width")
    height_text = original_size.findtext("height")

    if not width_text or not height_text:
        raise ValueError(
            f"Missing width/height in {xml_path}."
        )

    width = int(width_text)
    height = int(height_text)

    tracks = []

    for track in root.findall("track"):
        tracks.append(
            parse_track(
                track=track,
                video_id=video_id,
            )
        )

    if not tracks:
        raise ValueError(
            f"No tracks found in {xml_path}."
        )

    return {
        "category": category,
        "video_id": video_id,
        "width": width,
        "height": height,
        "tracks": tracks,
    }
