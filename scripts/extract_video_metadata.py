import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = ROOT / "videos"
OUTPUT_FILE = ROOT / "data" / "video_metadata.jsonl"


def probe_video(video_path: Path):
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames:format=duration",
        "-of", "json",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    data = json.loads(result.stdout)

    stream = data["streams"][0]
    format_info = data["format"]

    # FPS may be stored as a fraction such as "30/1" or "30000/1001"
    numerator, denominator = stream["r_frame_rate"].split("/")
    fps = float(numerator) / float(denominator)

    duration = float(format_info["duration"])

    # Prefer the exact frame count reported by ffprobe.
    # Fall back to duration * fps if unavailable.
    nb_frames = stream.get("nb_frames")

    if nb_frames and nb_frames != "N/A":
        num_frames = int(nb_frames)
    else:
        num_frames = round(duration * fps)

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": round(fps, 6),
        "duration": round(duration, 6),
        "num_frames": num_frames,
    }


def main():
    video_paths = sorted(VIDEOS_DIR.rglob("*.mp4"))

    print(f"Found {len(video_paths)} video(s).")

    records = []

    for video_path in video_paths:
        video_id = video_path.stem

        print(f"[PROBE] {video_id}")

        try:
            metadata = probe_video(video_path)

            record = {
                "video_id": video_id,
                **metadata,
            }

            records.append(record)

            print(
                f"[OK] {video_id}: "
                f"{metadata['width']}x{metadata['height']}, "
                f"{metadata['fps']} fps, "
                f"{metadata['duration']} s, "
                f"{metadata['num_frames']} frames"
            )

        except Exception as e:
            print(f"[FAILED] {video_id}: {e}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print()
    print("=" * 60)
    print(f"Videos found : {len(video_paths)}")
    print(f"Metadata saved: {len(records)}")
    print(f"Output       : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()