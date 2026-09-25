import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCES_FILE = ROOT / "data" / "sources.jsonl"
VIDEOS_DIR = ROOT / "videos"

VALID_CATEGORIES = {
    "sports",
    "traffic",
    "crowd",
    "game",
    "magic",
}


def check_dependencies():
    """Check whether yt-dlp and ffmpeg are available."""
    missing = []

    for command in ["yt-dlp", "ffmpeg"]:
        if shutil.which(command) is None:
            missing.append(command)

    if missing:
        print(f"[ERROR] Missing dependencies: {', '.join(missing)}")
        sys.exit(1)


def load_sources():
    """Load sources.jsonl."""
    sources = []

    with open(SOURCES_FILE, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                sources.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[ERROR] Invalid JSON at line {line_number}: {e}")
                sys.exit(1)

    return sources


def prepare_web_video(item):
    video_id = item["video_id"]
    category = item["category"]
    url = item["source_url"]

    start = item["clip_start"]
    end = item["clip_end"]

    if category not in VALID_CATEGORIES:
        raise ValueError(f"Unknown category '{category}' for {video_id}")

    output_dir = VIDEOS_DIR / category
    output_dir.mkdir(parents=True, exist_ok=True)

    final_path = output_dir / f"{video_id}.mp4"

    # Avoid downloading the same video again
    if final_path.exists():
        print(f"[SKIP] {video_id} already exists.")
        return

    #
    # Step 1: Download source video
    #

    raw_template = output_dir / f"{video_id}_raw.%(ext)s"

    download_command = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "-o",
        str(raw_template),
        url,
    ]

    print(f"[DOWNLOAD] {video_id}")
    subprocess.run(download_command, check=True)

    raw_candidates = list(output_dir.glob(f"{video_id}_raw.*"))

    if not raw_candidates:
        raise FileNotFoundError(f"Downloaded source file not found for {video_id}")

    raw_path = raw_candidates[0]

    #
    # Step 2: Clip video
    #

    if start is None or end is None:
        # No clipping requested
        raw_path.rename(final_path)

        print(f"[OK] Saved: {final_path}")
        return

    if end <= start:
        raise ValueError(f"Invalid clip range for {video_id}: {start} -> {end}")

    print(f"[CLIP] {video_id}: {start:.3f}s -> {end:.3f}s")

    clip_command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        str(raw_path),
        "-t",
        str(end - start),
        "-vf",
        (
            "zscale=t=linear:npl=100,"
            "format=gbrpf32le,"
            "zscale=p=bt709,"
            "tonemap=tonemap=hable:desat=0,"
            "zscale=t=bt709:m=bt709:r=tv,"
            "format=yuv420p"
        ),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(final_path),
    ]

    subprocess.run(clip_command, check=True)

    # Remove temporary full-length source video
    raw_path.unlink()

    print(f"[OK] Saved: {final_path}")


def main():
    check_dependencies()

    sources = load_sources()

    print(f"Loaded {len(sources)} source(s).")

    success = 0
    failed = 0

    for item in sources:
        if item.get("source_type") != "web":
            continue

        video_id = item.get("video_id", "unknown")

        print()
        print("=" * 60)
        print(f"Processing: {video_id}")
        print("=" * 60)

        try:
            prepare_web_video(item)
            success += 1

        except Exception as e:
            print(f"[FAILED] {video_id}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print("Finished")
    print(f"Success: {success}")
    print(f"Failed : {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
