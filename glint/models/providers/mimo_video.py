"""MiMo native video input transport.

The official OpenAI-compatible API accepts Base64 Data URLs up to 50 MB
(encoded) or publicly accessible HTTPS URLs. Never convert video into images.
"""
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from ..errors import ModelConfigError

# Conservative threshold: Base64 adds ~33%, leaving room below 50 MB.
MAX_INLINE_MIMO_VIDEO_BYTES = 37_000_000


def mimo_video_url(video, *, url_map_env="MIMO_VIDEO_URL_MAP_PATH"):
    """Use an explicit URL mapping when provided, otherwise inline small MP4s."""
    from .openai_video import _data_uri

    map_path = os.getenv(url_map_env)
    if map_path:
        file = Path(map_path)
        if not file.is_file():
            raise ModelConfigError(f"{url_map_env} points to a missing JSON file: {file}")
        try:
            mapping = json.loads(file.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ModelConfigError(f"Invalid MiMo video URL mapping JSON: {file}") from exc
        if not isinstance(mapping, dict):
            raise ModelConfigError("MiMo video URL mapping must be a JSON object")
        path = Path(video.path)
        url = mapping.get(str(path.resolve())) or mapping.get(path.name) or mapping.get(path.stem)
        if url is not None:
            parsed = urlparse(url) if isinstance(url, str) else None
            if not parsed or parsed.scheme != "https" or not parsed.netloc:
                raise ModelConfigError(f"MiMo video URL for {path.name} must be a public HTTPS URL")
            return url

    if Path(video.path).stat().st_size > MAX_INLINE_MIMO_VIDEO_BYTES:
        raise ModelConfigError(
            f"MiMo Base64 video exceeds safe 50 MB encoded limit: {video.path}. "
            f"Provide a publicly accessible HTTPS video URL via a JSON mapping "
            f"referenced by {url_map_env}, keyed by filename or absolute path."
        )
    return _data_uri(video)
