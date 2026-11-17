"""Model Studio temporary OSS upload for Qwen native-video files.

Official flow: request upload policy, POST multipart to OSS, then use the
model-bound 48-hour oss:// URL with X-DashScope-OssResourceResolve.
"""
import time
from pathlib import Path
from urllib.parse import urlparse

from ..errors import ModelConfigError

# Conservative margin below the documented <7 MB raw-file Base64 threshold.
MAX_INLINE_VIDEO_BYTES = 7_000_000
CACHE_SECONDS = 46 * 60 * 60

_REGION_HOSTS = {
    "dashscope-intl.aliyuncs.com",
    "dashscope.aliyuncs.com",
}


class QwenTemporaryUploader:
    def __init__(self, api_key, base_url, model_id, *, session=None):
        host = urlparse(str(base_url or "")).hostname
        if host not in _REGION_HOSTS:
            raise ModelConfigError(
                "Qwen large-video upload requires the matching Model Studio "
                "regional endpoint: dashscope-intl.aliyuncs.com (Singapore) "
                "or dashscope.aliyuncs.com (Beijing)"
            )
        if session is None:
            try:
                import requests
            except ImportError as exc:
                raise ModelConfigError("Install requests for Qwen video uploads") from exc
            session = requests.Session()
        self.session = session
        self.api_key = api_key
        self.model_id = model_id
        self.upload_endpoint = f"https://{host}/api/v1/uploads"
        self.cache = {}

    def get_url(self, video):
        path = Path(video.path)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        now = time.monotonic()
        previous = self.cache.get(key)
        if previous is not None and previous[1] > now:
            return previous[0]
        policy_response = self.session.get(
            self.upload_endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            params={"action": "getPolicy", "model": self.model_id},
            timeout=30,
        )
        policy_response.raise_for_status()
        policy = policy_response.json()["data"]
        object_key = f"{policy['upload_dir']}/{path.name}"
        with path.open("rb") as stream:
            files = {
                "OSSAccessKeyId": (None, policy["oss_access_key_id"]),
                "Signature": (None, policy["signature"]),
                "policy": (None, policy["policy"]),
                "x-oss-object-acl": (None, policy["x_oss_object_acl"]),
                "x-oss-forbid-overwrite": (None, policy["x_oss_forbid_overwrite"]),
                "key": (None, object_key),
                "success_action_status": (None, "200"),
                "file": (path.name, stream, video.mime_type),
            }
            response = self.session.post(
                policy["upload_host"], files=files, timeout=(30, 600)
            )
        response.raise_for_status()
        url = f"oss://{object_key}"
        self.cache[key] = (url, time.monotonic() + CACHE_SECONDS)
        return url
