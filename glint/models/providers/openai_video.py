"""OpenAI SDK transport with explicit vendor-specific native video protocols."""
import base64
from pathlib import Path
from typing import Any
from ..config import env_value
from ..errors import ModelConfigError, ModelLoadError
from ..inference import ModelResponse
from ..media import Text, Video, Image
from .qwen_upload import QwenTemporaryUploader, MAX_INLINE_VIDEO_BYTES

BACKEND = "openai_compatible"


def _data_uri(part: Video | Image) -> str:
    content = Path(part.path).read_bytes()
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{part.mime_type};base64,{encoded}"


def _encode_parts(parts, protocol: str | None, *, video_uploader=None):
    converted = []
    for part in parts:
        if isinstance(part, Text):
            converted.append({"type": "text", "text": part.value})
        elif isinstance(part, Image):
            converted.append({"type": "image_url", "image_url": {"url": _data_uri(part)}})
        elif isinstance(part, Video):
            if protocol == "moonshot":
                converted.append({"type": "video_url", "video_url": _data_uri(part)})
            elif protocol == "qwen":
                if Path(part.path).stat().st_size > MAX_INLINE_VIDEO_BYTES:
                    if video_uploader is None:
                        raise ModelConfigError(
                            "Qwen video exceeds safe inline Base64 size; configure "
                            "Model Studio temporary OSS upload for this provider"
                        )
                    video_url = video_uploader.get_url(part)
                else:
                    video_url = _data_uri(part)
                converted.append({"type": "video_url", "video_url": {"url": video_url}})
            else:
                raise ModelConfigError("Set verified video_protocol to 'moonshot' or 'qwen'")
        else:
            raise TypeError(f"Unsupported media part: {type(part).__name__}")
    return converted


class OpenAIVideoModel:
    def __init__(self, client: Any, spec, video_uploader=None):
        self.client = client
        self.name = spec.name
        self.model_id = spec.model_id
        self.video_protocol = spec.options.get("video_protocol")
        self.video_uploader = video_uploader

    def generate(self, messages, **kwargs) -> ModelResponse:
        # stage is GLINT evaluation metadata, not an OpenAI Chat Completions parameter.
        kwargs.pop("stage", None)
        converted = []
        for message in messages:
            if message["role"] not in {"user", "assistant", "system"}:
                raise ModelConfigError(f"Unsupported message role: {message['role']}")
            converted.append({"role": message["role"], "content": _encode_parts(
                message["content"], self.video_protocol, video_uploader=self.video_uploader
            )})
        if self.video_protocol == "qwen" and any(
            isinstance(part, Video) and Path(part.path).stat().st_size > MAX_INLINE_VIDEO_BYTES
            for message in messages for part in message["content"]
        ):
            # oss:// URLs require this model-service resolution header.
            options = dict(kwargs)
            extra_headers = dict(options.pop("extra_headers", {}))
            extra_headers["X-DashScope-OssResourceResolve"] = "enable"
            options["extra_headers"] = extra_headers
            kwargs = options
        raw = self.client.chat.completions.create(model=self.model_id, messages=converted, **kwargs)
        return ModelResponse(text=raw.choices[0].message.content or "", model=self.model_id,
                             usage=getattr(raw, "usage", None), raw_response=raw)


def load(spec) -> OpenAIVideoModel:
    protocol = spec.options.get("video_protocol")
    if protocol not in {"moonshot", "qwen"}:
        raise ModelConfigError("OpenAI-compatible video models require video_protocol: moonshot or qwen")
    options = spec.provider.options
    api_key = env_value(options.get("api_key_env", "OPENAI_API_KEY"))
    base_url_env = options.get("base_url_env")
    base_url = env_value(base_url_env, required=False) if base_url_env else options.get("base_url")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ModelLoadError("Install openai SDK") from exc
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    if "timeout" in options:
        client_kwargs["timeout"] = options["timeout"]
    if "max_retries" in options:
        client_kwargs["max_retries"] = options["max_retries"]
    uploader = None
    if protocol == "qwen" and spec.options.get("qwen_video_upload", "auto") == "auto":
        # Do not instantiate for small videos until needed; the uploader is
        # configured with the same region, model ID and API key as inference.
        uploader = QwenTemporaryUploader(api_key, base_url, spec.model_id)
    return OpenAIVideoModel(OpenAI(**client_kwargs), spec, video_uploader=uploader)
