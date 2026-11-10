"""OpenAI SDK transport with explicit vendor-specific native video protocols."""
import base64
from pathlib import Path
from typing import Any
from ..config import env_value
from ..errors import ModelConfigError, ModelLoadError
from ..inference import ModelResponse
from ..media import Text, Video, Image

BACKEND = "openai_compatible"


def _data_uri(part: Video | Image) -> str:
    content = Path(part.path).read_bytes()
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{part.mime_type};base64,{encoded}"


def _encode_parts(parts, protocol: str | None):
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
                converted.append({"type": "video_url", "video_url": {"url": _data_uri(part)}})
            else:
                raise ModelConfigError("Set verified video_protocol to 'moonshot' or 'qwen'")
        else:
            raise TypeError(f"Unsupported media part: {type(part).__name__}")
    return converted


class OpenAIVideoModel:
    def __init__(self, client: Any, spec):
        self.client = client
        self.name = spec.name
        self.model_id = spec.model_id
        self.video_protocol = spec.options.get("video_protocol")

    def generate(self, messages, **kwargs) -> ModelResponse:
        converted = []
        for message in messages:
            if message["role"] not in {"user", "assistant", "system"}:
                raise ModelConfigError(f"Unsupported message role: {message['role']}")
            converted.append({"role": message["role"], "content": _encode_parts(message["content"], self.video_protocol)})
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
    return OpenAIVideoModel(OpenAI(**client_kwargs), spec)
