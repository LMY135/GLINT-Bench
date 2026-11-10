"""Google GenAI SDK: native video upload and generation."""
import time
from typing import Any
from ..config import env_value
from ..errors import ModelConfigError, ModelLoadError
from ..inference import ModelResponse
from ..media import Text, Video, Image

BACKEND = "gemini"


class GeminiModel:
    def __init__(self, client: Any, spec):
        self.client = client
        self.name = spec.name
        self.model_id = spec.model_id
        self.file_poll_seconds = float(spec.provider.options.get("file_poll_seconds", 2))
        self.file_timeout_seconds = float(spec.provider.options.get("file_timeout_seconds", 300))

    def _upload(self, part):
        from google.genai import types
        uploaded = self.client.files.upload(file=part.path)
        # Video Files API uploads can be processed asynchronously.
        if isinstance(part, Video):
            deadline = time.monotonic() + self.file_timeout_seconds
            while getattr(getattr(uploaded, "state", None), "name", None) == "PROCESSING":
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Gemini video processing timed out: {part.path}")
                time.sleep(self.file_poll_seconds)
                uploaded = self.client.files.get(name=uploaded.name)
            if getattr(getattr(uploaded, "state", None), "name", None) == "FAILED":
                raise ModelLoadError(f"Gemini video processing failed: {part.path}")
        return types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type or part.mime_type)

    def generate(self, messages, **kwargs) -> ModelResponse:
        from google.genai import types
        contents = []
        for message in messages:
            role = "model" if message["role"] == "assistant" else message["role"]
            if role not in {"user", "model"}:
                raise ModelConfigError(f"Unsupported Gemini role: {role}; system instructions require separate SDK config")
            parts = []
            for part in message["content"]:
                if isinstance(part, Text):
                    parts.append(types.Part.from_text(text=part.value))
                elif isinstance(part, (Video, Image)):
                    parts.append(self._upload(part))
                else:
                    raise TypeError(f"Unsupported media part: {type(part).__name__}")
            contents.append(types.Content(role=role, parts=parts))
        raw = self.client.models.generate_content(model=self.model_id, contents=contents, **kwargs)
        return ModelResponse(text=raw.text or "", model=self.model_id,
                             usage=getattr(raw, "usage_metadata", None), raw_response=raw)


def load(spec) -> GeminiModel:
    api_key = env_value(spec.provider.options.get("api_key_env", "GEMINI_API_KEY"))
    try:
        from google import genai
    except ImportError as exc:
        raise ModelLoadError("Install google-genai SDK") from exc
    options = spec.provider.options
    client_kwargs = {"api_key": api_key}
    http_options = {}
    if "timeout" in options:
        # GenAI HttpOptions uses milliseconds; registry timeout uses seconds.
        http_options["timeout"] = int(float(options["timeout"]) * 1000)
    if "max_retries" in options:
        http_options["retry_options"] = {"attempts": int(options["max_retries"]) + 1}
    if http_options:
        client_kwargs["http_options"] = http_options
    return GeminiModel(genai.Client(**client_kwargs), spec)
