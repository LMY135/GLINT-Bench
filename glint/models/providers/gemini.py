"""Google GenAI SDK: native video upload and generation."""
import time
from typing import Any
from ..config import env_value
from ..errors import ModelConfigError, ModelLoadError
from ..inference import ModelResponse
from ..media import Text, Video, Image

BACKEND = "gemini"


def _stage_config(types, stage):
    choice = types.Schema(type=types.Type.STRING, enum=["A", "B", "C", "D"])
    window = types.Schema(type=types.Type.OBJECT, properties={
        "start": types.Schema(type=types.Type.NUMBER),
        "end": types.Schema(type=types.Type.NUMBER),
    }, required=["start", "end"])
    box_entry = types.Schema(type=types.Type.OBJECT, properties={
        "frame": types.Schema(type=types.Type.INTEGER),
        "box": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER),
                            min_items=4, max_items=4),
    }, required=["frame", "box"])
    if stage == "qa":
        mime, schema = "text/x.enum", choice
    elif stage == "video":
        mime = "application/json"
        schema = types.Schema(type=types.Type.OBJECT, properties={
            "answer": choice,
            "evidence_windows": types.Schema(type=types.Type.ARRAY, items=window,
                                               min_items=1, max_items=1),
        }, required=["answer", "evidence_windows"])
    elif stage == "spatial":
        mime = "application/json"
        schema = types.Schema(type=types.Type.OBJECT, properties={
            "evidence_boxes": types.Schema(type=types.Type.ARRAY, items=box_entry),
        }, required=["evidence_boxes"])
    else:
        raise ModelConfigError(f"Unsupported Gemini stage: {stage}")
    return types.GenerateContentConfig(
        response_mime_type=mime, response_schema=schema,
        thinking_config=types.ThinkingConfig(include_thoughts=True))


def _separate_response(raw):
    candidates = getattr(raw, "candidates", None) or []
    parts = getattr(getattr(candidates[0], "content", None), "parts", None) if candidates else None
    if parts is None:
        return raw.text or "", None
    final, thoughts = [], []
    for part in parts:
        value = getattr(part, "text", None)
        if not isinstance(value, str) or not value:
            continue
        if getattr(part, "thought", False):
            thoughts.append(value)
        else:
            final.append(value)
    return "".join(final), "\n".join(thoughts) if thoughts else None


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
        stage = kwargs.pop("stage", None)
        if stage is not None:
            if "config" in kwargs:
                raise ModelConfigError("Gemini stage config conflicts with explicit config")
            kwargs["config"] = _stage_config(types, stage)
        contents = []
        system_instructions = []
        for message in messages:
            if message["role"] == "system":
                if any(not isinstance(part, Text) for part in message["content"]):
                    raise ModelConfigError("Gemini system instructions must contain text only")
                system_instructions.extend(part.value for part in message["content"])
                continue
            role = "model" if message["role"] == "assistant" else message["role"]
            if role not in {"user", "model"}:
                raise ModelConfigError(f"Unsupported Gemini role: {role}")
            parts = []
            for part in message["content"]:
                if isinstance(part, Text):
                    parts.append(types.Part.from_text(text=part.value))
                elif isinstance(part, (Video, Image)):
                    parts.append(self._upload(part))
                else:
                    raise TypeError(f"Unsupported media part: {type(part).__name__}")
            contents.append(types.Content(role=role, parts=parts))
        if system_instructions:
            config = kwargs.get("config")
            if config is None:
                config = types.GenerateContentConfig()
            if getattr(config, "system_instruction", None):
                raise ModelConfigError("Gemini system instruction conflicts with explicit config")
            config.system_instruction = "\n".join(system_instructions)
            kwargs["config"] = config
        raw = self.client.models.generate_content(model=self.model_id, contents=contents, **kwargs)
        final_text, thoughts = _separate_response(raw)
        return ModelResponse(text=final_text, model=self.model_id,
                             usage=getattr(raw, "usage_metadata", None),
                             raw_response=raw, thought_summary=thoughts)


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
