"""Local Qwen3-VL via official Transformers processor's native video path.

Only explicitly verified native-video processor paths are supported. No frame extraction
or fallback to an image-only model is performed by the adapter.
"""
from ..config import env_value
from ..errors import ModelConfigError, ModelLoadError
from ..inference import ModelResponse
from ..media import Text, Video, Image

BACKEND = "huggingface"
DTYPE_NAMES = {"auto", "float16", "bfloat16", "float32"}


class HuggingFaceModel:
    def __init__(self, model, processor, spec):
        self.model = model
        self.processor = processor
        self.name = spec.name
        self.model_id = spec.model_id
        self.native_video_protocol = spec.options.get("native_video_protocol")

    def generate(self, messages, **kwargs) -> ModelResponse:
        if self.native_video_protocol != "qwen3_vl":
            raise ModelConfigError("Local native-video inference requires verified native_video_protocol: qwen3_vl")
        if not hasattr(self.processor, "apply_chat_template"):
            raise ModelLoadError("Processor does not implement apply_chat_template")
        conversation = []
        for message in messages:
            role = message["role"]
            if role not in {"user", "assistant", "system"}:
                raise ModelConfigError(f"Unsupported role: {role}")
            content = []
            for part in message["content"]:
                if isinstance(part, Text):
                    content.append({"type": "text", "text": part.value})
                elif isinstance(part, Video):
                    content.append({"type": "video", "path": str(part.path)})
                elif isinstance(part, Image):
                    content.append({"type": "image", "path": str(part.path)})
                else:
                    raise TypeError(f"Unsupported media part: {type(part).__name__}")
            conversation.append({"role": role, "content": content})
        # Processor is responsible for native video decoding. No adapter-level frame sampling.
        try:
            inputs = self.processor.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")
        except Exception as exc:
            raise ModelLoadError(f"Local processor failed to handle native video input: {exc}") from exc
        inputs = inputs.to(self.model.device)
        max_new_tokens = kwargs.pop("max_new_tokens", 256)
        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        text = self.processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)[0]
        return ModelResponse(text=text, model=self.model_id)


def load(spec) -> HuggingFaceModel:
    if spec.options.get("native_video_protocol") != "qwen3_vl":
        raise ModelConfigError("Set native_video_protocol: qwen3_vl for supported local models")
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except ImportError as exc:
        raise ModelLoadError("Install compatible torch, transformers and accelerate") from exc
    opts = spec.provider.options
    path_var = spec.options.get("model_path_env")
    model_path = env_value(path_var) if path_var else spec.model_id
    dtype_name = opts.get("dtype", "auto")
    if dtype_name not in DTYPE_NAMES:
        raise ModelConfigError(f"Unsupported dtype {dtype_name!r}")
    dtype = dtype_name if dtype_name == "auto" else getattr(torch, dtype_name)
    common = {"trust_remote_code": bool(opts.get("trust_remote_code", False))}
    if spec.options.get("revision"):
        common["revision"] = spec.options["revision"]
    try:
        processor = AutoProcessor.from_pretrained(model_path, **common)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=dtype, device_map=opts.get("device_map", "auto"), **common)
        model.eval()
    except Exception as exc:
        raise ModelLoadError(f"Failed to initialize local model {spec.name!r}: {exc}") from exc
    return HuggingFaceModel(model, processor, spec)
