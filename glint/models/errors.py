class ModelConfigError(ValueError):
    """Invalid or incomplete model configuration."""

class ModelLoadError(RuntimeError):
    """Provider could not initialize the selected model."""


class ModelInferenceError(RuntimeError):
    """A model request failed after Adapter-owned retries."""
    def __init__(self, cause, attempts):
        super().__init__(str(cause))
        self.cause = cause
        self.attempts = attempts
