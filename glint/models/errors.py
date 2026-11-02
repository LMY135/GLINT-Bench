class ModelConfigError(ValueError):
    """Invalid or incomplete model configuration."""

class ModelLoadError(RuntimeError):
    """Provider could not initialize the selected model."""
