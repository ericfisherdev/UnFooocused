"""Domain-specific exceptions for the UnFooocused image generation pipeline.

These exceptions represent domain errors (not infrastructure errors).
Infrastructure adapters must translate low-level errors into these
domain exceptions at the boundary.
"""


class ModelNotFoundError(Exception):
    """Raised when a checkpoint or LoRA file cannot be found on disk.

    Domain callers should catch this instead of FileNotFoundError
    to stay decoupled from filesystem infrastructure.
    """


class UnsupportedModelError(Exception):
    """Raised when a loaded model is not a supported architecture.

    Currently only SDXL models are supported. Loading a SD1.5 or
    other architecture triggers this error.
    """


class GPUMemoryError(Exception):
    """Raised when a GPU operation fails due to insufficient VRAM.

    Infrastructure adapters must catch CUDA OOM errors and translate
    them into this domain exception so callers can handle memory
    pressure without coupling to torch internals.
    """
