"""LdmTextEncoder -- concrete TextEncoder using ldm_patched CLIP.

Bridges the domain TextEncoder protocol to ldm_patched's CLIP tokenizer
and encoder. SDXL uses dual CLIP text encoders (CLIP-L and CLIP-G/OpenCLIP).
The encoder takes prompt strings, tokenizes them, and produces conditioning
tensors that guide the diffusion process.

Domain errors raised:
    RuntimeError -- CLIP model not loaded when encode() is called.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _concatenate_conds(cond_list: list[Any]) -> Any:
    """Concatenate conditioning tensors along dim=1.

    Uses torch.cat when tensors are available (production),
    falls back to returning the list as-is (unit tests with fakes).
    """
    if len(cond_list) == 1:
        return cond_list[0]
    try:
        import torch

        if isinstance(cond_list[0], torch.Tensor):
            return torch.cat(cond_list, dim=1)
    except ImportError:
        pass
    return cond_list


class LdmTextEncoder:
    """Concrete TextEncoder adapter backed by ldm_patched CLIP.

    Wraps clip.tokenize() and clip.encode_from_tokens() with:
    - clip_skip support via clip.clip_layer(-abs(clip_skip))
    - Per-text conditioning cache to avoid redundant GPU work
    - Multi-text encoding with pooled output accumulation

    Args:
        clip: An ldm_patched CLIP model instance, or None if not yet loaded.
    """

    __slots__ = ("_cache", "_clip")

    def __init__(self, *, clip: Any) -> None:
        self._clip = clip
        self._cache: dict[tuple[str, int], tuple[Any, Any]] = {}

    def __repr__(self) -> str:
        return f"LdmTextEncoder(has_clip={self._clip is not None}, cache_size={len(self._cache)})"

    def encode(self, texts: list[str], clip_skip: int) -> Any:
        """Encode text prompts into conditioning tensors.

        Args:
            texts: One or more text prompts to encode. Empty list returns None.
            clip_skip: Number of final CLIP layers to skip (1 = use all layers).

        Returns:
            Conditioning data as [[cond_tensor, {"pooled_output": pooled}]],
            or None if texts is empty.

        Raises:
            RuntimeError: If the CLIP model is not loaded.
        """
        if self._clip is None:
            msg = "CLIP model is not loaded"
            raise RuntimeError(msg)

        if not texts:
            return None

        effective_skip = max(clip_skip, 1) if clip_skip >= 0 else abs(clip_skip)
        self._clip.clip_layer(-effective_skip)

        cond_list = []
        pooled_acc: Any = 0

        for i, text in enumerate(texts):
            cond, pooled = self._encode_single(text, effective_skip)
            cond_list.append(cond)
            pooled_acc = pooled if i == 0 else pooled_acc + pooled

        return [[_concatenate_conds(cond_list), {"pooled_output": pooled_acc}]]

    def clear_cache(self) -> None:
        """Discard all cached conditioning results.

        Call after model or LoRA changes to avoid stale encodings.
        """
        self._cache.clear()

    def _encode_single(self, text: str, clip_skip: int) -> tuple[Any, Any]:
        """Encode a single text prompt, using cache when available.

        Args:
            text: The text prompt to encode.
            clip_skip: Effective clip_skip value (already clamped/normalized).

        Returns:
            Tuple of (conditioning_tensor, pooled_output).
        """
        cache_key = (text, clip_skip)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("[CLIP Cached] %d chars, clip_skip=%d", len(text), clip_skip)
            return cached

        tokens = self._clip.tokenize(text)
        cond, pooled = self._clip.encode_from_tokens(tokens, return_pooled=True)
        self._cache[cache_key] = (cond, pooled)
        logger.debug("[CLIP Encoded] %d chars, clip_skip=%d", len(text), clip_skip)
        return cond, pooled
