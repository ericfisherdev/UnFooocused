"""Unit tests for LdmTextEncoder — no GPU, no real CLIP model.

Tests use a FakeClip stub that mimics ldm_patched's CLIP interface
(tokenize, encode_from_tokens, clip_layer) so we can verify encoding
logic, caching, clip_skip, empty-prompt handling, and protocol
conformance without loading any model weights.

Acceptance Criteria covered:
  AC1: modules/infrastructure/text_encoder.py exists implementing TextEncoder protocol
  AC3: clip_skip applied via clip.clip_layer(-abs(clip_skip))
  AC4: Conditioning cache prevents re-encoding identical prompts
  AC5: clear_cache() resets cache
  AC6: Empty prompt handling does not crash
  AC7: Negative prompt encoding produces valid negative conditioning
  AC9: Unit tests pass without GPU using fakes
"""

from __future__ import annotations

from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fake CLIP stub — mimics ldm_patched.modules.sd.CLIP interface
# ---------------------------------------------------------------------------


class FakeClip:
    """Minimal stub for ldm_patched CLIP model.

    Records calls to clip_layer, tokenize, and encode_from_tokens
    so tests can verify behavior without a real model.
    """

    def __init__(self) -> None:
        self.layer_idx: int | None = None
        self.tokenize_call_count: int = 0
        self.encode_call_count: int = 0

    def clip_layer(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx

    def tokenize(self, text: str, return_word_ids: bool = False) -> dict[str, list]:
        self.tokenize_call_count += 1
        # Return a fake token structure — shape doesn't matter for unit tests
        return {"tokens": [[1, 2, 3]], "text": text}

    def encode_from_tokens(self, tokens: Any, return_pooled: bool = False) -> Any:
        self.encode_call_count += 1
        # Simulate different output based on layer_idx to test clip_skip
        layer_marker = self.layer_idx if self.layer_idx is not None else 0
        fake_cond = f"cond_layer{layer_marker}"
        fake_pooled = f"pooled_layer{layer_marker}"
        if return_pooled:
            return fake_cond, fake_pooled
        return fake_cond


# ---------------------------------------------------------------------------
# AC1: text_encoder.py exists and LdmTextEncoder implements TextEncoder protocol
# ---------------------------------------------------------------------------


class TestLdmTextEncoderExistsAndSatisfiesProtocol:
    """AC1: modules/infrastructure/text_encoder.py exists implementing TextEncoder."""

    def test_module_importable(self) -> None:
        from modules.infrastructure.text_encoder import LdmTextEncoder

        assert LdmTextEncoder is not None

    def test_satisfies_text_encoder_protocol(self) -> None:
        from modules.domain.protocols import TextEncoder
        from modules.infrastructure.text_encoder import LdmTextEncoder

        encoder = LdmTextEncoder(clip=FakeClip())
        assert isinstance(encoder, TextEncoder)


# ---------------------------------------------------------------------------
# AC2: Encoding a prompt produces conditioning compatible with ksampler
# ---------------------------------------------------------------------------


class TestEncodingProducesConditioning:
    """AC2: Encoding a prompt produces a conditioning tensor."""

    def test_single_prompt_returns_list_of_cond_pooled_pairs(self) -> None:
        from modules.infrastructure.text_encoder import LdmTextEncoder

        encoder = LdmTextEncoder(clip=FakeClip())
        result = encoder.encode(["a photo of a cat"], clip_skip=1)
        # Conditioning format: [[cond_tensor, {"pooled_output": pooled_tensor}]]
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == 2
        assert "pooled_output" in result[0][1]

    def test_multiple_prompts_concatenated(self) -> None:
        from modules.infrastructure.text_encoder import LdmTextEncoder

        encoder = LdmTextEncoder(clip=FakeClip())
        result = encoder.encode(["line one", "line two"], clip_skip=1)
        assert isinstance(result, list)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# AC3: clip_skip applied via clip.clip_layer(-abs(clip_skip))
# ---------------------------------------------------------------------------


class TestClipSkipApplied:
    """AC3: clip_skip is applied correctly via clip.clip_layer(-abs(clip_skip))."""

    def test_clip_skip_1_sets_layer_minus_1(self) -> None:
        from modules.infrastructure.text_encoder import LdmTextEncoder

        clip = FakeClip()
        encoder = LdmTextEncoder(clip=clip)
        encoder.encode(["test"], clip_skip=1)
        assert clip.layer_idx == -1

    def test_clip_skip_2_sets_layer_minus_2(self) -> None:
        from modules.infrastructure.text_encoder import LdmTextEncoder

        clip = FakeClip()
        encoder = LdmTextEncoder(clip=clip)
        encoder.encode(["test"], clip_skip=2)
        assert clip.layer_idx == -2

    def test_negative_clip_skip_normalized_to_positive(self) -> None:
        from modules.infrastructure.text_encoder import LdmTextEncoder

        clip = FakeClip()
        encoder = LdmTextEncoder(clip=clip)
        encoder.encode(["test"], clip_skip=-3)
        assert clip.layer_idx == -3

    def test_clip_skip_zero_clamped_to_one(self) -> None:
        from modules.infrastructure.text_encoder import LdmTextEncoder

        clip = FakeClip()
        encoder = LdmTextEncoder(clip=clip)
        encoder.encode(["test"], clip_skip=0)
        assert clip.layer_idx == -1


# ---------------------------------------------------------------------------
# AC4: Conditioning cache prevents re-encoding identical prompts
# ---------------------------------------------------------------------------


class TestConditioningCache:
    """AC4: Cache prevents re-encoding identical prompts."""

    def test_same_text_same_clip_skip_uses_cache(self) -> None:
        from modules.infrastructure.text_encoder import LdmTextEncoder

        clip = FakeClip()
        encoder = LdmTextEncoder(clip=clip)
        encoder.encode(["cached prompt"], clip_skip=1)
        encode_count_after_first = clip.encode_call_count
        encoder.encode(["cached prompt"], clip_skip=1)
        # Should not have called encode again
        assert clip.encode_call_count == encode_count_after_first

    def test_same_text_different_clip_skip_not_cached(self) -> None:
        from modules.infrastructure.text_encoder import LdmTextEncoder

        clip = FakeClip()
        encoder = LdmTextEncoder(clip=clip)
        encoder.encode(["same prompt"], clip_skip=1)
        count_after_skip1 = clip.encode_call_count
        encoder.encode(["same prompt"], clip_skip=2)
        assert clip.encode_call_count > count_after_skip1

    def test_different_text_not_cached(self) -> None:
        from modules.infrastructure.text_encoder import LdmTextEncoder

        clip = FakeClip()
        encoder = LdmTextEncoder(clip=clip)
        encoder.encode(["prompt A"], clip_skip=1)
        count_after_a = clip.encode_call_count
        encoder.encode(["prompt B"], clip_skip=1)
        assert clip.encode_call_count > count_after_a


# ---------------------------------------------------------------------------
# AC5: clear_cache() resets the cache when models change
# ---------------------------------------------------------------------------


class TestClearCache:
    """AC5: clear_cache() resets the cache."""

    def test_clear_cache_forces_re_encode(self) -> None:
        from modules.infrastructure.text_encoder import LdmTextEncoder

        clip = FakeClip()
        encoder = LdmTextEncoder(clip=clip)
        encoder.encode(["test"], clip_skip=1)
        count_after_first = clip.encode_call_count
        encoder.clear_cache()
        encoder.encode(["test"], clip_skip=1)
        assert clip.encode_call_count > count_after_first


# ---------------------------------------------------------------------------
# AC6: Empty prompt handling does not crash
# ---------------------------------------------------------------------------


class TestEmptyPromptHandling:
    """AC6: Empty prompt handling does not crash."""

    def test_empty_list_returns_none(self) -> None:
        from modules.infrastructure.text_encoder import LdmTextEncoder

        encoder = LdmTextEncoder(clip=FakeClip())
        result = encoder.encode([], clip_skip=1)
        assert result is None

    def test_list_with_empty_string_does_not_crash(self) -> None:
        from modules.infrastructure.text_encoder import LdmTextEncoder

        encoder = LdmTextEncoder(clip=FakeClip())
        result = encoder.encode([""], clip_skip=1)
        # Should succeed and return valid conditioning
        assert result is not None
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# AC7: Negative prompt encoding produces valid negative conditioning
# ---------------------------------------------------------------------------


class TestNegativePromptEncoding:
    """AC7: Negative prompt produces valid negative conditioning.

    Negative prompts are just regular prompts encoded the same way.
    The 'negative' semantics come from how the sampler uses them,
    not from encoding. We verify the encoding succeeds and returns
    the same structure as positive conditioning.
    """

    def test_negative_prompt_returns_same_structure_as_positive(self) -> None:
        from modules.infrastructure.text_encoder import LdmTextEncoder

        encoder = LdmTextEncoder(clip=FakeClip())
        positive = encoder.encode(["beautiful landscape"], clip_skip=1)
        negative = encoder.encode(["ugly, blurry, distorted"], clip_skip=1)
        # Both should have same structure: [[cond, {"pooled_output": pooled}]]
        assert len(positive) == len(negative) == 1
        assert len(positive[0]) == len(negative[0]) == 2
        assert "pooled_output" in positive[0][1]
        assert "pooled_output" in negative[0][1]


# ---------------------------------------------------------------------------
# Encoder raises RuntimeError when CLIP not set
# ---------------------------------------------------------------------------


class TestEncoderWithoutClip:
    """Encoder raises RuntimeError if clip is None."""

    def test_encode_without_clip_raises_runtime_error(self) -> None:
        from modules.infrastructure.text_encoder import LdmTextEncoder

        encoder = LdmTextEncoder(clip=None)
        with pytest.raises(RuntimeError, match="CLIP model is not loaded"):
            encoder.encode(["test"], clip_skip=1)
