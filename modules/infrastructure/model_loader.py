"""LdmModelLoader — concrete ModelLoader using ldm_patched.

Bridges the domain ModelLoader protocol to ldm_patched's
load_checkpoint_guess_config and LoRA patching infrastructure.
All ldm_patched and torch dependencies are confined to this module.

Domain errors raised:
    ModelNotFoundError — checkpoint or LoRA file not found on disk.
    UnsupportedModelError — loaded model is not SDXL architecture.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import ldm_patched.modules.latent_formats as latent_formats
import ldm_patched.modules.utils
from ldm_patched.contrib.external_freelunch import FreeU_V2
from ldm_patched.modules.lora import model_lora_keys_clip, model_lora_keys_unet
from modules.domain.exceptions import ModelNotFoundError, UnsupportedModelError
from modules.fast_checkpoint import _find_in_folders
from modules.infrastructure.lora_matching import match_lora

logger = logging.getLogger(__name__)

_freeu_op = FreeU_V2()


class _LoadedModel:
    """Internal model bundle -- wraps unet, clip, vae with LoRA state.

    Provides the same interface as the legacy StableDiffusionModel
    but without God-object coupling to pipeline orchestration.
    """

    __slots__ = (
        "_lora_key_map_clip",
        "_lora_key_map_unet",
        "_visited_loras",
        "clip",
        "clip_vision",
        "clip_with_lora",
        "filename",
        "unet",
        "unet_with_lora",
        "vae",
        "vae_filename",
    )

    def __init__(
        self,
        *,
        unet: Any,
        clip: Any,
        vae: Any,
        clip_vision: Any | None,
        filename: str,
        vae_filename: str | None,
    ) -> None:
        self.unet = unet
        self.clip = clip
        self.vae = vae
        self.clip_vision = clip_vision
        self.filename = filename
        self.vae_filename = vae_filename
        self.unet_with_lora = unet
        self.clip_with_lora = clip
        self._visited_loras: str = ""

        self._lora_key_map_unet: dict[str, str] = {}
        self._lora_key_map_clip: dict[str, str] = {}

        if self.unet is not None:
            self._lora_key_map_unet = model_lora_keys_unet(self.unet.model, self._lora_key_map_unet)
            self._lora_key_map_unet.update({k: k for k in self.unet.model.state_dict()})

        if self.clip is not None:
            self._lora_key_map_clip = model_lora_keys_clip(self.clip.cond_stage_model, self._lora_key_map_clip)
            self._lora_key_map_clip.update({k: k for k in self.clip.cond_stage_model.state_dict()})

    @property
    def model(self) -> Any:
        """Delegate to the inner model from the UNet patcher.

        The LdmSampler and sigma calculator access ``model.model`` to
        get the underlying diffusion model. This property bridges
        ``_LoadedModel`` to the model patcher's inner model.
        """
        if self.unet_with_lora is not None:
            return self.unet_with_lora.model
        return None

    def __repr__(self) -> str:
        return (
            f"_LoadedModel(filename={self.filename!r}, "
            f"has_unet={self.unet is not None}, "
            f"has_clip={self.clip is not None}, "
            f"has_vae={self.vae is not None})"
        )


class LdmModelLoader:
    """Concrete ModelLoader adapter backed by ldm_patched.

    Dependencies are injected via constructor:
        resolve_path: resolves checkpoint name to absolute path
            (typically fast_checkpoint.resolve_checkpoint_path)
        load_fn: the actual checkpoint loader
            (typically ldm_patched.modules.sd.load_checkpoint_guess_config)
        embedding_directory: path to text embeddings
        lora_paths: directories to search for LoRA files
    """

    __slots__ = (
        "_cache",
        "_embedding_directory",
        "_load_fn",
        "_lora_paths",
        "_resolve_path",
    )

    def __init__(
        self,
        *,
        resolve_path: Callable[[str], str],
        load_fn: Callable[..., tuple],
        embedding_directory: str,
        lora_paths: Sequence[str],
    ) -> None:
        self._resolve_path = resolve_path
        self._load_fn = load_fn
        self._embedding_directory = embedding_directory
        self._lora_paths = list(lora_paths)
        self._cache: dict[str, _LoadedModel] = {}

    def load_checkpoint(self, path: str) -> _LoadedModel:
        """Load an SDXL checkpoint from disk.

        Args:
            path: Checkpoint filename or absolute path.

        Returns:
            A fresh model bundle with unet, clip, vae components.
            Each call returns a new ``_LoadedModel`` so LoRA mutations
            on one caller's instance never affect another's.

        Raises:
            ModelNotFoundError: If the checkpoint file does not exist.
            UnsupportedModelError: If the model is not SDXL architecture.
        """
        resolved = self._resolve_path(path)

        if resolved not in self._cache:
            self._load_and_cache(resolved, path)

        cached = self._cache[resolved]
        return _LoadedModel(
            unet=cached.unet,
            clip=cached.clip,
            vae=cached.vae,
            clip_vision=cached.clip_vision,
            filename=cached.filename,
            vae_filename=cached.vae_filename,
        )

    def _load_and_cache(self, resolved: str, original_path: str) -> None:
        """Load checkpoint from disk and store immutable base data in cache."""
        if not os.path.isfile(resolved):
            raise ModelNotFoundError(f"Checkpoint not found: {original_path} (resolved to {resolved})")

        start = time.monotonic()
        try:
            unet, clip, vae, vae_filename, clip_vision = self._load_fn(
                resolved,
                embedding_directory=self._embedding_directory,
            )
        except FileNotFoundError as exc:
            raise ModelNotFoundError(f"Checkpoint not found: {original_path}") from exc

        elapsed = time.monotonic() - start
        logger.info("Loaded checkpoint %s in %.1fs", original_path, elapsed)

        model = _LoadedModel(
            unet=unet,
            clip=clip,
            vae=vae,
            clip_vision=clip_vision,
            filename=resolved,
            vae_filename=vae_filename,
        )

        _validate_sdxl(model, original_path)

        self._cache[resolved] = model

    def load_loras(
        self,
        model: _LoadedModel,
        loras: list[Any],
    ) -> _LoadedModel:
        """Apply LoRA adapters to a loaded model.

        Args:
            model: Base model to patch.
            loras: List of LoRAConfig(filename, weight) to apply.

        Returns:
            The model with LoRA weights applied to unet_with_lora
            and clip_with_lora.

        Raises:
            ModelNotFoundError: If a LoRA file cannot be found.
        """
        lora_key = str([(cfg.filename, cfg.weight) for cfg in loras])

        if model._visited_loras == lora_key:
            return model

        if model.unet is None:
            model._visited_loras = lora_key
            return model

        loras_to_load = self._resolve_lora_paths(loras)

        # Stage clones locally — only publish onto model after all LoRAs
        # succeed, so a mid-loop failure leaves the model unchanged.
        staged_unet = model.unet.clone()
        staged_clip = model.clip.clone() if model.clip is not None else None

        for lora_filename, weight in loras_to_load:
            self._apply_single_lora(staged_unet, staged_clip, model, lora_filename, weight)

        # All LoRAs applied without error — commit staged state
        model.unet_with_lora = staged_unet
        model.clip_with_lora = staged_clip
        model._visited_loras = lora_key

        return model

    def _resolve_lora_paths(self, loras: list[Any]) -> list[tuple[str, float]]:
        """Resolve LoRA filenames to absolute paths.

        Raises:
            ModelNotFoundError: If any requested LoRA file cannot be found.
        """
        result: list[tuple[str, float]] = []
        for lora_config in loras:
            if lora_config.filename == "None":
                continue

            if os.path.isfile(lora_config.filename):
                lora_path = lora_config.filename
            else:
                lora_path = _find_in_folders(lora_config.filename, self._lora_paths)

            if not os.path.isfile(lora_path):
                raise ModelNotFoundError(f"LoRA file not found: {lora_config.filename}")

            result.append((lora_path, lora_config.weight))
        return result

    def _apply_single_lora(
        self,
        staged_unet: Any,
        staged_clip: Any | None,
        model: _LoadedModel,
        lora_filename: str,
        weight: float,
    ) -> None:
        """Load and apply a single LoRA file to staged clones.

        Operates on *staged* unet/clip objects rather than on ``model``
        directly, so the caller can discard them on failure without
        leaving the model in a partial state.
        """
        lora_sd = ldm_patched.modules.utils.load_torch_file(lora_filename, safe_load=False)

        lora_unet, lora_remaining = match_lora(lora_sd, model._lora_key_map_unet)
        lora_clip, lora_remaining = match_lora(lora_remaining, model._lora_key_map_clip)

        if len(lora_remaining) > 12:
            logger.warning(
                "LoRA %s has %d unmatched keys — possible model mismatch",
                lora_filename,
                len(lora_remaining),
            )
            return

        if lora_remaining:
            logger.info(
                "LoRA %s has %d unmatched keys: %s",
                lora_filename,
                len(lora_remaining),
                list(lora_remaining.keys())[:5],
            )

        if staged_unet is not None and lora_unet:
            loaded_keys = staged_unet.add_patches(lora_unet, weight)
            logger.info(
                "Applied LoRA %s to UNet with %d keys at weight %.2f",
                os.path.basename(lora_filename),
                len(loaded_keys),
                weight,
            )

        if staged_clip is not None and lora_clip:
            loaded_keys = staged_clip.add_patches(lora_clip, weight)
            logger.info(
                "Applied LoRA %s to CLIP with %d keys at weight %.2f",
                os.path.basename(lora_filename),
                len(loaded_keys),
                weight,
            )

    def apply_freeu(
        self,
        model: _LoadedModel,
        b1: float,
        b2: float,
        s1: float,
        s2: float,
    ) -> _LoadedModel:
        """Apply FreeU V2 parameters to a loaded model's UNet.

        Patches the UNet's output blocks via ldm_patched's FreeU_V2 operation,
        which modifies skip connections to improve generation quality without
        additional training.

        Args:
            model: The model to patch with FreeU parameters.
            b1: FreeU b1 backbone feature scaling factor.
            b2: FreeU b2 backbone feature scaling factor.
            s1: FreeU s1 skip feature scaling factor.
            s2: FreeU s2 skip feature scaling factor.

        Returns:
            The same model instance with unet_with_lora replaced by
            the FreeU-patched version.
        """
        (patched_unet,) = _freeu_op.patch(
            model=model.unet_with_lora,
            b1=b1,
            b2=b2,
            s1=s1,
            s2=s2,
        )
        model.unet_with_lora = patched_unet
        logger.info("Applied FreeU V2 (b1=%.2f, b2=%.2f, s1=%.2f, s2=%.2f)", b1, b2, s1, s2)
        return model


def _validate_sdxl(model: _LoadedModel, original_path: str) -> None:
    """Validate that a loaded model is SDXL architecture.

    Raises:
        UnsupportedModelError: If the model's latent format is not SDXL.
    """
    if model.unet is None:
        raise UnsupportedModelError(f"Model {original_path} has no UNet — cannot validate architecture")

    latent_format = model.unet.model.latent_format
    if not isinstance(latent_format, latent_formats.SDXL):
        format_name = type(latent_format).__name__
        raise UnsupportedModelError(
            f"Model {original_path} is {format_name}, not SDXL. Only SDXL models are supported."
        )
