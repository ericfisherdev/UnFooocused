"""Pipeline factory — composition root for the DiffusionPipeline.

Constructs the full DiffusionPipeline by wiring infrastructure adapters
to their ldm_patched dependencies. This is the only module that knows
how to assemble the production pipeline from concrete implementations.

Called once at application startup. All ldm_patched imports are deferred
to build-time so the module can be imported (but not called) without CUDA.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import modules.config as config

logger = logging.getLogger(__name__)


def build_pipeline() -> Any | None:
    """Build a production DiffusionPipeline from ldm_patched infrastructure.

    Returns None if STUB_MODE is set or if ldm_patched/CUDA is unavailable,
    allowing the Worker to fall back to stub generation gracefully.

    Returns:
        A DiffusionPipeline instance wired to real ldm_patched adapters,
        or None if the pipeline cannot be constructed.
    """
    if os.environ.get("STUB_MODE", "").lower() == "true":
        logger.info("STUB_MODE=true — skipping pipeline construction")
        return None

    if not _cuda_available():
        logger.info("CUDA not available — skipping pipeline construction")
        return None

    try:
        return _build_pipeline_from_ldm()
    except ImportError:
        logger.warning(
            "ldm_patched unavailable — falling back to stub mode",
            exc_info=True,
        )
        return None


def _cuda_available() -> bool:
    """Check if CUDA is available and can initialize without OOM.

    Returns False if torch lacks CUDA support or if the GPU cannot
    allocate a CUDA context (e.g. VRAM fully occupied by another process).
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        # Force CUDA context initialization to catch OOM early
        torch.cuda.current_device()
        return True
    except RuntimeError:
        return False


def _get_model_patcher(model: Any) -> Any:
    """Extract the model patcher from a _LoadedModel or return as-is.

    The DiffusionPipeline passes _LoadedModel objects, but ldm_patched
    functions expect the model patcher (unet_with_lora). This bridges
    between the two representations.
    """
    return getattr(model, "unet_with_lora", model)


def _configure_ldm_args() -> None:
    """Configure ldm_patched args for low VRAM before any ldm_patched import.

    Normally called by launch.py before the server starts. This is a
    safety net for cases where the pipeline is built without the launcher
    (e.g. direct tests). Idempotent — does not overwrite values already
    set by the launcher.
    """
    from ldm_patched.modules.args_parser import args

    if not getattr(args, "always_offload_from_vram", False):
        args.always_offload_from_vram = True
    if not getattr(args, "disable_async_cuda_allocation", False):
        args.disable_async_cuda_allocation = True


def _build_pipeline_from_ldm() -> Any:
    """Construct the pipeline from ldm_patched adapters.

    Imports ldm_patched modules (requires CUDA) and wires each
    infrastructure adapter with its concrete dependencies.
    """
    _configure_ldm_args()

    import ldm_patched.contrib.external as external_nodes
    import ldm_patched.modules.sd
    from ldm_patched.k_diffusion.sampling import BrownianTreeNoiseSampler
    from modules.infrastructure.model_loader import LdmModelLoader
    from modules.infrastructure.patch_system import patch_all
    from modules.infrastructure.sampler import LdmSampler
    from modules.infrastructure.text_encoder import LdmTextEncoder
    from modules.infrastructure.vae_decoder import LdmVAEDecoder
    from modules.services.diffusion_pipeline import DiffusionPipeline

    # Install monkey-patches for sharpness, ADM guidance, adaptive CFG
    patch_all()

    cfg = config.get_config()

    # --- ModelLoader ---
    model_loader = LdmModelLoader(
        resolve_path=_make_checkpoint_resolver(cfg),
        load_fn=ldm_patched.modules.sd.load_checkpoint_guess_config,
        embedding_directory=cfg.path_embeddings,
        lora_paths=list(cfg.paths_loras),
    )

    # --- TextEncoder ---
    # The CLIP model is obtained from the loaded checkpoint at runtime.
    # We initialize with clip=None; the pipeline sets it after loading.
    text_encoder = LdmTextEncoder(clip=None)

    # --- Sampler ---
    sampler = LdmSampler(
        ksampler_fn=_make_ksampler_wrapper(),
        sigma_calculator=_make_sigma_calculator(),
        brownian_tree_init=BrownianTreeNoiseSampler,
        patch_settings_applier=_make_patch_applier(),
        generate_empty_latent_fn=_make_empty_latent_generator(),
    )

    # --- VAE Decoder ---
    vae_decode_op = external_nodes.VAEDecode()
    vae_decode_tiled_op = external_nodes.VAEDecodeTiled()
    vae_decoder = LdmVAEDecoder(
        vae_decode_op=vae_decode_op,
        vae_decode_tiled_op=vae_decode_tiled_op,
    )

    # --- Assemble pipeline ---
    pipeline = DiffusionPipeline(
        model_loader=model_loader,
        text_encoder=text_encoder,
        sampler=sampler,
        vae_decoder=vae_decoder,
    )

    logger.info("DiffusionPipeline constructed successfully")
    return pipeline


def _make_checkpoint_resolver(cfg: Any):
    """Create a checkpoint path resolver using config paths."""
    from modules.fast_checkpoint import resolve_checkpoint_path

    def resolve(name: str) -> str:
        return resolve_checkpoint_path(
            checkpoint_name=name,
            checkpoint_folders=list(cfg.paths_checkpoints),
            fast_path=cfg.path_fast_checkpoints or None,
        )

    return resolve


def _make_ksampler_wrapper():
    """Create a ksampler wrapper matching LdmSampler's expected signature.

    LdmSampler calls:
        ksampler_fn(model, positive, negative, latent, seed, steps,
                    cfg, sampler_name, scheduler, denoise, callback_function)

    The ``model`` argument is a _LoadedModel from the pipeline. We extract
    the model patcher (unet_with_lora) for ldm_patched, which expects an
    object with .load_device, .model_options, .model attributes.
    """
    import ldm_patched.modules.model_management
    import ldm_patched.modules.sample
    import torch

    @torch.no_grad()
    @torch.inference_mode()
    def ksampler_fn(
        *,
        model,
        positive,
        negative,
        latent,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        callback_function=None,
    ):
        patcher = _get_model_patcher(model)

        # Ensure model is loaded to GPU before sampling
        ldm_patched.modules.model_management.load_models_gpu([patcher])

        latent_image = latent["samples"]
        noise = ldm_patched.modules.sample.prepare_noise(latent_image, seed)

        samples = ldm_patched.modules.sample.sample(
            model=patcher,
            noise=noise,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
            callback=callback_function,
            seed=seed,
        )

        out = latent.copy()
        out["samples"] = samples
        return out

    return ksampler_fn


def _make_sigma_calculator():
    """Create a sigma calculator bridging to ldm_patched's KSampler.

    LdmSampler calls:
        sigma_calculator(sampler, model, scheduler, steps, denoise)

    The ``model`` argument here is model.model (the inner model from the
    model patcher), as extracted by LdmSampler line 167.
    """
    import ldm_patched.modules.model_management
    import ldm_patched.modules.samplers

    def calculate_sigmas(*, sampler, model, scheduler, steps, denoise):
        device = ldm_patched.modules.model_management.get_torch_device()
        ks = ldm_patched.modules.samplers.KSampler(
            model=model,
            steps=steps,
            device=device,
            sampler=sampler,
            scheduler=scheduler,
            denoise=denoise,
        )
        return ks.sigmas

    return calculate_sigmas


def _make_patch_applier():
    """Create a patch settings applier using the global registry."""
    from modules.infrastructure.patch_system import patch_settings_registry

    def apply_settings(**kwargs):
        pid = os.getpid()
        patch_settings_registry.set(pid, **kwargs)

    return apply_settings


def _make_empty_latent_generator():
    """Create a function that generates empty latent tensors."""
    import ldm_patched.contrib.external as external_nodes

    empty_latent_op = external_nodes.EmptyLatentImage()

    def generate_empty_latent(width: int, height: int):
        result = empty_latent_op.generate(width=width, height=height, batch_size=1)
        return result[0]

    return generate_empty_latent
