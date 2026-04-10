# Vendored: ldm_patched

## Upstream Source

This directory is vendored from FwdFooocus, a fork of Fooocus which itself
uses a heavily patched ComfyUI diffusion engine.

- **Upstream repository:** FwdFooocus (private fork)
- **Vendored date:** 2026-04-10
- **Jira ticket:** UNF-30

## Purpose

ldm_patched provides the core diffusion engine for UnFooocused:

- **Model loading** (`modules/sd.py`) -- checkpoint loading via `load_checkpoint_guess_config`
- **Sampling** (`modules/samplers.py`) -- KSampler, sampler/scheduler name lists
- **Model management** (`modules/model_management.py`) -- GPU/CPU device management, VRAM offloading
- **VAE** (`ldm/models/autoencoder.py`) -- encode/decode latent representations
- **CLIP** (`modules/sd1_clip.py`, `modules/sdxl_clip.py`) -- text tokenization and encoding
- **ControlNet** (`controlnet/`) -- conditional generation support
- **LoRA** (`modules/lora.py`) -- low-rank adaptation support

## Modifications from Upstream

- Added `__init__.py` files to all subdirectories for proper Python package imports
- Fixed invalid escape sequences in `unipc/uni_pc.py` docstrings

## Maintenance Policy

This is a managed vendored dependency. Changes should be minimal and documented.
When upstream changes are needed, apply them as discrete, documented patches.
