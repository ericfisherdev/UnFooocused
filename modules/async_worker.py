"""Asynchronous generation pipeline for UnFooocused.

Owns the task queue, task state, and worker that processes generation
requests. The actual diffusion step is currently a stub (solid-color
placeholder images) — real backend integration is a future task.

Domain concepts:
  - AsyncTask: holds all generation parameters, progress yields, and results
  - Worker: processes tasks from the queue, yielding progress events
  - async_tasks: module-level task queue (list)
  - current_task: module-level reference to the task being processed
"""

from __future__ import annotations

import logging
import os
import random
from typing import TYPE_CHECKING

from modules.heartbeat import is_browser_connected

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state — the task queue and current task reference
# ---------------------------------------------------------------------------

async_tasks: list[AsyncTask] = []
current_task: AsyncTask | None = None


# ---------------------------------------------------------------------------
# LoRA filtering helper
# ---------------------------------------------------------------------------


def _get_enabled_loras(
    raw_loras: list[tuple[bool, str, float]],
) -> list[tuple[str, float]]:
    """Filter LoRA entries to only those that are enabled and not 'None'.

    Args:
        raw_loras: List of (enabled, filename, weight) triples.

    Returns:
        List of (filename, weight) pairs for enabled LoRAs.
    """
    return [(name, weight) for enabled, name, weight in raw_loras if enabled and name != "None"]


# ---------------------------------------------------------------------------
# AsyncTask — holds task state and parsed generation parameters
# ---------------------------------------------------------------------------


class AsyncTask:
    """A generation task with parsed parameters, progress yields, and results.

    Constructed from the positional args list produced by
    ``ui.app._build_generate_args()``. The args list is consumed via
    ``reverse()`` + ``pop()``.

    Attributes:
        yields: List of (flag, product) tuples for progress streaming.
            flag is one of "preview", "results", "finish".
        results: List of output file paths.
        processing: True while the worker is actively processing this task.
        last_stop: Set to "stop" to request cancellation.
    """

    def __init__(self, args: list) -> None:
        from modules.config import get_config
        from modules.flags import MetadataScheme, ip_list

        self.args = args.copy()
        self.yields: list[tuple[str, object]] = []
        self.results: list[str] = []
        self.last_stop: bool | str = False
        self.processing: bool = False

        if len(args) == 0:
            return

        cfg = get_config()

        args = args.copy()
        args.reverse()

        self.generate_image_grid: bool = args.pop()
        self.prompt: str = args.pop()
        self.negative_prompt: str = args.pop()

        self.aspect_ratios_selection: str = args.pop()
        self.width, self.height = _parse_resolution(self.aspect_ratios_selection)

        self.image_number: int = args.pop()
        self.output_format: str = args.pop()
        self.seed: int = int(args.pop())
        self.read_wildcards_in_order: bool = args.pop()
        self.sharpness: float = args.pop()
        self.cfg_scale: float = args.pop()
        self.base_model_name: str = args.pop()
        self.refiner_model_name: str = args.pop()
        self.refiner_switch: float = args.pop()

        self.loras: list[tuple[str, float]] = _get_enabled_loras(
            [(bool(args.pop()), str(args.pop()), float(args.pop())) for _ in range(cfg.default_max_lora_number)]
        )

        self.input_image_checkbox: bool = args.pop()
        self.current_tab: str = args.pop()
        self.uov_method: str = args.pop()
        self.uov_input_image = args.pop()
        self.outpaint_selections: list = args.pop()
        self.inpaint_input_image = args.pop()
        self.inpaint_additional_prompt: str = args.pop()
        self.inpaint_mask_image_upload = args.pop()

        self.disable_preview: bool = args.pop()
        self.disable_intermediate_results: bool = args.pop()
        self.disable_seed_increment: bool = args.pop()
        self.black_out_nsfw: bool = args.pop()
        self.adm_scaler_positive: float = args.pop()
        self.adm_scaler_negative: float = args.pop()
        self.adm_scaler_end: float = args.pop()
        self.adaptive_cfg: float = args.pop()
        self.clip_skip: int = args.pop()
        self.sampler_name: str = args.pop()
        self.scheduler_name: str = args.pop()
        self.vae_name: str = args.pop()
        self.overwrite_step: int = args.pop()
        self.overwrite_switch: int = args.pop()
        self.overwrite_width: int = args.pop()
        self.overwrite_height: int = args.pop()
        self.overwrite_vary_strength: float = args.pop()
        self.overwrite_upscale_strength: float = args.pop()
        self.mixing_image_prompt_and_vary_upscale: bool = args.pop()
        self.mixing_image_prompt_and_inpaint: bool = args.pop()
        self.debugging_cn_preprocessor: bool = args.pop()
        self.skipping_cn_preprocessor: bool = args.pop()
        self.canny_low_threshold: int = args.pop()
        self.canny_high_threshold: int = args.pop()
        self.refiner_swap_method: str = args.pop()
        self.controlnet_softness: float = args.pop()
        self.freeu_enabled: bool = args.pop()
        self.freeu_b1: float = args.pop()
        self.freeu_b2: float = args.pop()
        self.freeu_s1: float = args.pop()
        self.freeu_s2: float = args.pop()
        self.debugging_inpaint_preprocessor: bool = args.pop()
        self.inpaint_disable_initial_latent: bool = args.pop()
        self.inpaint_engine: str = args.pop()
        self.inpaint_strength: float = args.pop()
        self.inpaint_respective_field: float = args.pop()
        self.inpaint_advanced_masking_checkbox: bool = args.pop()
        self.invert_mask_checkbox: bool = args.pop()
        self.inpaint_erode_or_dilate: int = args.pop()
        self.save_final_enhanced_image_only: bool = args.pop()
        self.save_metadata_to_images: bool = args.pop()
        self.metadata_scheme: MetadataScheme = MetadataScheme(args.pop())

        self.cn_tasks: dict[str, list] = {x: [] for x in ip_list}
        for _ in range(cfg.default_controlnet_image_count):
            cn_img = args.pop()
            cn_stop = args.pop()
            cn_weight = args.pop()
            cn_type = args.pop()
            if cn_img is not None:
                self.cn_tasks[cn_type].append([cn_img, cn_stop, cn_weight])

        self.debugging_dino: bool = args.pop()
        self.dino_erode_or_dilate: int = args.pop()
        self.debugging_enhance_masks_checkbox: bool = args.pop()
        self.enhance_input_image = args.pop()
        self.enhance_checkbox: bool = args.pop()
        self.enhance_uov_method: str = args.pop()
        self.enhance_uov_processing_order: str = args.pop()
        self.enhance_uov_prompt_type: str = args.pop()

    @property
    def effective_steps(self) -> int:
        """Return the step count to use for generation.

        If ``overwrite_step`` is positive the user explicitly chose a step
        count and it takes precedence.  Otherwise fall back to the
        hardcoded default of 30 steps.
        """
        if hasattr(self, "overwrite_step") and self.overwrite_step > 0:
            return self.overwrite_step
        return 30


# ---------------------------------------------------------------------------
# Resolution parsing
# ---------------------------------------------------------------------------


def _parse_resolution(aspect_ratio: str) -> tuple[int, int]:
    """Parse 'WIDTHxHEIGHT' or 'WIDTH*HEIGHT' into (width, height).

    Args:
        aspect_ratio: String like "1024*1024" or "1152*896".

    Returns:
        Tuple of (width, height) as integers.
    """
    separator = "*" if "*" in aspect_ratio else "x"
    parts = aspect_ratio.split(separator)
    return int(parts[0]), int(parts[1])


# ---------------------------------------------------------------------------
# Worker — processes tasks from the queue
# ---------------------------------------------------------------------------


class Worker:
    """Processes AsyncTask instances, generating stub images.

    The actual diffusion step is a placeholder that creates solid-color
    images. Real backend integration (ComfyUI, diffusers, etc.) will
    replace ``_generate_stub_image()`` in a future task.

    Args:
        output_dir: Base directory for saving output images.
    """

    def __init__(self, output_dir: str = "./outputs/") -> None:
        self.output_dir = output_dir

    def process_task(self, task: AsyncTask) -> None:
        """Process a single generation task.

        Sets ``task.processing = True`` during execution, yields progress
        events to ``task.yields``, saves images via ``modules.output``,
        and yields a final "finish" event with all output paths.

        Args:
            task: The AsyncTask to process.
        """
        global current_task
        task.processing = True
        current_task = task

        try:
            self._run_generation(task)
        finally:
            task.processing = False
            current_task = None

    def _run_generation(self, task: AsyncTask) -> None:
        """Execute the generation loop for a task."""
        logger.info("Loading checkpoint: %s", task.base_model_name)

        for lora_name, lora_weight in task.loras:
            logger.info("Applying LoRA: %s (weight=%.2f)", lora_name, lora_weight)

        steps = task.effective_steps
        output_paths: list[str] = []

        for image_index in range(task.image_number):
            if task.last_stop == "stop":
                logger.info("Generation cancelled at image %d/%d", image_index + 1, task.image_number)
                break

            if not is_browser_connected() and image_index > 0:
                logger.info(
                    "Browser disconnected, skipping remaining images at %d/%d", image_index + 1, task.image_number
                )
                break

            filepath = self._generate_single_image(task, image_index, steps)
            if filepath is None:
                break
            output_paths.append(filepath)

        task.results = output_paths
        task.yields.append(("finish", output_paths))

    def _generate_single_image(
        self,
        task: AsyncTask,
        image_index: int,
        steps: int,
    ) -> str | None:
        """Generate a single image with progress updates.

        Yields step-level progress events, creates a stub image, and
        saves it to the output directory.  Returns ``None`` if the task
        was stopped during step simulation — the caller must not save.

        Args:
            task: The parent task (for yielding progress).
            image_index: Zero-based index of the current image.
            steps: Number of diffusion steps to simulate.

        Returns:
            Absolute path to the saved image file, or None if stopped.
        """
        self._yield_step_progress(task, image_index, steps)

        if task.last_stop == "stop":
            return None

        effective_seed = task.seed if task.disable_seed_increment else task.seed + image_index
        image = _generate_stub_image(task.width, task.height, effective_seed)
        return self._save_generated_image(task, image)

    def _yield_step_progress(
        self,
        task: AsyncTask,
        image_index: int,
        steps: int,
    ) -> None:
        """Yield preview events for each diffusion step.

        Appends ("preview", (percentage, text, None)) to task.yields
        for each step, checking for cancellation between steps.
        """
        total_steps = task.image_number * steps
        for step in range(steps):
            if task.last_stop == "stop":
                break

            completed = image_index * steps + step + 1
            percentage = int(completed / total_steps * 100)
            text = f"Image {image_index + 1}/{task.image_number}, Step {step + 1}/{steps}"
            if not task.disable_preview:
                task.yields.append(("preview", (percentage, text, None)))

    def _save_generated_image(self, task: AsyncTask, image: Image) -> str:
        """Save a generated image to the output directory.

        Creates the date-based subdirectory if it doesn't exist, then
        delegates to ``modules.output.save_image()``.

        Returns:
            Absolute path to the saved image file.
        """
        from modules.output import generate_temp_filename, save_image

        _date_string, filepath, _filename = generate_temp_filename(
            folder=self.output_dir,
            extension=task.output_format,
        )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if task.save_metadata_to_images:
            metadata = [
                ("Prompt", "prompt", task.prompt),
                ("Steps", "steps", str(task.effective_steps)),
            ]
            parsed_parameters = f"{task.prompt}\nSteps: {task.effective_steps}"
        else:
            metadata = []
            parsed_parameters = ""

        save_image(
            image=image,
            filepath=filepath,
            output_format=task.output_format,
            metadata=metadata,
            parsed_parameters=parsed_parameters,
        )

        return filepath


# ---------------------------------------------------------------------------
# Stub diffusion — placeholder until real backend integration
# ---------------------------------------------------------------------------


def _generate_stub_image(width: int, height: int, seed: int) -> Image:
    """Generate a solid-color placeholder image.

    Uses the seed to deterministically pick an RGB color so tests
    are reproducible. This function will be replaced by real diffusion
    in a future task.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        seed: Random seed for deterministic color selection.

    Returns:
        A PIL Image with a solid color.
    """
    from PIL import Image

    rng = random.Random(seed)  # noqa: S311  # nosec B311
    color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    return Image.new("RGB", (width, height), color)
