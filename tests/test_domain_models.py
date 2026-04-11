"""Unit tests for domain models — UNF-31 acceptance criteria.

RED phase: all tests must FAIL before implementation exists.

AC1: modules/domain/models.py exists with CheckpointMetadata, LoRAConfig,
     SamplerConfig, PipelineConfig, GenerationResult, DiffusionProgress
AC2: All domain models are frozen dataclasses with slots=True
AC3: All domain models have __post_init__ validation for invariants
AC4: SamplerConfig rejects sampler names not in the known sampler list
AC5: LoRAConfig rejects weights outside the configured min/max range
AC6: PipelineConfig.from_task(task) extracts generation parameters from AsyncTask
AC7: Domain models have zero imports from torch, ldm_patched, PIL, or infrastructure
AC8: Unit tests cover construction, validation, equality, and factory methods
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

# ===========================================================================
# AC1: modules/domain/models.py exists with all required classes
# ===========================================================================


class TestModuleExistsWithRequiredClasses:
    """AC1: Domain models module exists and exports required names."""

    def test_models_module_importable(self) -> None:
        from modules.domain import models  # noqa: F401

    def test_checkpoint_metadata_exists(self) -> None:
        from modules.domain.models import CheckpointMetadata

        assert CheckpointMetadata is not None

    def test_lora_config_exists(self) -> None:
        from modules.domain.models import LoRAConfig

        assert LoRAConfig is not None

    def test_sampler_config_exists(self) -> None:
        from modules.domain.models import SamplerConfig

        assert SamplerConfig is not None

    def test_pipeline_config_exists(self) -> None:
        from modules.domain.models import PipelineConfig

        assert PipelineConfig is not None

    def test_generation_result_exists(self) -> None:
        from modules.domain.models import GenerationResult

        assert GenerationResult is not None

    def test_diffusion_progress_exists(self) -> None:
        from modules.domain.models import DiffusionProgress

        assert DiffusionProgress is not None


# ===========================================================================
# AC2: All domain models are frozen dataclasses with slots=True
# ===========================================================================


_DOMAIN_MODEL_NAMES = [
    "CheckpointMetadata",
    "LoRAConfig",
    "SamplerConfig",
    "PipelineConfig",
    "GenerationResult",
    "DiffusionProgress",
]


class TestFrozenDataclassesWithSlots:
    """AC2: All domain models are frozen dataclasses with slots=True."""

    @pytest.mark.parametrize("class_name", _DOMAIN_MODEL_NAMES)
    def test_is_dataclass(self, class_name: str) -> None:
        import modules.domain.models as m

        cls = getattr(m, class_name)
        assert dataclasses.is_dataclass(cls), f"{class_name} must be a dataclass"

    @pytest.mark.parametrize("class_name", _DOMAIN_MODEL_NAMES)
    def test_has_slots(self, class_name: str) -> None:
        import modules.domain.models as m

        cls = getattr(m, class_name)
        assert hasattr(cls, "__slots__"), f"{class_name} must use slots=True"

    @pytest.mark.parametrize("class_name", _DOMAIN_MODEL_NAMES)
    def test_is_frozen(self, class_name: str) -> None:
        import modules.domain.models as m

        cls = getattr(m, class_name)
        params = dataclasses.fields(cls)
        instance = _make_minimal_instance(cls)
        first_field = params[0].name
        with pytest.raises(AttributeError):
            setattr(instance, first_field, "SHOULD_FAIL")


# ===========================================================================
# AC3: __post_init__ validation for invariants
# ===========================================================================


class TestCheckpointMetadataConstruction:
    """AC3/AC8: CheckpointMetadata construction and validation."""

    def test_valid_construction(self) -> None:
        from modules.domain.models import CheckpointMetadata

        meta = CheckpointMetadata(
            filename="juggernaut.safetensors",
            file_path="/models/checkpoints/juggernaut.safetensors",
            is_sdxl=True,
        )
        assert meta.filename == "juggernaut.safetensors"
        assert meta.file_path == "/models/checkpoints/juggernaut.safetensors"
        assert meta.is_sdxl is True

    def test_equality(self) -> None:
        from modules.domain.models import CheckpointMetadata

        a = CheckpointMetadata(filename="a.safetensors", file_path="/a", is_sdxl=True)
        b = CheckpointMetadata(filename="a.safetensors", file_path="/a", is_sdxl=True)
        assert a == b

    def test_inequality(self) -> None:
        from modules.domain.models import CheckpointMetadata

        a = CheckpointMetadata(filename="a.safetensors", file_path="/a", is_sdxl=True)
        b = CheckpointMetadata(filename="b.safetensors", file_path="/b", is_sdxl=True)
        assert a != b

    def test_immutability(self) -> None:
        from modules.domain.models import CheckpointMetadata

        meta = CheckpointMetadata(filename="a.safetensors", file_path="/a", is_sdxl=True)
        with pytest.raises(AttributeError):
            meta.filename = "other.safetensors"  # type: ignore[misc]

    def test_rejects_empty_filename(self) -> None:
        from modules.domain.models import CheckpointMetadata

        with pytest.raises(ValueError, match="filename"):
            CheckpointMetadata(filename="", file_path="/a", is_sdxl=True)

    def test_rejects_empty_file_path(self) -> None:
        from modules.domain.models import CheckpointMetadata

        with pytest.raises(ValueError, match="file_path"):
            CheckpointMetadata(filename="a.safetensors", file_path="", is_sdxl=True)


# ===========================================================================
# AC5: LoRAConfig rejects weights outside min/max range
# ===========================================================================


class TestLoRAConfigConstruction:
    """AC3/AC5/AC8: LoRAConfig construction and validation."""

    def test_valid_construction(self) -> None:
        from modules.domain.models import LoRAConfig

        lora = LoRAConfig(filename="detail.safetensors", weight=0.8)
        assert lora.filename == "detail.safetensors"
        assert lora.weight == pytest.approx(0.8)

    def test_equality(self) -> None:
        from modules.domain.models import LoRAConfig

        a = LoRAConfig(filename="a.safetensors", weight=0.5)
        b = LoRAConfig(filename="a.safetensors", weight=0.5)
        assert a == b

    def test_immutability(self) -> None:
        from modules.domain.models import LoRAConfig

        lora = LoRAConfig(filename="a.safetensors", weight=0.5)
        with pytest.raises(AttributeError):
            lora.weight = 0.9  # type: ignore[misc]

    def test_rejects_empty_filename(self) -> None:
        from modules.domain.models import LoRAConfig

        with pytest.raises(ValueError, match="filename"):
            LoRAConfig(filename="", weight=0.5)

    def test_rejects_weight_above_max(self) -> None:
        from modules.domain.models import LoRAConfig

        with pytest.raises(ValueError, match="weight"):
            LoRAConfig(filename="a.safetensors", weight=3.0)

    def test_rejects_weight_below_min(self) -> None:
        from modules.domain.models import LoRAConfig

        with pytest.raises(ValueError, match="weight"):
            LoRAConfig(filename="a.safetensors", weight=-3.0)

    def test_accepts_weight_at_min_boundary(self) -> None:
        from modules.domain.models import LoRAConfig

        lora = LoRAConfig(filename="a.safetensors", weight=-2.0)
        assert lora.weight == pytest.approx(-2.0)

    def test_accepts_weight_at_max_boundary(self) -> None:
        from modules.domain.models import LoRAConfig

        lora = LoRAConfig(filename="a.safetensors", weight=2.0)
        assert lora.weight == pytest.approx(2.0)

    def test_custom_weight_range(self) -> None:
        from modules.domain.models import LoRAConfig

        lora = LoRAConfig(filename="a.safetensors", weight=0.5, min_weight=-1.0, max_weight=1.0)
        assert lora.weight == pytest.approx(0.5)

    def test_custom_weight_range_rejects_out_of_range(self) -> None:
        from modules.domain.models import LoRAConfig

        with pytest.raises(ValueError, match="weight"):
            LoRAConfig(filename="a.safetensors", weight=1.5, min_weight=-1.0, max_weight=1.0)


# ===========================================================================
# AC4: SamplerConfig rejects sampler names not in the known list
# ===========================================================================


class TestSamplerConfigConstruction:
    """AC3/AC4/AC8: SamplerConfig construction and validation."""

    def test_valid_construction(self) -> None:
        from modules.domain.models import SamplerConfig

        sc = SamplerConfig(
            sampler_name="euler",
            scheduler_name="normal",
            steps=30,
            cfg_scale=7.0,
            seed=42,
        )
        assert sc.sampler_name == "euler"
        assert sc.scheduler_name == "normal"
        assert sc.steps == 30
        assert sc.cfg_scale == pytest.approx(7.0)
        assert sc.seed == 42

    def test_equality(self) -> None:
        from modules.domain.models import SamplerConfig

        a = SamplerConfig(sampler_name="euler", scheduler_name="normal", steps=30, cfg_scale=7.0, seed=42)
        b = SamplerConfig(sampler_name="euler", scheduler_name="normal", steps=30, cfg_scale=7.0, seed=42)
        assert a == b

    def test_immutability(self) -> None:
        from modules.domain.models import SamplerConfig

        sc = SamplerConfig(sampler_name="euler", scheduler_name="normal", steps=30, cfg_scale=7.0, seed=42)
        with pytest.raises(AttributeError):
            sc.steps = 50  # type: ignore[misc]

    def test_rejects_unknown_sampler_name(self) -> None:
        from modules.domain.models import SamplerConfig

        with pytest.raises(ValueError, match="sampler_name"):
            SamplerConfig(sampler_name="not_a_real_sampler", scheduler_name="normal", steps=30, cfg_scale=7.0, seed=42)

    def test_rejects_unknown_scheduler_name(self) -> None:
        from modules.domain.models import SamplerConfig

        with pytest.raises(ValueError, match="scheduler_name"):
            SamplerConfig(sampler_name="euler", scheduler_name="not_a_scheduler", steps=30, cfg_scale=7.0, seed=42)

    def test_rejects_zero_steps(self) -> None:
        from modules.domain.models import SamplerConfig

        with pytest.raises(ValueError, match="steps"):
            SamplerConfig(sampler_name="euler", scheduler_name="normal", steps=0, cfg_scale=7.0, seed=42)

    def test_rejects_negative_steps(self) -> None:
        from modules.domain.models import SamplerConfig

        with pytest.raises(ValueError, match="steps"):
            SamplerConfig(sampler_name="euler", scheduler_name="normal", steps=-5, cfg_scale=7.0, seed=42)

    def test_rejects_negative_cfg_scale(self) -> None:
        from modules.domain.models import SamplerConfig

        with pytest.raises(ValueError, match="cfg_scale"):
            SamplerConfig(sampler_name="euler", scheduler_name="normal", steps=30, cfg_scale=-1.0, seed=42)

    def test_accepts_all_known_samplers(self) -> None:
        from modules.domain.models import SamplerConfig
        from modules.flags import SAMPLER_NAMES

        for name in SAMPLER_NAMES:
            sc = SamplerConfig(sampler_name=name, scheduler_name="normal", steps=20, cfg_scale=7.0, seed=1)
            assert sc.sampler_name == name

    def test_accepts_all_known_schedulers(self) -> None:
        from modules.domain.models import SamplerConfig
        from modules.flags import SCHEDULER_NAMES

        for name in SCHEDULER_NAMES:
            sc = SamplerConfig(sampler_name="euler", scheduler_name=name, steps=20, cfg_scale=7.0, seed=1)
            assert sc.scheduler_name == name


# ===========================================================================
# AC8: PipelineConfig construction and defaults
# ===========================================================================


class TestPipelineConfigConstruction:
    """AC3/AC8: PipelineConfig construction, defaults, and validation."""

    def test_valid_construction(self) -> None:
        from modules.domain.models import CheckpointMetadata, PipelineConfig, SamplerConfig

        checkpoint = CheckpointMetadata(
            filename="model.safetensors", file_path="/models/model.safetensors", is_sdxl=True
        )
        sampler = SamplerConfig(sampler_name="euler", scheduler_name="normal", steps=30, cfg_scale=7.0, seed=42)

        config = PipelineConfig(
            checkpoint=checkpoint,
            loras=[],
            sampler=sampler,
            width=1024,
            height=1024,
            positive_prompt="a cat",
            negative_prompt="ugly",
            clip_skip=1,
        )
        assert config.checkpoint == checkpoint
        assert config.sampler == sampler
        assert config.width == 1024
        assert config.height == 1024

    def test_default_refiner_is_none(self) -> None:
        config = _make_pipeline_config()
        assert config.refiner is None

    def test_default_freeu_disabled(self) -> None:
        config = _make_pipeline_config()
        assert config.freeu_enabled is False

    def test_rejects_non_positive_width(self) -> None:
        with pytest.raises(ValueError, match="width"):
            _make_pipeline_config(width=0)

    def test_rejects_non_positive_height(self) -> None:
        with pytest.raises(ValueError, match="height"):
            _make_pipeline_config(height=-1)

    def test_equality(self) -> None:
        a = _make_pipeline_config(width=1024, height=1024)
        b = _make_pipeline_config(width=1024, height=1024)
        assert a == b

    def test_inequality_on_width(self) -> None:
        a = _make_pipeline_config(width=1024)
        b = _make_pipeline_config(width=768)
        assert a != b

    def test_immutability(self) -> None:
        config = _make_pipeline_config()
        with pytest.raises(AttributeError):
            config.width = 512  # type: ignore[misc]

    def test_rejects_negative_clip_skip(self) -> None:
        with pytest.raises(ValueError, match="clip_skip"):
            _make_pipeline_config(clip_skip=-1)

    def test_rejects_refiner_switch_above_one(self) -> None:
        from modules.domain.models import CheckpointMetadata

        refiner = CheckpointMetadata(filename="refiner.safetensors", file_path="/refiner", is_sdxl=True)
        with pytest.raises(ValueError, match="refiner_switch"):
            _make_pipeline_config(refiner=refiner, refiner_switch=1.5)

    def test_rejects_refiner_switch_below_zero(self) -> None:
        from modules.domain.models import CheckpointMetadata

        refiner = CheckpointMetadata(filename="refiner.safetensors", file_path="/refiner", is_sdxl=True)
        with pytest.raises(ValueError, match="refiner_switch"):
            _make_pipeline_config(refiner=refiner, refiner_switch=-0.1)

    def test_refiner_switch_without_refiner_skips_validation(self) -> None:
        config = _make_pipeline_config(refiner=None, refiner_switch=5.0)
        assert config.refiner_switch == pytest.approx(5.0)

    def test_loras_stored_as_tuple(self) -> None:
        from modules.domain.models import LoRAConfig

        lora = LoRAConfig(filename="detail.safetensors", weight=0.8)
        config = _make_pipeline_config(loras=[lora])
        assert isinstance(config.loras, tuple)

    def test_image_paths_stored_as_tuple(self) -> None:
        from modules.domain.models import GenerationResult

        result = GenerationResult(image_paths=["/a.png", "/b.png"], elapsed_time=1.0, seed_used=1)
        assert isinstance(result.image_paths, tuple)


# ===========================================================================
# AC6: PipelineConfig.from_task(task) factory
# ===========================================================================


class TestPipelineConfigFromTask:
    """AC6: PipelineConfig.from_task extracts params from AsyncTask."""

    def test_extracts_checkpoint_basename_from_path(self) -> None:
        from modules.domain.models import PipelineConfig

        task = _make_fake_task(base_model_name="/models/checkpoints/juggernaut.safetensors")
        config = PipelineConfig.from_task(task)
        assert config.checkpoint.filename == "juggernaut.safetensors"
        assert config.checkpoint.file_path == "/models/checkpoints/juggernaut.safetensors"

    def test_extracts_checkpoint_name_when_bare(self) -> None:
        from modules.domain.models import PipelineConfig

        task = _make_fake_task(base_model_name="juggernaut.safetensors")
        config = PipelineConfig.from_task(task)
        assert config.checkpoint.filename == "juggernaut.safetensors"

    def test_extracts_prompt(self) -> None:
        from modules.domain.models import PipelineConfig

        task = _make_fake_task(prompt="a photo of a cat")
        config = PipelineConfig.from_task(task)
        assert config.positive_prompt == "a photo of a cat"

    def test_extracts_negative_prompt(self) -> None:
        from modules.domain.models import PipelineConfig

        task = _make_fake_task(negative_prompt="ugly, bad")
        config = PipelineConfig.from_task(task)
        assert config.negative_prompt == "ugly, bad"

    def test_extracts_sampler_config(self) -> None:
        from modules.domain.models import PipelineConfig

        task = _make_fake_task(sampler_name="dpmpp_2m_sde_gpu", scheduler_name="karras")
        config = PipelineConfig.from_task(task)
        assert config.sampler.sampler_name == "dpmpp_2m_sde_gpu"
        assert config.sampler.scheduler_name == "karras"

    def test_extracts_steps_from_effective_steps(self) -> None:
        from modules.domain.models import PipelineConfig

        task = _make_fake_task(effective_steps=45)
        config = PipelineConfig.from_task(task)
        assert config.sampler.steps == 45

    def test_extracts_cfg_scale(self) -> None:
        from modules.domain.models import PipelineConfig

        task = _make_fake_task(cfg_scale=4.0)
        config = PipelineConfig.from_task(task)
        assert config.sampler.cfg_scale == pytest.approx(4.0)

    def test_extracts_seed(self) -> None:
        from modules.domain.models import PipelineConfig

        task = _make_fake_task(seed=12345)
        config = PipelineConfig.from_task(task)
        assert config.sampler.seed == 12345

    def test_extracts_dimensions(self) -> None:
        from modules.domain.models import PipelineConfig

        task = _make_fake_task(width=768, height=1024)
        config = PipelineConfig.from_task(task)
        assert config.width == 768
        assert config.height == 1024

    def test_extracts_loras(self) -> None:
        from modules.domain.models import PipelineConfig

        task = _make_fake_task(loras=[("detail.safetensors", 0.8), ("style.safetensors", 0.5)])
        config = PipelineConfig.from_task(task)
        assert len(config.loras) == 2
        assert config.loras[0].filename == "detail.safetensors"
        assert config.loras[0].weight == pytest.approx(0.8)

    def test_extracts_clip_skip(self) -> None:
        from modules.domain.models import PipelineConfig

        task = _make_fake_task(clip_skip=2)
        config = PipelineConfig.from_task(task)
        assert config.clip_skip == 2

    def test_extracts_freeu_params(self) -> None:
        from modules.domain.models import PipelineConfig

        task = _make_fake_task(freeu_enabled=True, freeu_b1=1.3, freeu_b2=1.4, freeu_s1=0.9, freeu_s2=0.2)
        config = PipelineConfig.from_task(task)
        assert config.freeu_enabled is True
        assert config.freeu_b1 == pytest.approx(1.3)


# ===========================================================================
# AC8: GenerationResult
# ===========================================================================


class TestGenerationResultConstruction:
    """AC3/AC8: GenerationResult construction and validation."""

    def test_valid_construction(self) -> None:
        from modules.domain.models import GenerationResult

        result = GenerationResult(
            image_paths=["/output/img_001.png"],
            elapsed_time=2.5,
            seed_used=42,
        )
        assert result.image_paths == ("/output/img_001.png",)
        assert result.elapsed_time == pytest.approx(2.5)
        assert result.seed_used == 42

    def test_equality(self) -> None:
        from modules.domain.models import GenerationResult

        a = GenerationResult(image_paths=("/a.png",), elapsed_time=1.0, seed_used=1)
        b = GenerationResult(image_paths=("/a.png",), elapsed_time=1.0, seed_used=1)
        assert a == b

    def test_immutability(self) -> None:
        from modules.domain.models import GenerationResult

        result = GenerationResult(image_paths=("/a.png",), elapsed_time=1.0, seed_used=1)
        with pytest.raises(AttributeError):
            result.seed_used = 99  # type: ignore[misc]

    def test_rejects_negative_elapsed_time(self) -> None:
        from modules.domain.models import GenerationResult

        with pytest.raises(ValueError, match="elapsed_time"):
            GenerationResult(image_paths=("/a.png",), elapsed_time=-1.0, seed_used=1)


# ===========================================================================
# AC8: DiffusionProgress
# ===========================================================================


class TestDiffusionProgressConstruction:
    """AC3/AC8: DiffusionProgress construction and validation."""

    def test_valid_construction(self) -> None:
        from modules.domain.models import DiffusionProgress

        progress = DiffusionProgress(step=5, total_steps=20)
        assert progress.step == 5
        assert progress.total_steps == 20

    def test_percentage_computed(self) -> None:
        from modules.domain.models import DiffusionProgress

        progress = DiffusionProgress(step=10, total_steps=20)
        assert progress.percentage == pytest.approx(50.0)

    def test_percentage_zero_steps(self) -> None:
        from modules.domain.models import DiffusionProgress

        progress = DiffusionProgress(step=0, total_steps=20)
        assert progress.percentage == pytest.approx(0.0)

    def test_preview_image_default_none(self) -> None:
        from modules.domain.models import DiffusionProgress

        progress = DiffusionProgress(step=1, total_steps=10)
        assert progress.preview_image is None

    def test_preview_image_set(self) -> None:
        from modules.domain.models import DiffusionProgress

        sentinel = object()
        progress = DiffusionProgress(step=1, total_steps=10, preview_image=sentinel)
        assert progress.preview_image is sentinel

    def test_equality(self) -> None:
        from modules.domain.models import DiffusionProgress

        a = DiffusionProgress(step=5, total_steps=20)
        b = DiffusionProgress(step=5, total_steps=20)
        assert a == b

    def test_immutability(self) -> None:
        from modules.domain.models import DiffusionProgress

        progress = DiffusionProgress(step=5, total_steps=20)
        with pytest.raises(AttributeError):
            progress.step = 10  # type: ignore[misc]

    def test_rejects_negative_step(self) -> None:
        from modules.domain.models import DiffusionProgress

        with pytest.raises(ValueError, match="step"):
            DiffusionProgress(step=-1, total_steps=20)

    def test_rejects_step_greater_than_total(self) -> None:
        from modules.domain.models import DiffusionProgress

        with pytest.raises(ValueError, match="step"):
            DiffusionProgress(step=25, total_steps=20)

    def test_rejects_non_positive_total_steps(self) -> None:
        from modules.domain.models import DiffusionProgress

        with pytest.raises(ValueError, match="total_steps"):
            DiffusionProgress(step=0, total_steps=0)


# ===========================================================================
# AC7: Zero imports from torch, ldm_patched, PIL, or infrastructure
# ===========================================================================


class TestNoForbiddenImports:
    """AC7: Domain models have zero imports from forbidden packages."""

    _FORBIDDEN_MODULES: ClassVar[list[str]] = [
        "torch",
        "ldm_patched",
        "PIL",
        "modules.infrastructure",
        "modules.services",
    ]

    def test_no_forbidden_imports(self) -> None:
        source = _get_models_source()
        tree = ast.parse(source)
        base_package = ["modules", "domain"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in self._FORBIDDEN_MODULES:
                        assert not alias.name.startswith(forbidden), (
                            f"models.py must not import {forbidden}, found: import {alias.name}"
                        )
            if isinstance(node, ast.ImportFrom):
                prefix = base_package[: -node.level] if node.level else []
                imported_roots: list[str] = []

                if node.module is not None:
                    imported_roots.append(".".join([*prefix, node.module]))
                else:
                    imported_roots.extend(".".join([*prefix, alias.name]) for alias in node.names)

                for imported in imported_roots:
                    for forbidden in self._FORBIDDEN_MODULES:
                        assert not imported.startswith(forbidden), (
                            f"models.py must not import from {forbidden}, found: from {imported}"
                        )

                if node.module == "modules":
                    forbidden_children = {"infrastructure", "services"}
                    for alias in node.names:
                        assert alias.name not in forbidden_children, (
                            f"models.py must not import modules.{alias.name}, found: from modules import {alias.name}"
                        )


# ===========================================================================
# Helpers
# ===========================================================================


def _get_models_source() -> str:
    """Read source of models.py for AST analysis."""
    path = Path(__file__).resolve().parent.parent / "modules" / "domain" / "models.py"
    return path.read_text()


def _make_pipeline_config(**overrides: Any) -> Any:
    """Build a PipelineConfig with sensible defaults."""
    from modules.domain.models import CheckpointMetadata, PipelineConfig, SamplerConfig

    checkpoint = overrides.pop(
        "checkpoint",
        CheckpointMetadata(filename="model.safetensors", file_path="/models/model.safetensors", is_sdxl=True),
    )
    sampler = overrides.pop(
        "sampler",
        SamplerConfig(sampler_name="euler", scheduler_name="normal", steps=30, cfg_scale=7.0, seed=42),
    )

    defaults: dict[str, Any] = {
        "checkpoint": checkpoint,
        "loras": [],
        "sampler": sampler,
        "width": 1024,
        "height": 1024,
        "positive_prompt": "a cat",
        "negative_prompt": "ugly",
        "clip_skip": 1,
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_fake_task(**overrides: Any) -> SimpleNamespace:
    """Build a fake AsyncTask-like object with sensible defaults."""
    defaults: dict[str, Any] = {
        "base_model_name": "juggernaut.safetensors",
        "prompt": "a photo of a cat",
        "negative_prompt": "ugly",
        "sampler_name": "dpmpp_2m_sde_gpu",
        "scheduler_name": "karras",
        "effective_steps": 30,
        "cfg_scale": 4.0,
        "seed": 42,
        "width": 1024,
        "height": 1024,
        "loras": [("sd_xl_offset_example-lora_1.0.safetensors", 0.1)],
        "clip_skip": 1,
        "freeu_enabled": False,
        "freeu_b1": 1.3,
        "freeu_b2": 1.4,
        "freeu_s1": 0.9,
        "freeu_s2": 0.2,
        "refiner_model_name": "None",
        "refiner_switch": 0.5,
        "disable_seed_increment": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_minimal_instance(cls: type) -> Any:
    """Create a minimal instance of a dataclass for immutability testing."""
    import modules.domain.models as m

    if cls is m.CheckpointMetadata:
        return cls(filename="a.safetensors", file_path="/a", is_sdxl=True)
    if cls is m.LoRAConfig:
        return cls(filename="a.safetensors", weight=0.5)
    if cls is m.SamplerConfig:
        return cls(sampler_name="euler", scheduler_name="normal", steps=20, cfg_scale=7.0, seed=1)
    if cls is m.PipelineConfig:
        return _make_pipeline_config()
    if cls is m.GenerationResult:
        return cls(image_paths=("/a.png",), elapsed_time=1.0, seed_used=1)
    if cls is m.DiffusionProgress:
        return cls(step=1, total_steps=10)
    raise ValueError(f"Unknown class: {cls}")
