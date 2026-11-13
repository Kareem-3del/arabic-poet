"""Typed, YAML-backed configuration.

Every behaviour that an experiment might vary lives in a YAML file under ``configs/``.
Paths in YAML are relative to the project root, never machine-specific absolute paths.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- root/paths


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Directory holding ``pyproject.toml`` (or ``$ARABIC_POET_ROOT`` when set)."""
    env = os.environ.get("ARABIC_POET_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise ConfigError(
        "Could not locate the project root (pyproject.toml + configs/). "
        "Run from inside the repository or set ARABIC_POET_ROOT."
    )


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else project_root() / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = resolve_path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")
    with p.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a mapping at top level: {p}")
    return data


# --------------------------------------------------------------------------- models


class ModelSpec(_Strict):
    """A model registered in ``configs/models/*.yaml``."""

    id: str
    hf_repo: str
    revision: str = Field(description="Pinned Hugging Face commit SHA.")
    local_dir: str
    backend: Literal["mlx"] = "mlx"
    # "mlx": repo already in MLX format. "hf": PyTorch/HF weights, converted locally with
    # mlx_lm.convert (quantized to ``convert_q_bits``) into local_dir.
    source_format: Literal["mlx", "hf"] = "mlx"
    convert_q_bits: int | None = 4
    allow_patterns: list[str] | None = None  # restrict which repo files are downloaded
    license: str
    base_model: str | None = None
    quantization: str | None = None
    approx_size_gb: float | None = None
    # Extra variables passed into the tokenizer's chat template (e.g. Qwen3 enable_thinking).
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    # Strip a leading <think>...</think> block from outputs, if the model emits one.
    strip_think_blocks: bool = True
    notes: str = ""

    @field_validator("revision")
    @classmethod
    def _pinned(cls, v: str) -> str:
        if len(v) != 40 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("revision must be a full 40-char commit SHA for reproducibility")
        return v

    @property
    def path(self) -> Path:
        return resolve_path(self.local_dir)


# --------------------------------------------------------------------------- generation


class SamplingConfig(_Strict):
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.8, ge=0.0, le=1.0)
    top_k: int = Field(20, ge=0)
    min_p: float = Field(0.0, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(None, gt=0.0)
    repetition_context_size: int = 64
    max_tokens: int = Field(768, gt=0)


class StageConfig(_Strict):
    """Settings for one stage of a multi-stage strategy."""

    max_tokens: int = Field(..., gt=0)
    temperature: float | None = None
    language: Literal["ar", "en"] = "ar"


class StrategyConfig(_Strict):
    stages: dict[str, StageConfig] = Field(default_factory=dict)


class SelectionConfig(_Strict):
    """Best-of-N: generate ``candidates`` full poems, keep all, select by automatic score."""

    candidates: int = Field(1, ge=1, le=16)
    # Weights of the automatic candidate score (see evaluation/metrics.py::candidate_score).
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "structure": 1.0,
            "count": 1.0,
            "rhyme": 1.0,
            "requested_rhyme": 1.0,
            "meter": 2.0,
            "non_repetition": 1.0,
        }
    )


class GenerationConfig(_Strict):
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    prompt_version: Literal["v1", "v2", "v3", "shaer"] = "v2"
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    strategies: dict[str, StrategyConfig] = Field(default_factory=dict)
    # Hemistich separator the prompts ask for (classical style).
    hemistich_separator: str = " ... "


# --------------------------------------------------------------------------- evaluation


class EvaluationConfig(_Strict):
    benchmark: str
    strategies: list[str] = Field(default_factory=lambda: ["direct"])
    seeds: list[int] = Field(default_factory=lambda: [0])
    limit: int | None = Field(None, gt=0)
    output_dir: str = "artifacts/evaluations"


# --------------------------------------------------------------------------- revision


class RevisionConfig(_Strict):
    """Critique-and-revise pipeline v2 (generation/revise.py). Thresholds are pyarud best-match
    scores, NOT probabilities: on meter_gold_v1 every real bayt scored >= 0.695 and no broken
    or prose line reached 0.90 (docs/REVISION_V2.md)."""

    meter_valid_min: float = Field(
        0.90, gt=0.0, le=1.0, description="score >= this: supported-valid"
    )
    meter_invalid_max: float = Field(
        0.65, gt=0.0, le=1.0, description="score < this on every reading: supported-invalid"
    )
    uncertain_meter_policy: Literal["guarded", "never"] = Field(
        "guarded",
        description="guarded: an uncertain scan may be edited only at hemistich scope and only "
        "if the literary guard confirms the meaning is preserved; never: report only",
    )
    candidates: int = Field(
        4, ge=1, le=12, description="edit attempts per repair, split over scopes"
    )
    rounds: int = Field(3, ge=1, le=6)
    edit_temperature: float = Field(0.8, ge=0.0, le=1.5)
    edit_max_tokens: int = Field(48, gt=8)
    protect_opening: bool = True
    protect_ending: bool = True
    critic_model: str | None = "qwen3-30b-a3b-4bit"
    editor_model: str | None = Field(None, description="None: the model that wrote the draft")
    actionable_min_severity: Literal["low", "medium", "high"] = "medium"
    actionable_min_confidence: Literal["low", "medium", "high"] = "medium"
    max_guards_per_repair: int = Field(2, ge=1, le=8)
    add_missing_verses: bool = True

    @field_validator("meter_invalid_max")
    @classmethod
    def _order(cls, v: float, info) -> float:
        hi = info.data.get("meter_valid_min", 0.9)
        if v >= hi:
            raise ValueError("meter_invalid_max must be below meter_valid_min")
        return v


# --------------------------------------------------------------------------- training


class LoraParameters(_Strict):
    rank: int = Field(16, gt=0)
    scale: float = Field(20.0, gt=0)
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    keys: list[str] | None = None


class TrainingConfig(_Strict):
    """Maps 1:1 onto supported ``mlx_lm lora`` options (validated here, executed there)."""

    name: str
    model: str = Field(description="Model id from the registry.")
    data: str = Field(description="Directory with train/valid/test.jsonl (MLX chat format).")
    fine_tune_type: Literal["lora", "dora"] = "lora"
    mask_prompt: bool = True
    num_layers: int = Field(16, ge=-1)
    batch_size: int = Field(4, gt=0)
    grad_accumulation_steps: int = Field(1, gt=0)
    iters: int = Field(1000, gt=0)
    learning_rate: float = Field(1e-5, gt=0)
    lr_schedule: dict | None = Field(
        None,
        description="MLX-LM schedule, e.g. {name: cosine_decay, warmup: 100, "
        "arguments: [peak_lr, decay_steps, end_lr]}; None = constant learning_rate",
    )
    max_seq_length: int = Field(1024, gt=0)
    lora_parameters: LoraParameters = Field(default_factory=LoraParameters)
    optimizer: Literal["adam", "adamw", "sgd", "adafactor"] = "adamw"
    steps_per_report: int = 10
    steps_per_eval: int = 100
    val_batches: int = 25
    save_every: int = 100
    grad_checkpoint: bool = False
    seed: int = 0
    test: bool = False
    test_batches: int = -1
    output_dir: str = "artifacts/adapters"


# --------------------------------------------------------------------------- app


class PathsConfig(_Strict):
    models_dir: str = "models"
    artifacts_dir: str = "artifacts"
    data_dir: str = "data"
    model_configs: str = "configs/models"


class AppConfig(_Strict):
    default_model: str
    generation_config: str = "configs/generation/default.yaml"
    evaluation_config: str = "configs/evaluation/default.yaml"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    log_level: str = "INFO"


def _validate(cls: type[BaseModel], data: dict[str, Any], source: str) -> Any:
    try:
        return cls.model_validate(data)
    except Exception as exc:  # pydantic.ValidationError, re-raised with the file name
        raise ConfigError(f"Invalid config {source}:\n{exc}") from exc


def load_app_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    return _validate(AppConfig, load_yaml(path), str(path))


def load_generation_config(path: str | Path | None = None) -> GenerationConfig:
    path = path or load_app_config().generation_config
    return _validate(GenerationConfig, load_yaml(path), str(path))


def load_evaluation_config(path: str | Path | None = None) -> EvaluationConfig:
    path = path or load_app_config().evaluation_config
    return _validate(EvaluationConfig, load_yaml(path), str(path))


def load_revision_config(path: str | Path = "configs/revision/default.yaml") -> RevisionConfig:
    return _validate(RevisionConfig, load_yaml(path), str(path))


def load_training_config(path: str | Path) -> TrainingConfig:
    return _validate(TrainingConfig, load_yaml(path), str(path))


def load_model_spec(path: str | Path) -> ModelSpec:
    return _validate(ModelSpec, load_yaml(path), str(path))
