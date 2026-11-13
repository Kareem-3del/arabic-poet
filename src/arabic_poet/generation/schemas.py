"""Request/result schemas for poem generation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from arabic_poet.config import SamplingConfig
from arabic_poet.poetry.meter import canonical_meter

Style = Literal["classical", "tafeela", "prose_poem"]
Variety = Literal["fusha", "egyptian", "levantine", "gulf", "iraqi", "maghrebi"]
#: Period whose manner the poem follows (by the poet's death year in the training corpus).
Era = Literal["jahili", "early_islamic", "abbasid", "late", "modern"]


class PoemRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(..., min_length=1)
    style: Style = "classical"
    meter: str | None = Field(None, description="Canonical meter key, e.g. 'tawil'.")
    rhyme: str | None = Field(None, description="Rawi letter (e.g. 'ر') or a rhyme word.")
    line_count: int = Field(8, ge=1, le=60)
    mood: str | None = None
    era: Era | None = None
    plan: str | None = Field(None, description="A plan written beforehand (strategy 'preplanned').")
    language: Variety = "fusha"
    strategy: str = "direct"
    seed: int = 0
    sampling: SamplingConfig | None = None

    @field_validator("meter")
    @classmethod
    def _meter(cls, v: str | None) -> str | None:
        if v is None or not v.strip():
            return None
        key = canonical_meter(v)
        if key is None:
            raise ValueError(f"Unknown meter {v!r}; use a classical meter name (e.g. الطويل).")
        return key

    @field_validator("topic")
    @classmethod
    def _topic(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("topic must not be blank")
        return v.strip()

    @field_validator("rhyme", "mood")
    @classmethod
    def _optional_text(cls, v: str | None) -> str | None:
        return (v.strip() or None) if isinstance(v, str) else v


class StageRecord(BaseModel):
    name: str
    messages: list[dict[str, str]]
    output: str
    raw_output: str
    seed: int
    sampling: SamplingConfig
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    latency_s: float | None = None
    generation_tps: float | None = None
    peak_memory_gb: float | None = None


class Candidate(BaseModel):
    index: int
    seed: int
    stages: list[StageRecord]
    raw_poem: str  # final stage output exactly as generated
    poem: str  # parsed units, one bayt/line per line (see poetry/structure.py)
    metrics: dict
    score: float
    score_components: dict[str, float]
    latency_s: float
    completion_tokens: int | None


class PoemResult(BaseModel):
    run_id: str
    created_at: str
    request: PoemRequest
    strategy: str
    candidates: list[Candidate]
    selected: int
    selection: dict
    model: dict
    total_latency_s: float
    total_completion_tokens: int | None

    @property
    def best(self) -> Candidate:
        return self.candidates[self.selected]

    @property
    def poem(self) -> str:
        return self.best.poem

    @property
    def stages(self) -> list[StageRecord]:
        return self.best.stages
