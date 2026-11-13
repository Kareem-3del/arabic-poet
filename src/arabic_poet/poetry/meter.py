"""Arabic meter (بحر) names and the validator extension point.

Meter validation is NOT implemented. Reliable 'arud analysis needs full vocalisation
(including unwritten case endings and ishba'), syllabification into watad/sabab units,
and the permitted zihaf/'illa variations of each meter. A regex over undiacritized text
cannot do this, so the default validator reports ``not_implemented`` instead of a score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from arabic_poet.poetry.normalization import fold_for_matching

#: The sixteen classical meters (al-Khalil's fifteen plus al-Akhfash's mutadarik).
CLASSICAL_METERS: dict[str, dict[str, str]] = {
    "tawil": {"ar": "الطويل", "pattern": "فعولن مفاعيلن فعولن مفاعيلن"},
    "madid": {"ar": "المديد", "pattern": "فاعلاتن فاعلن فاعلاتن"},
    "basit": {"ar": "البسيط", "pattern": "مستفعلن فاعلن مستفعلن فاعلن"},
    "wafir": {"ar": "الوافر", "pattern": "مفاعلتن مفاعلتن فعولن"},
    "kamil": {"ar": "الكامل", "pattern": "متفاعلن متفاعلن متفاعلن"},
    "hazaj": {"ar": "الهزج", "pattern": "مفاعيلن مفاعيلن"},
    "rajaz": {"ar": "الرجز", "pattern": "مستفعلن مستفعلن مستفعلن"},
    "ramal": {"ar": "الرمل", "pattern": "فاعلاتن فاعلاتن فاعلاتن"},
    "sari": {"ar": "السريع", "pattern": "مستفعلن مستفعلن مفعولات"},
    "munsarih": {"ar": "المنسرح", "pattern": "مستفعلن مفعولات مستفعلن"},
    "khafif": {"ar": "الخفيف", "pattern": "فاعلاتن مستفع لن فاعلاتن"},
    "mudari": {"ar": "المضارع", "pattern": "مفاعيلن فاع لاتن"},
    "muqtadab": {"ar": "المقتضب", "pattern": "مفعولات مستفعلن"},
    "mujtath": {"ar": "المجتث", "pattern": "مستفع لن فاعلاتن"},
    "mutaqarib": {"ar": "المتقارب", "pattern": "فعولن فعولن فعولن فعولن"},
    "mutadarik": {"ar": "المتدارك", "pattern": "فاعلن فاعلن فاعلن فاعلن"},
}

_LOOKUP: dict[str, str] = {}
for _key, _info in CLASSICAL_METERS.items():
    _ar = fold_for_matching(_info["ar"])
    for _alias in (
        _key,
        _ar,
        _ar.removeprefix("ال"),
        f"بحر {_ar}",
        f"بحر {_ar.removeprefix('ال')}",
    ):
        _LOOKUP[_alias] = _key


def canonical_meter(name: str) -> str | None:
    """Map 'الطويل' / 'طويل' / 'بحر الطويل' / 'tawil' to the canonical key, else None."""
    key = fold_for_matching(name).strip().lower()
    return _LOOKUP.get(key)


def meter_arabic_name(key: str) -> str:
    return CLASSICAL_METERS[key]["ar"]


@dataclass(frozen=True)
class MeterResult:
    status: Literal["not_implemented", "ok", "error"]
    method: str
    requested_meter: str | None = None
    detected_meter: str | None = None  # dominant label across abyat
    consistency: float | None = None  # share of abyat with the dominant label
    requested_match: float | None = None  # share of abyat labelled as the requested meter
    per_line: list[dict] = field(default_factory=list)
    note: str = ""
    metrical_rate: float | None = None  # share of abyat judged metrical (any meter)
    mean_score: float | None = None  # mean best-match score over scannable abyat
    scannable_rate: float | None = None  # share of abyat with two hemistichs to scan
    #: Stricter counts (score >= STRICT_THRESHOLD; see docs/EVALUATION.md, calibration v2):
    #: the lenient rule passes 20-28% of abyat with one word removed or replaced, the strict
    #: rule 4-5%, at the cost of missing most real abyat. A repeated bayt counts once.
    strict_rate: float | None = None
    strict_requested_match: float | None = None
    strict_distinct_rate: float | None = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class MeterValidator(Protocol):
    name: str

    def validate(self, abyat: list[str], requested_meter: str | None) -> MeterResult: ...


class UnimplementedMeterValidator:
    """Honest placeholder until a real prosody engine is integrated (see docs/ROADMAP.md)."""

    name = "none"

    def validate(self, abyat: list[str], requested_meter: str | None) -> MeterResult:
        return MeterResult(
            status="not_implemented",
            method=self.name,
            requested_meter=requested_meter,
            note="No prosody engine is integrated; meter adherence is not measured.",
        )


def load_meter_classifier():
    """The project's automatic meter checker (CATT + pyarud); see poetry/prosody.py."""
    from arabic_poet.poetry.prosody import CattPyarudValidator

    return CattPyarudValidator()
