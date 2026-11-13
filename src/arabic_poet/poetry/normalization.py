"""Arabic text normalization, configurable and non-destructive by default.

Raw text is always kept separately by callers; normalization only produces derived views.
Diacritics (tashkeel) are preserved unless ``strip_diacritics`` is explicitly enabled,
because in poetry they carry prosodic and grammatical information.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict

# Harakat, tanween, shadda, sukun, combining maddah/hamza, other combining marks.
DIACRITICS = "".join(chr(c) for c in range(0x064B, 0x0660)) + "ٰ"
QURANIC_MARKS = "".join(chr(c) for c in range(0x06D6, 0x06EE))
TATWEEL = "ـ"
# Zero-width and bidi control characters that silently break matching.
INVISIBLES = "​‌‍‎‏‪‫‬‭‮⁦⁧⁨⁩﻿"

_DIACRITICS_RE = re.compile(f"[{DIACRITICS}{QURANIC_MARKS}]")
_INVISIBLES_RE = re.compile(f"[{INVISIBLES}]")
_SPACES_RE = re.compile(r"[ \t  - 　]+")

_ALEF_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_HAMZA_CARRIERS = str.maketrans({"أ": "ء", "إ": "ء", "ؤ": "ء", "ئ": "ء"})

ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)


class NormalizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unicode_form: Literal["NFC", "NFKC"] = "NFKC"
    remove_invisibles: bool = True
    remove_tatweel: bool = True
    collapse_whitespace: bool = True
    strip_diacritics: bool = False
    normalize_alef: bool = False
    normalize_alef_maqsura: bool = False
    normalize_teh_marbuta: bool = False


#: Aggressive folding used only for matching (dedup, rhyme keys), never for stored text.
MATCHING = NormalizationConfig(
    strip_diacritics=True,
    normalize_alef=True,
    normalize_alef_maqsura=True,
    normalize_teh_marbuta=True,
)


def normalize(text: str, cfg: NormalizationConfig | None = None) -> str:
    cfg = cfg or NormalizationConfig()
    out = unicodedata.normalize(cfg.unicode_form, text)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    if cfg.remove_invisibles:
        out = _INVISIBLES_RE.sub("", out)
    if cfg.remove_tatweel:
        out = out.replace(TATWEEL, "")
    if cfg.strip_diacritics:
        out = _DIACRITICS_RE.sub("", out)
    if cfg.normalize_alef:
        out = out.translate(_ALEF_VARIANTS)
    if cfg.normalize_alef_maqsura:
        out = out.replace("ى", "ي")
    if cfg.normalize_teh_marbuta:
        out = out.replace("ة", "ه")
    if cfg.collapse_whitespace:
        out = "\n".join(_SPACES_RE.sub(" ", line).strip() for line in out.split("\n"))
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def fold_for_matching(text: str) -> str:
    return normalize(text, MATCHING)


def strip_diacritics(text: str) -> str:
    return _DIACRITICS_RE.sub("", text)


def is_arabic_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ARABIC_RANGES)


def arabic_letter_ratio(text: str) -> float:
    """Share of alphabetic characters that are Arabic script (0.0 when there are none)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if is_arabic_char(c)) / len(letters)


def poem_lines(text: str) -> list[str]:
    """Non-empty, stripped lines of a poem (one bayt or one free-verse line per line)."""
    return [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]


# Hemistich separators seen in practice: "...", "…", "*", "***", tabs, 3+ spaces.
_HEMISTICH_RE = re.compile(r"\s*(?:\.{3,}|…|\*+|\t+| {3,})\s*")


def split_hemistichs(line: str) -> list[str]:
    parts = [p.strip() for p in _HEMISTICH_RE.split(line.strip())]
    return [p for p in parts if p]


def arabic_words(text: str) -> list[str]:
    """Arabic word tokens on folded text (letters only)."""
    folded = fold_for_matching(text)
    return re.findall(r"[ء-يٱ-ۓ]+", folded)
