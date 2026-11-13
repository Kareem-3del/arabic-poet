"""An 'arud acceptor over fully diacritized text, for constraining generation.

A hemistich is legal in a meter when its mutaharrik/sakin pattern (pyarud's 'arudi
transcription: 1 = moving letter, 0 = quiescent) is one of the meter's patterns, zihafat and
'ilal included (pyarud's precomputed tables). While a hemistich is being written, its text is
*feasible* when the pattern of its settled part is a prefix of at least one legal pattern.

The settled part leaves out the last letter while a word is unfinished: its vowel may not have
been written yet, and the next letter can change how it is read (shadda, lam of the article).
"""

from __future__ import annotations

import re
from functools import lru_cache

from arabic_poet.poetry.prosody import KEY_TO_PYARUD

MARKS = "ًٌٍَُِّْٰ"
_LETTER = re.compile(r"[ء-ي]")
_ALLOWED = re.compile(rf"^[ء-ي{MARKS} ]*$")


@lru_cache(maxsize=1)
def _converter():
    from pyarud.core.phonetics import ArudiConverter

    return ArudiConverter()


@lru_cache(maxsize=64)
def legal_patterns(meter: str, half: str) -> frozenset[str]:
    """Legal patterns of one hemistich (``half`` is 'sadr' or 'ajuz') of a meter key."""
    from pyarud.processor import _init_global_patterns

    _, sadr, ajuz = _init_global_patterns()
    table = (sadr if half == "sadr" else ajuz)[KEY_TO_PYARUD[meter]]
    return frozenset(table)


@lru_cache(maxsize=64)
def legal_prefixes(meter: str, half: str) -> frozenset[str]:
    return frozenset(p[:i] for p in legal_patterns(meter, half) for i in range(len(p) + 1))


# Words written without a long alif that is pronounced, which the converter does not know.
_UNWRITTEN_ALIF = (
    (re.compile("إِلَه"), "إِلَاه"),
    (re.compile("رَّحْمَن"), "رَّحْمَان"),
    (re.compile("هَؤُلَا"), "هَاؤُلَا"),
    (re.compile("دَاوُد"), "دَاوُود"),
)


@lru_cache(maxsize=200_000)
def pattern(text: str, final: bool) -> str:
    """Pattern of ``text``; ``final`` applies end-of-hemistich saturation (ishba')."""
    for rx, full in _UNWRITTEN_ALIF:
        text = rx.sub(full, text)
    try:
        return _converter().prepare_text(text, saturate=final)[1]
    except Exception:
        return "?"


def settled(text: str) -> str:
    """The part of an unfinished hemistich whose reading can no longer change."""
    if not text or text.endswith(" "):
        return text.rstrip()
    i = len(text) - 1
    while i >= 0 and text[i] in MARKS:
        i -= 1
    return text[:i].rstrip() if i >= 0 else ""


_VOWELS = MARKS.replace("\u0651", "")
_TANWEEN = "\u064b\u064c\u064d"
_RUN = re.compile(rf"(^|[^{MARKS}])([{MARKS}]+)")
# A tanween closes its word; only a bare alif (or alif maqsura) may follow it.
_TANWEEN_MID = re.compile(rf"[{_TANWEEN}]\u0651?(?:[ء-ي](?<![اى])|[اى][ء-ي{MARKS}])")
# An alif of prolongation (after a fatha) and an alif maqsura carry no short vowel or sukun.
_VOWELLED_ALIF = re.compile("\u064e\u0651?ا[\u064e-\u0650\u0652]|ى[\u064e-\u0650\u0652]")


def _marks_ok(text: str) -> bool:
    """Each letter carries at most one vowel mark and one shadda (never shadda with sukun);
    a mark needs a letter; a word does not open with a doubled or quiescent letter."""
    for m in _RUN.finditer(text):
        base, run = m.group(1), m.group(2)
        if base in ("", " "):
            return False
        shadda = run.count("\u0651")
        if shadda > 1 or len(run) - shadda > 1 or (shadda and "\u0652" in run):
            return False
        start = m.start(1)
        if (start == 0 or text[start - 1] == " ") and (shadda or "\u0652" in run):
            return False
    return not _TANWEEN_MID.search(text) and not _VOWELLED_ALIF.search(text)


def well_formed(text: str) -> bool:
    return bool(_ALLOWED.match(text)) and "  " not in text and _marks_ok(text)


def feasible(text: str, meter: str, half: str) -> bool:
    """Can ``text`` still be continued into a legal hemistich?"""
    if not well_formed(text):
        return False
    return pattern(settled(text), False) in legal_prefixes(meter, half)


def complete(text: str, meter: str, half: str) -> bool:
    """Is ``text`` a whole legal hemistich?"""
    text = text.strip()
    if not well_formed(text) or len(_LETTER.findall(text)) < 6:
        return False
    return pattern(text, True) in legal_patterns(meter, half)


def fully_vowelled(text: str, min_share: float = 0.7) -> bool:
    """Most letters carry a mark (long vowels and the article's alif/lam legitimately do not)."""
    letters = [m.start() for m in _LETTER.finditer(text)]
    if not letters:
        return False
    marked = sum(1 for i in letters if i + 1 < len(text) and text[i + 1] in MARKS)
    return marked / len(letters) >= min_share
