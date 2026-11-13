"""Vocabulary acceptor for constrained decoding: only words attested in the corpus.

The meter constraint alone lets the model satisfy a pattern with non-words. This acceptor
answers, for a line being written, "is every finished word a known word, and can the word in
progress still become one?". A word is known if its undiacritized, normalized form (or that
form minus one common proclitic) occurs at least ``min_count`` times in the corpus. It checks
spelling only: a real word with the wrong vowels or in the wrong place still passes.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache

from arabic_poet.config import resolve_path
from arabic_poet.poetry.normalization import strip_diacritics

PREFIXES = ("وال", "بال", "فال", "كال", "ولل", "فلل", "لل", "ال", "و", "ف", "ب", "ل", "ك")
_WORD = re.compile(r"[ء-ي]+")
_ALEF = re.compile("[أإآٱ]")
_BARE = frozenset({"ال", "وال", "بال", "فال", "كال", "لل", "ولل", "فلل"})
# a bare long vowel cannot follow a sukun (a vowelled alif, as in الْاِنتظار, can)
_BAD_MARKS = re.compile("ْ[اى](?=[^\u064b-\u0652])")


def norm(word: str) -> str:
    w = _ALEF.sub("ا", strip_diacritics(word).replace("ـ", ""))
    return w.replace("ى", "ي").replace("ة", "ه")


class Lexicon:
    def __init__(self, forms: Iterable[str]):
        self.words = frozenset(forms)
        self.prefixes = frozenset(w[:i] for w in self.words for i in range(1, len(w) + 1))

    def known(self, word: str) -> bool:
        w = norm(word)
        if len(w) < 2 or w in _BARE:
            return False  # a lone letter or a proclitic left on its own is a fragment
        if w in self.words:
            return True
        return any(w.startswith(p) and w[len(p):] in self.words for p in PREFIXES)

    def can_grow(self, part: str) -> bool:
        """True if ``part`` is the beginning of some known word (with or without a proclitic)."""
        w = norm(part)
        if w in self.prefixes:
            return True
        return any(
            (w.startswith(p) and w[len(p):] in self.prefixes) or p.startswith(w) for p in PREFIXES
        )

    def feasible(self, text: str, final: bool = False) -> bool:
        """Every finished word of ``text`` is known; the last may be unfinished unless ``final``."""
        if _BAD_MARKS.search(text + (" " if final else "")):
            return False
        words = _WORD.findall(strip_diacritics(text))
        if not words:
            return True
        open_last = not final and not text[-1:].isspace()
        done = words[:-1] if open_last else words
        if not all(self.known(w) for w in done):
            return False
        return not open_last or self.can_grow(words[-1])


@lru_cache(maxsize=2)
def corpus_lexicon(min_count: int = 2) -> Lexicon:
    counts: Counter[str] = Counter()
    for f in resolve_path("data/processed/wikisource-pd-v1/records").glob("*.jsonl"):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                counts.update(norm(w) for w in _WORD.findall(json.loads(line)["text_normalized"]))
    return Lexicon(w for w, c in counts.items() if c >= min_count)
