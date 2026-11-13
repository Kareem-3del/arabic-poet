"""Probe: how many training abyat scan in their poem's meter after full diacritization?

Two diacritizations are compared on a sample of SFT training poems:
  catt     CATT alone (source diacritics stripped, as the validator does);
  overlay  CATT, with the source's own diacritics kept wherever the source has them.
Reports the share of abyat pyarud accepts as exactly metrical in the poem's meter, and the speed.

Usage: uv run python scripts/tashkeel_probe.py [n_poems=150]
"""
from __future__ import annotations

import json
import random
import re
import sys
import time

from arabic_poet.config import resolve_path
from arabic_poet.poetry.meter import CLASSICAL_METERS
from arabic_poet.poetry.prosody import CattPyarudValidator

MARKS = "ًٌٍَُِّْ"
SHADDA = "ّ"


def split_marks(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for ch in text:
        if ch in MARKS and out:
            out[-1] = (out[-1][0], out[-1][1] + ch)
        elif ch not in MARKS:
            out.append((ch, ""))
    return out


def overlay(source: str, catt: str) -> str:
    """CATT's diacritics, replaced by the source's where the source marks a letter."""
    s, c = split_marks(source), split_marks(catt)
    if [x for x, _ in s] != [x for x, _ in c]:
        return catt
    res = []
    for (ch, sm), (_, cm) in zip(s, c, strict=True):
        s_v, c_v = sm.replace(SHADDA, ""), cm.replace(SHADDA, "")
        shadda = SHADDA if SHADDA in sm or SHADDA in cm else ""
        res.append(ch + shadda + (s_v or c_v))
    return "".join(res)


def exact(v: CattPyarudValidator, sadr: str, ajuz: str, key: str) -> bool:
    from arabic_poet.poetry.prosody import KEY_TO_PYARUD

    try:
        return bool(v._arud.analyze_verse(sadr, ajuz, forced_meter=KEY_TO_PYARUD.get(key)).is_valid)
    except Exception:
        return False


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    ar_to_key = {v["ar"]: k for k, v in CLASSICAL_METERS.items()}
    rows = [json.loads(x) for x in resolve_path(
        "data/processed/sft-allam-full-v1/train.jsonl").open(encoding="utf-8")]
    random.Random(3).shuffle(rows)
    items = []
    for r in rows[:n]:
        m = re.search(r"- البحر: ([^،\n]+)", r["messages"][1]["content"])
        key = ar_to_key.get(m.group(1).strip()) if m else None
        if not key:
            continue
        for block in r["messages"][2]["content"].split("\n\n"):
            h = block.split("\n")
            if len(h) == 2:
                items.append((key, h[0], h[1]))
    v = CattPyarudValidator()
    flat = [h for _, s, a in items for h in (s, a)]
    t = time.time()
    dia = []
    for i in range(0, len(flat), 64):
        dia += v.diacritize(flat[i:i + 64])
    dt = time.time() - t
    res = {"catt": [], "overlay": []}
    for j, (key, s, a) in enumerate(items):
        cs, ca = dia[2 * j], dia[2 * j + 1]
        res["catt"].append(exact(v, cs, ca, key))
        res["overlay"].append(exact(v, overlay(s, cs), overlay(a, ca), key))
    print(f"abyat {len(items)}, diacritize {dt:.0f}s ({len(flat) / dt:.1f} hemistichs/s)")
    for name, sc in res.items():
        print(name, "exactly metrical (pyarud is_valid, forced meter):",
              round(sum(sc) / len(sc), 3))
    both = sum(a or b for a, b in zip(res["catt"], res["overlay"], strict=True))
    print("either:", round(both / len(items), 3))


if __name__ == "__main__":
    main()
