"""Prepare a sheet for a prosodist: abyat to judge by hand, so the checkers can be measured
against expert labels instead of against their own pass rates.

200 abyat from the diacritized test split, stratified by meter, 100 copies of other test
abyat with one word removed or swapped, and 24 abyat from each system's output (so the expert's
verdict can be compared with what each checker said about machine-written verse). Shuffled.
The expert fills: does it scan in the stated meter (yes / no / yes with a licence), and a note.
The key (which rows were damaged, and how)
is written separately. Seeded, so the sheet is the same on every run.

Usage: uv run python scripts/build_prosody_gold_sheet.py   -> artifacts/prosody_gold/
"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict

from arabic_poet.config import resolve_path
from arabic_poet.poetry import arud_fsa as fsa
from arabic_poet.poetry.meter import CLASSICAL_METERS
from arabic_poet.utils.reproducibility import write_json

E = "paper/data/"
SYSTEM_FILES = [
    ("allam_constrained", E + "constrained_tashkeel_v2_marks.json", "lexicon"),
    ("jais_constrained", E + "constrained_jais3_32b_tashkeel_v2.json", "lexicon"),
    ("gpt61sol", E + "direct_gpt61sol_scored.json", "free"),
    ("gpt6astra", E + "direct_gpt6astra_scored.json", "free"),
    ("claudefable52", E + "direct_claudefable52_scored.json", "free"),
]
PER_SYSTEM = 24


def main() -> None:
    rng = random.Random(0)
    ar = {v["ar"]: k for k, v in CLASSICAL_METERS.items()}
    by_meter: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen = set()
    for line in resolve_path("data/processed/sft-allam-tashkeel-v2/test.jsonl").open(
            encoding="utf-8"):
        m = json.loads(line)["messages"]
        mm = re.search(r"- البحر: ([^،\n]+)", m[1]["content"])
        if not mm or mm.group(1).strip() not in ar or m[2]["content"] in seen:
            continue
        seen.add(m[2]["content"])
        for b in m[2]["content"].split("\n\n"):
            h = b.split("\n")
            if len(h) == 2:
                by_meter[mm.group(1).strip()].append((h[0], h[1]))
    meters = sorted(by_meter, key=lambda k: -len(by_meter[k]))
    rows, key = [], {}

    def take(n_total: int, damaged: bool) -> None:
        per = max(n_total // len(meters), 1)
        for meter in meters:
            pool = by_meter[meter]
            for _ in range(min(per, len(pool))):
                s, a = pool.pop(rng.randrange(len(pool)))
                how = ""
                if damaged:
                    words = a.split()
                    if len(words) < 3:
                        continue
                    i = rng.randrange(len(words) - 1)
                    if rng.random() < 0.5:
                        how = f"removed word {i + 1} of the 'ajuz"
                        del words[i]
                    else:
                        j = rng.randrange(len(s.split()))
                        how = f"word {i + 1} of the 'ajuz replaced by word {j + 1} of the sadr"
                        words[i] = s.split()[j]
                    a = " ".join(words)
                rid = f"g{len(rows) + 1:03d}"
                rows.append({"id": rid, "meter": meter, "sadr": s, "ajuz": a,
                             "scans (yes / no / licence)": "", "note": ""})
                key[rid] = {"damaged": damaged, "how": how}

    take(200, False)
    take(100, True)
    # abyat written by the systems, so the expert's verdict can be set against each checker
    key_ar = {k: v["ar"] for k, v in CLASSICAL_METERS.items()}
    for name, path, mode in SYSTEM_FILES:
        fp = resolve_path(path)
        if not fp.is_file():
            continue
        pool = []
        for it in json.loads(fp.read_text(encoding="utf-8"))["items"]:
            m = it["request"]["meter"]
            for b in it[mode]["poem"].split("\n\n"):
                h = b.split("\n")
                if len(h) == 2 and "…" not in h:
                    pool.append((m, h[0], h[1]))
        for m, sadr, ajuz in rng.sample(pool, min(PER_SYSTEM, len(pool))):
            rid = f"g{len(rows) + 1:03d}"
            rows.append({"id": rid, "meter": key_ar.get(m, m), "sadr": sadr, "ajuz": ajuz,
                         "scans (yes / no / licence)": "", "note": ""})
            key[rid] = {"damaged": False, "how": "", "system": name,
                        "acceptor": bool(fsa.complete(sadr, m, "sadr")
                                         and fsa.complete(ajuz, m, "ajuz"))}
    rng.shuffle(rows)
    out = resolve_path("artifacts/prosody_gold")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sheet_for_prosodist.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_json(out / "KEY_do_not_share.json", key)
    print(len(rows), "rows;", sum(v["damaged"] for v in key.values()), "damaged;",
          sum("system" in v for v in key.values()), "from systems;", len(meters), "meters")


if __name__ == "__main__":
    main()
