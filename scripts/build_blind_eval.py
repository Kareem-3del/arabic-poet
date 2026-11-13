"""Assemble a blind human evaluation from the saved poems of every system.

One poem per system per prompt goes into a single shuffled set. Two outputs:
  * a combined result file, imported as one review run (the review app's blind mode hides the
    system; the shuffled order hides it too);
  * a rating sheet for raters who do not use the app (CSV: code, request, poem, empty score
    columns from the rubric) and a separate key (code -> system) that raters must not see.

Usage: uv run python scripts/build_blind_eval.py   -> artifacts/blind_eval/
"""
from __future__ import annotations

import csv
import hashlib
import json
import random

from arabic_poet.config import resolve_path
from arabic_poet.utils.reproducibility import write_json

E = "paper/data/"
SYSTEMS = [
    ("allam_free", E + "constrained_tashkeel_v2.json", "free"),
    ("allam_constrained", E + "constrained_tashkeel_v2_marks.json", "lexicon"),
    ("jais_constrained", E + "constrained_jais3_32b_tashkeel_v2.json", "lexicon"),
    ("gpt61sol", E + "direct_gpt61sol_scored.json", "free"),
    ("gpt6astra", E + "direct_gpt6astra_scored.json", "free"),
    ("claudefable52", E + "direct_claudefable52_scored.json", "free"),
]
SCORES = ("meaning_coherence", "language", "imagery", "meter", "rhyme", "overall")


def main() -> None:
    out = resolve_path("artifacts/blind_eval")
    out.mkdir(parents=True, exist_ok=True)
    items: dict[str, dict] = {}
    modes = []
    for name, path, mode in SYSTEMS:
        if not resolve_path(path).is_file():
            continue
        modes.append(name)
        for it in json.loads(resolve_path(path).read_text(encoding="utf-8"))["items"]:
            row = items.setdefault(it["id"], {"id": it["id"], "request": it["request"]})
            row[name] = {k: it[mode].get(k) for k in ("poem", "exact", "failed", "retries",
                                                      "seconds")}
    write_json(out / "combined.json", {"model": "blind-mix", "revision": None, "modes": modes,
                                       "items": list(items.values())})
    sheet, key = [], {}
    for it in items.values():
        r = it["request"]
        for name in modes:
            code = hashlib.sha256(f"{it['id']}/{name}/blind-v1".encode()).hexdigest()[:8]
            key[code] = {"prompt": it["id"], "system": name}
            sheet.append({"code": code, "topic": r["topic"], "meter": r["meter"],
                          "rhyme_letter": r.get("rhyme") or "", "abyat": r["line_count"],
                          "poem": it[name]["poem"], **dict.fromkeys(SCORES, ""), "notes": ""})
    random.Random(0).shuffle(sheet)
    with (out / "rating_sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    write_json(out / "KEY_do_not_share.json", key)
    print(len(sheet), "poems from", modes)


if __name__ == "__main__":
    main()
