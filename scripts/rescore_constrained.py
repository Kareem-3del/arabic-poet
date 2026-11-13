"""Rescore poems produced on a remote machine with the same measures as eval_constrained.py.

The SageMaker job writes poems plus the measures that need no CATT. This adds catt_090 /
catt_095 / dup per mode and a summary, so the file is comparable with the local runs.

Usage: uv run python scripts/rescore_constrained.py <in.json> <out.json>
"""

from __future__ import annotations

import json
import sys

from arabic_poet.config import resolve_path
from arabic_poet.poetry.normalization import strip_diacritics
from arabic_poet.poetry.prosody import CattPyarudValidator
from arabic_poet.utils.reproducibility import utc_now, write_json

KEYS = ("exact", "catt_090", "catt_095", "rawi", "failed", "retries", "dup", "seconds")


def main() -> None:
    src = json.loads(resolve_path(sys.argv[1]).read_text(encoding="utf-8"))
    v = CattPyarudValidator()
    modes = src["modes"]
    for row in src["items"]:
        req = row["request"]
        n = req["line_count"]
        for mode in modes:
            r = row[mode]
            abyat = [x.split("\n") for x in r["poem"].split("\n\n")]
            abyat = [x for x in abyat if len(x) == 2]
            flat = [strip_diacritics(h) for x in abyat for h in x if h != "…"]
            dia = v.diacritize(flat) if flat else []
            sc, k = [], 0
            for s, a in abyat:
                if "…" in (s, a):
                    sc.append(0.0)
                    k += (s != "…") + (a != "…")
                    continue
                sc.append(v.scan_detail(dia[k], dia[k + 1], req["meter"]).get("score") or 0.0)
                k += 2
            r["catt_090"] = round(sum(x >= 0.90 for x in sc) / n, 3)
            r["catt_095"] = round(sum(x >= 0.95 for x in sc) / n, 3)
            r["dup"] = len(abyat) - len({tuple(x) for x in abyat})
            r["abyat"] = len(abyat)
            print(row["id"], mode, r["exact"], r["catt_090"], r["failed"], flush=True)

    def mean(mode: str, key: str) -> float | None:
        xs = [r[mode][key] for r in src["items"] if r[mode].get(key) is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    summary = {m: {k: mean(m, k) for k in KEYS} for m in modes}
    write_json(resolve_path(sys.argv[2]), {
        "created_at": utc_now(), "model": src["model"], "revision": src["revision"],
        "n": len(src["items"]), "summary": summary, "items": src["items"]})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
