"""Rhyme words attested in the training poems, per (meter, rawi), with counts.

Read from the plans of the plan-conditioned training split (build_plan_sft.py). A planner picks
each bayt's rhyme word from this list, so every candidate has ended a real bayt in that meter
and rawi at least once.

Usage: uv run python scripts/build_rhyme_inventory.py [min_count=1]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict

from arabic_poet.config import resolve_path
from arabic_poet.poetry.meter import CLASSICAL_METERS
from arabic_poet.utils.reproducibility import utc_now, write_json

_ROW = re.compile(r"^\d+\. القافية: (\S+)", re.M)


def main() -> None:
    min_count = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ar_to_key = {v["ar"]: k for k, v in CLASSICAL_METERS.items()}
    inv: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[str] = set()
    src = "data/processed/sft-allam-tashkeel-v3/train.jsonl"
    for line in resolve_path(src).open(encoding="utf-8"):
        msgs = json.loads(line)["messages"]
        if msgs[2]["content"] in seen:  # oversampled meters repeat poems
            continue
        seen.add(msgs[2]["content"])
        user = msgs[1]["content"]
        m = re.search(r"- البحر: ([^،\n]+)", user)
        r = re.search(r"رويّها «(.)»", user)
        key = ar_to_key.get(m.group(1).strip()) if m else None
        if key and r:
            inv[f"{key}|{r.group(1)}"].update(_ROW.findall(user))
    out = {k: [[w, c] for w, c in v.most_common() if c >= min_count]
           for k, v in sorted(inv.items())}
    write_json(resolve_path("data/processed/rhyme_words_v1.json"),
               {"created_at": utc_now(), "source": src,
                "poems": len(seen), "keys": len(out), "words": out})
    sizes = sorted(((len(v), k) for k, v in out.items()), reverse=True)
    print(len(seen), "poems,", len(out), "meter|rawi keys; largest:", sizes[:5], "smallest:",
          sizes[-3:])


if __name__ == "__main__":
    main()
