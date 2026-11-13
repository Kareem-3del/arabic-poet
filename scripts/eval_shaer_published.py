"""Score the Shaer authors' published generations with this project's strict meter checker.

The Shaer base model (Yehia-7B) is gated, so the model itself cannot be run here. The authors
publish the 3,481 test generations behind the paper's headline numbers; this script scores a
random sample of them with the same measure used for our runs (strip the diacritics,
re-diacritize with CATT, pyarud scan against the requested meter, score >= 0.90 / 0.95).
The same sample's reference poems (real verse) and the untuned Yehia generations are scored
too: the reference row is the checker's ceiling on these prompts.

Per source: abyat_090 / abyat_095 = share of written abyat that pass; req_090 = passes divided
by requested abyat (unwritten abyat count as failures, as in eval_constrained.py);
rhyme_majority = share of abyat ending in the poem's most common final letter (orthographic
heuristic, poems of >= 2 abyat); classifier_hit = the paper's own meter_hit on the same poems.

Needs the tables downloaded under artifacts/cache/shaer_published (see docs).
Usage: uv run --with pyarrow python scripts/eval_shaer_published.py <out.json> [n_poems=500]
"""
from __future__ import annotations

import random
import sys
from collections import Counter

import pyarrow.parquet as pq

from arabic_poet.config import resolve_path
from arabic_poet.poetry.meter import CLASSICAL_METERS
from arabic_poet.poetry.normalization import strip_diacritics
from arabic_poet.poetry.prosody import CattPyarudValidator
from arabic_poet.utils.reproducibility import utc_now, write_json

BASE = "artifacts/cache/shaer_published"
SHAER = f"{BASE}/shaer-sft-test-generations-k5-meter-count-selected-evaluated"
YEHIA = f"{BASE}/shaer-eval-instruction-yehia-base-sft-chat-template-with-hits"
AR_TO_KEY = {v["ar"]: k for k, v in CLASSICAL_METERS.items()}


def abyat_of(text: str) -> list[tuple[str, str]]:
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    return [(lines[i], lines[i + 1]) for i in range(0, len(lines) - 1, 2)]


def score(v: CattPyarudValidator, rows: list[dict], field: str) -> dict:
    poems = [(abyat_of(r[field] or ""), AR_TO_KEY[r["base_meter"]], int(r["requested_bayts"]))
             for r in rows]
    flat = [strip_diacritics(h) for ab, _, _ in poems for b in ab for h in b]
    dia = v.diacritize(flat)
    k = written = requested = p90 = p95 = 0
    rh_num = rh_den = 0
    for ab, meter, req in poems:
        requested += req
        ends = []
        for _ in ab:
            sc = v.scan_detail(dia[k], dia[k + 1], meter).get("score") or 0.0
            w = strip_diacritics(dia[k + 1]).split()
            ends.append((w[-1].rstrip("اىيوه") or w[-1])[-1] if w else "")
            k += 2
            written += 1
            p90 += sc >= 0.90
            p95 += sc >= 0.95
        if len(ab) >= 2:
            rh_num += Counter(ends).most_common(1)[0][1]
            rh_den += len(ab)
    return {"poems": len(rows), "abyat_written": written, "abyat_requested": requested,
            "abyat_090": round(p90 / max(written, 1), 3),
            "abyat_095": round(p95 / max(written, 1), 3),
            "req_090": round(p90 / max(requested, 1), 3),
            "rhyme_majority": round(rh_num / max(rh_den, 1), 3),
            "classifier_hit": (round(sum(int(r["meter_hit"]) for r in rows) / len(rows), 3)
                               if field == "generated_text" else None)}


def main() -> None:
    out = resolve_path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    shaer = pq.read_table(resolve_path(f"{SHAER}/data/test-00000-of-00001.parquet")).to_pylist()
    yehia = pq.read_table(resolve_path(f"{YEHIA}/data/test-00000-of-00001.parquet")).to_pylist()
    shaer = [r for r in shaer if r["form"] == "تام" and r["base_meter"] in AR_TO_KEY]
    random.Random(0).shuffle(shaer)
    shaer = shaer[:n]
    ids = {str(r["source_id"]) for r in shaer}
    yehia = [r for r in yehia if str(r["source_id"]) in ids and r["form"] == "تام"]
    v = CattPyarudValidator()
    res = {"shaer": score(v, shaer, "generated_text"),
           "reference_real_poems": score(v, shaer, "reference_completion"),
           "yehia_untuned": score(v, yehia, "generated_text")}
    for k, r in res.items():
        print(k, r, flush=True)
    write_json(out, {"created_at": utc_now(), "sample": n, "seed": 0, "filter": "form == تام",
                     "measure": "strip diacritics -> CATT -> pyarud forced meter", "results": res})


if __name__ == "__main__":
    main()
