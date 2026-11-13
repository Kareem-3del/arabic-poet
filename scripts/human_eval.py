# ruff: noqa: E501, SIM115
"""Summarize the blind human ratings: per-system means for each rater, agreement between the
raters, and paired contrasts with prompt-level bootstrap intervals.

Input: the returned rating sheets (code + seven scores per poem) and the code key. Rater A used
a 0-10 scale and rater B the sheet's 1-5 scale; A is halved for the combined figures, which is
a rescaling and not an exact equivalence, so both raters are also reported separately.

Usage: uv run python scripts/human_eval.py <rater_A.csv> <rater_B.csv>
       -> paper/human_eval.json, paper/data/human_ratings/rater_{A,B}.csv (codes, systems, scores)
"""
from __future__ import annotations

import csv
import json
import random
import statistics
import sys
from collections import defaultdict

from arabic_poet.config import resolve_path
from arabic_poet.utils.reproducibility import utc_now, write_json

COLS = {"الالتزام بالموضوع": "topic", "الترابط بين الأبيات": "coherence", "سلامة اللغة": "language",
        "الجودة الشعرية": "quality", "الوزن": "meter", "القافية": "rhyme", "التقدير العام": "overall"}
SYSTEMS = ["allam_free", "allam_constrained", "jais_constrained", "gpt61sol", "claudefable52"]


def rank(x: list[float]) -> list[float]:
    o = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float | None:
    ra, rb = rank(a), rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return round(sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True)) / den, 3) if den else None


def boot(d: list[float], n: int = 10_000) -> list[float]:
    rng = random.Random(0)
    k = len(d)
    m = sorted(sum(d[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [round(m[int(0.025 * n)], 2), round(m[int(0.975 * n)], 2)]


def main() -> None:
    key = json.loads(resolve_path("artifacts/blind_eval/KEY_do_not_share.json").read_text(
        encoding="utf-8"))
    raters = {}
    for name, path, scale in (("A", sys.argv[1], 0.5), ("B", sys.argv[2], 1.0)):
        rows = {}
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            rows[r["الكود"]] = {v: float(r[k]) * scale for k, v in COLS.items()}
        raters[name] = rows
        with (resolve_path("paper/data/human_ratings") / f"rater_{name}.csv").open(
                "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["code", "prompt", "system", "scale", *COLS.values()])
            for code, sc in sorted(rows.items()):
                w.writerow([code, key[code]["prompt"], key[code]["system"],
                            "0-10" if scale != 1 else "1-5", *[sc[c] / scale for c in COLS.values()]])
    codes = sorted(set(raters["A"]) & set(raters["B"]))
    out: dict = {"created_at": utc_now(), "poems": len(codes), "systems": SYSTEMS,
                 "note": "rater A used 0-10; values shown on a 0-5 scale (halved)", "means": {},
                 "agreement": {}, "contrasts": {}}
    for name, rows in raters.items():
        by = defaultdict(lambda: defaultdict(list))
        for c in codes:
            for d, v in rows[c].items():
                by[key[c]["system"]][d].append(v)
        out["means"][name] = {s: {d: round(statistics.mean(v), 2) for d, v in by[s].items()}
                              for s in SYSTEMS}
    out["means"]["both"] = {s: {d: round((out["means"]["A"][s][d] + out["means"]["B"][s][d]) / 2, 2)
                                for d in COLS.values()} for s in SYSTEMS}
    groups = {"all": SYSTEMS, "fine_tuned": SYSTEMS[:3], "hosted": SYSTEMS[3:]}
    for g, members in groups.items():
        ks = [c for c in codes if key[c]["system"] in members]
        out["agreement"][g] = {"n": len(ks), **{
            d: spearman([raters["A"][c][d] for c in ks], [raters["B"][c][d] for c in ks])
            for d in COLS.values()}}
    idx = {(key[c]["prompt"], key[c]["system"]): c for c in codes}
    prompts = sorted({key[c]["prompt"] for c in codes})

    def contrast(a: str, b: str) -> dict:
        res = {}
        for name, rows in raters.items():
            res[name] = {}
            for d in ("topic", "coherence", "language", "quality", "meter", "overall"):
                diff = [rows[idx[p, a]][d] - rows[idx[p, b]][d] for p in prompts]
                res[name][d] = {"diff": round(statistics.mean(diff), 2), "ci95": boot(diff)}
        return res

    out["contrasts"] = {
        "allam_constrained minus allam_free": contrast("allam_constrained", "allam_free"),
        "gpt61sol minus allam_constrained": contrast("gpt61sol", "allam_constrained"),
        "claudefable52 minus gpt61sol": contrast("claudefable52", "gpt61sol"),
    }
    write_json(resolve_path("paper/human_eval.json"), out)
    print(json.dumps({k: out[k] for k in ("means", "agreement")}, ensure_ascii=False, indent=1))
    for k, v in out["contrasts"].items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
