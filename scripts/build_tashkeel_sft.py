"""Build a fully diacritized SFT set in which every bayt scans exactly in the poem's meter.

For each example of an existing SFT set: diacritize every hemistich with CATT, keep the source's
own diacritics where it has them (scripts/tashkeel_probe.py: overlay), and keep only the abyat
pyarud accepts as exactly metrical in the labelled meter (is_valid with forced meter). A poem is
kept when at least ``min_abyat`` abyat survive. The request is rewritten for the new count and
asks for full tashkeel; the example bayt in the prompt is shown fully diacritized.

Why: meter is decided by vowels the model does not write. A model that writes them can be
checked without a diacritizer, and constrained at decoding time by an 'arud automaton.

v2 options (the decoder now enforces the meter, so the filter can be looser and poems can stay
whole, which keeps the thread of meaning between abyat):
  threshold    "exact" (v1) or a score such as 0.90: an abyat passes if it is exact or the
               checker scores its diacritized text at least that much in the poem's meter;
  whole_share  if at least this share of a poem's abyat pass, the poem is kept intact,
               failing abyat included (0 disables).
CATT output is cached in artifacts/cache/tashkeel_dia_<split>.json, so a re-filter is quick.

Usage: uv run python scripts/build_tashkeel_sft.py <src_sft_dir> <out_dir> [min_abyat=3]
           [threshold=exact] [whole_share=0]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tashkeel_probe import exact, overlay

from arabic_poet.config import resolve_path
from arabic_poet.generation.prompts import with_tashkeel
from arabic_poet.poetry.meter import CLASSICAL_METERS
from arabic_poet.poetry.prosody import CattPyarudValidator
from arabic_poet.utils.reproducibility import utc_now, write_json


def rewrite_prompt(user: str, old_n: int, new_n: int) -> str:
    user = user.replace(f"- عدد الأبيات: {old_n} بالضبط", f"- عدد الأبيات: {new_n} بالضبط")
    user = user.replace(
        f"- المطلوب {old_n} من الأبيات بالضبط، أي {2 * old_n} من الأسطر (الأشطر) في {old_n} من",
        f"- المطلوب {new_n} من الأبيات بالضبط، أي {2 * new_n} من الأسطر (الأشطر) في {new_n} من",
    )
    return with_tashkeel(user)


_WORKER: CattPyarudValidator | None = None


def _init_worker() -> None:
    global _WORKER
    _WORKER = CattPyarudValidator()


def _dia_batch(batch: list[str]) -> list[str]:
    assert _WORKER is not None
    return _WORKER.diacritize(batch)


def diacritize_all(flat: list[str], label: str, workers: int = 4, size: int = 64) -> list[str]:
    """CATT is far faster on batches of similar length, so sort by length, batch, and fan out."""
    from multiprocessing import get_context

    order = sorted(range(len(flat)), key=lambda i: len(flat[i]))
    batches = [[flat[i] for i in order[k:k + size]] for k in range(0, len(order), size)]
    out: list[str] = [""] * len(flat)
    done = 0
    with get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for k, res in enumerate(pool.imap(_dia_batch, batches)):
            for i, d in zip(order[k * size:(k + 1) * size], res, strict=True):
                out[i] = d
            done += len(res)
            if k % 100 == 0:
                print(label, "diacritized", done, len(flat), flush=True)
    return out


def main() -> None:
    src, out = resolve_path(sys.argv[1]), resolve_path(sys.argv[2])
    min_abyat = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    thr = None if len(sys.argv) <= 4 or sys.argv[4] == "exact" else float(sys.argv[4])
    whole = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    out.mkdir(parents=True, exist_ok=True)
    cdir = resolve_path("artifacts/cache")
    cdir.mkdir(parents=True, exist_ok=True)
    ar_to_key = {v["ar"]: k for k, v in CLASSICAL_METERS.items()}
    v = CattPyarudValidator()
    cache: dict[str, dict | None] = {}
    stats: dict = {"created_at": utc_now(), "source": str(sys.argv[1]), "min_abyat": min_abyat,
                   "threshold": thr or "exact", "whole_share": whole, "poems_whole": 0,
                   "splits": {}, "abyat_in": 0, "abyat_kept": 0, "meters": Counter()}
    for split in ("train", "valid", "test"):
        rows = [json.loads(x) for x in (src / f"{split}.jsonl").open(encoding="utf-8")]
        # Unique poems first (oversampled meters repeat), then one batched diacritization pass.
        todo: dict[str, tuple[str, list[list[str]]]] = {}
        for r in rows:
            user, poem = r["messages"][1]["content"], r["messages"][2]["content"]
            if poem in cache or poem in todo:
                continue
            m = re.search(r"- البحر: ([^،\n]+)", user)
            key = ar_to_key.get(m.group(1).strip()) if m else None
            abyat = [b.split("\n") for b in poem.split("\n\n")]
            abyat = [b for b in abyat if len(b) == 2]
            if key and abyat:
                todo[poem] = (key, abyat)
            else:
                cache[poem] = None
        flat = [h for _, abyat in todo.values() for b in abyat for h in b]
        cfile = cdir / f"tashkeel_dia_{split}.json"
        known = json.loads(cfile.read_text(encoding="utf-8")) if cfile.is_file() else {}
        missing = sorted({h for h in flat if h not in known})
        if missing:
            known.update(zip(missing, diacritize_all(missing, split), strict=True))
            cfile.write_text(json.dumps(known, ensure_ascii=False), encoding="utf-8")
        dia = [known[h] for h in flat]
        pos = 0
        for poem, (key, abyat) in todo.items():
            kept, every = [], []
            for s, a in abyat:
                ds, da = overlay(s, dia[pos]), overlay(a, dia[pos + 1])
                pos += 2
                every.append(f"{ds}\n{da}")
                ok = exact(v, ds, da, key)
                if not ok and thr is not None:
                    ok = (v.scan_detail(ds, da, key).get("score") or 0.0) >= thr
                if ok:
                    kept.append(every[-1])
            if whole and len(kept) >= whole * len(abyat) and len(kept) < len(abyat):
                kept = every
                stats["poems_whole"] += 1
            stats["abyat_in"] += len(abyat)
            stats["abyat_kept"] += len(kept)
            cache[poem] = ({"n_old": len(abyat), "kept": kept, "meter": key}
                           if len(kept) >= min_abyat else None)
        rows_out = []
        for r in rows:
            sys_m, user, poem = (m["content"] for m in r["messages"])
            res = cache[poem]
            if res:
                new_user = rewrite_prompt(user, res["n_old"], len(res["kept"]))
                rows_out.append({"messages": [
                    {"role": "system", "content": sys_m},
                    {"role": "user", "content": new_user},
                    {"role": "assistant", "content": "\n\n".join(res["kept"])}]})
                stats["meters"][res["meter"]] += 1
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats["splits"][split] = {"in": len(rows), "out": len(rows_out)}
    stats["meters"] = dict(stats["meters"].most_common())
    write_json(out / "stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
