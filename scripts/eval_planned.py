"""Does the model follow a per-bayt plan under the constraint? (oracle plans)

Takes requests from the plan-conditioned test split (plans read off real poems, see
build_plan_sft.py) and generates each twice with the meter + vocabulary constraint:
  plain    the request without the plan (tashkeel-v2 prompt);
  planned  the request with the plan, and the decoder holding each 'ajuz to its planned rhyme word.
Reports exact meter, unfinished hemistichs, share of abyat ending on the planned rhyme word and
share of planned content words that appear in their bayt. The plans come from real poems, so this
measures plan-following only; a planner is a separate step.

Usage: uv run python scripts/eval_planned.py <adapter_dir|none> <out.json> [n=10] [max_abyat=8]
           [model_id]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from arabic_poet.config import resolve_path
from arabic_poet.generation import constrained
from arabic_poet.generation.prompts import PLAN_HEAD
from arabic_poet.models.mlx_backend import MLXBackend
from arabic_poet.models.registry import get_spec
from arabic_poet.poetry import arud_fsa as fsa
from arabic_poet.poetry.lexicon import corpus_lexicon, norm
from arabic_poet.poetry.meter import CLASSICAL_METERS
from arabic_poet.poetry.normalization import strip_diacritics
from arabic_poet.utils.reproducibility import utc_now, write_json

_ROW = re.compile(r"^\d+\. القافية: (\S+)(?: \| الكلمات: (.*))?$", re.M)


def main() -> None:
    adapter, out = Path(sys.argv[1]), resolve_path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    max_abyat = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    model_id = sys.argv[5] if len(sys.argv) > 5 else "allam-7b-instruct-4bit"
    ar_to_key = {v["ar"]: k for k, v in CLASSICAL_METERS.items()}
    rows = []
    for line in resolve_path("data/processed/sft-allam-tashkeel-v3/test.jsonl").open(
            encoding="utf-8"):
        sys_m, user, _ = (m["content"] for m in json.loads(line)["messages"])
        plan = [(m.group(1), (m.group(2) or "").split("، ") if m.group(2) else [])
                for m in _ROW.finditer(user)]
        meter = ar_to_key.get(re.search(r"- البحر: ([^،\n]+)", user).group(1).strip())
        rawi = re.search(r"رويّها «(.)»", user)
        if meter and 3 <= len(plan) <= max_abyat and len({norm(r) for r, _ in plan}) == len(plan):
            rows.append((sys_m, user, plan, meter, rawi.group(1) if rawi else None))
        if len(rows) == n:
            break
    b = MLXBackend(get_spec(model_id), adapter_path=None if adapter.name == "none" else adapter)
    b.load()
    lex = corpus_lexicon()
    items = []
    for i, (sys_m, user, plan, meter, rawi) in enumerate(rows):
        head, _, tail = user.partition("\n\n" + PLAN_HEAD)
        plain_user = head + "\n\nطريقة الكتابة:" + tail.partition("\n\nطريقة الكتابة:")[2]
        row = {"id": i, "meter": meter, "rawi": rawi, "plan": plan}
        for mode, u in (("plain", plain_user), ("planned", user)):
            prompt = b.render_prompt([{"role": "system", "content": sys_m},
                                      {"role": "user", "content": u}])
            r = constrained.generate(
                b._model, b._tokenizer, prompt, meter, len(plan), rawi, seed=0, lexicon=lex,
                rhyme_words=[w for w, _ in plan] if mode == "planned" else None)
            abyat = [x.split("\n") for x in r.poem.split("\n\n")]
            exact = sum(fsa.complete(s, meter, "sadr") + fsa.complete(a, meter, "ajuz")
                        for s, a in abyat) / (2 * len(plan))
            hit = used = total = 0
            for (s, a), (rw, kws) in zip(abyat, plan, strict=True):
                ws = [norm(w) for w in strip_diacritics(f"{s} {a}").split()]
                hit += bool(ws) and ws[-1] == norm(rw)
                total += len(kws)
                used += sum(norm(k) in ws for k in kws)
            row[mode] = {"poem": r.poem, "exact": round(exact, 3), "failed": r.failed_hemistichs,
                         "retries": r.retries, "rhyme_hit": round(hit / len(plan), 3),
                         "words_used": round(used / total, 3) if total else None,
                         "seconds": r.seconds}
            print(i, mode, row[mode]["exact"], row[mode]["failed"], row[mode]["rhyme_hit"],
                  row[mode]["words_used"], flush=True)
        items.append(row)

    def mean(mode: str, key: str) -> float | None:
        xs = [r[mode][key] for r in items if r[mode][key] is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    summary = {m: {k: mean(m, k) for k in ("exact", "failed", "retries", "rhyme_hit",
                                           "words_used", "seconds")}
               for m in ("plain", "planned")}
    write_json(out, {"created_at": utc_now(), "adapter": str(adapter), "n": len(items),
                     "summary": summary, "items": items})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
