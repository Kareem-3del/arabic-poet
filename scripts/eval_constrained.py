"""Benchmark the tashkeel adapter with and without the 'arud constraint.

For every benchmark prompt that names a meter: one poem unconstrained and one constrained
(same seed, tashkeel prompt). Reported per mode:
  exact       share of requested hemistichs that are whole legal hemistichs of the meter, read
              from the model's own diacritics (poetry/arud_fsa.py; no diacritizer involved);
  catt_090 / catt_095   share of requested abyat the usual checker accepts in the requested
              meter after stripping the diacritics and re-diacritizing with CATT (comparable
              with earlier runs);
  rawi        share of abyat ending in the requested rawi (orthographic);
  failed      hemistichs the constrained decoder could not complete; dup: repeated abyat.

Usage: uv run python scripts/eval_constrained.py <adapter_dir> <out.json> [limit] [modes]
           [model_id] [candidates=1] [select=mass|lm] [seed=0] [benchmark.jsonl]
Pass "none" as adapter_dir to run a base model.
modes: comma list of free, form (well-formed diacritics only), rhyme, meter, vocab,
       constrained (meter + rhyme), lexicon (meter + rhyme + corpus vocabulary).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from arabic_poet.config import resolve_path
from arabic_poet.generation import constrained
from arabic_poet.generation.prompts import tashkeel_messages
from arabic_poet.generation.schemas import PoemRequest
from arabic_poet.models.mlx_backend import MLXBackend
from arabic_poet.models.registry import get_spec
from arabic_poet.poetry import arud_fsa as fsa
from arabic_poet.poetry.lexicon import corpus_lexicon
from arabic_poet.poetry.normalization import strip_diacritics
from arabic_poet.poetry.prosody import CattPyarudValidator
from arabic_poet.utils.reproducibility import utc_now, write_json


def main() -> None:
    adapter, out = Path(sys.argv[1]), resolve_path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 999
    bench_path = sys.argv[9] if len(sys.argv) > 9 else "data/benchmarks/poetry_prompts_v2.jsonl"
    bench = [json.loads(x) for x in resolve_path(bench_path).open(encoding="utf-8")]
    bench = [b for b in bench if b.get("meter") and b.get("topic")][:limit]
    modes = sys.argv[4].split(",") if len(sys.argv) > 4 else ["free", "constrained"]
    lex = corpus_lexicon() if {"lexicon", "vocab"} & set(modes) else None
    model_id = sys.argv[5] if len(sys.argv) > 5 else "allam-7b-instruct-4bit"
    seed = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    cands = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    select = sys.argv[7] if len(sys.argv) > 7 else "mass"
    b = MLXBackend(get_spec(model_id), adapter_path=None if adapter.name == "none" else adapter)
    b.load()
    v = CattPyarudValidator()
    items = []
    for it in bench:
        req = PoemRequest(**{k: it[k] for k in ("topic", "meter", "rhyme", "line_count", "mood")
                             if it.get(k) is not None})
        prompt = b.render_prompt(tashkeel_messages(req))
        row = {"id": it["id"], "request": req.model_dump(exclude_none=True)}
        for mode in modes:
            r = constrained.generate(b._model, b._tokenizer, prompt, req.meter, req.line_count,
                                     req.rhyme, constrain=mode != "free", seed=seed,
                                     use_meter=mode not in ("form", "rhyme", "vocab"),
                                     use_rhyme=mode not in ("form", "meter", "vocab"),
                                     lexicon=lex if mode in ("lexicon", "vocab") else None,
                                     candidates=cands, select=select,
                                     max_retries=max(6, 2 * cands))
            abyat = [x.split("\n") for x in r.poem.split("\n\n")]
            abyat = [x for x in abyat if len(x) == 2]
            n = req.line_count
            exact = sum(fsa.complete(s, req.meter, "sadr") + fsa.complete(a, req.meter, "ajuz")
                        for s, a in abyat) / (2 * n)
            flat = [strip_diacritics(h) for x in abyat for h in x if h != "…"]
            dia = v.diacritize(flat) if flat else []
            sc, k = [], 0
            for s, a in abyat:
                if "…" in (s, a):
                    sc.append(0.0)
                    k += (s != "…") + (a != "…")
                    continue
                sc.append(v.scan_detail(dia[k], dia[k + 1], req.meter).get("score") or 0.0)
                k += 2
            last = [strip_diacritics(a).split()[-1] if a.strip() else "" for _, a in abyat]
            rawi = (sum(1 for w in last if req.rhyme and req.rhyme in w[-2:]) / n
                    if req.rhyme else None)
            row[mode] = {
                "poem": r.poem, "exact": round(exact, 3),
                "catt_090": round(sum(x >= 0.90 for x in sc) / n, 3),
                "catt_095": round(sum(x >= 0.95 for x in sc) / n, 3),
                "rawi": rawi, "failed": r.failed_hemistichs, "retries": r.retries,
                "dup": len(abyat) - len({tuple(x) for x in abyat}), "seconds": r.seconds,
            }
            print(it["id"], mode, row[mode]["exact"], row[mode]["catt_090"], r.failed_hemistichs,
                  flush=True)
        items.append(row)

    def mean(mode: str, key: str) -> float | None:
        xs = [r[mode][key] for r in items if r[mode][key] is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    summary = {m: {k: mean(m, k) for k in ("exact", "catt_090", "catt_095", "rawi", "failed",
                                           "retries", "dup", "seconds")}
               for m in modes}
    write_json(out, {"created_at": utc_now(), "adapter": str(adapter), "n": len(items),
                     "summary": summary, "items": items})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
