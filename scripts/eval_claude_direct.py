"""Ask a Claude model for the benchmark poems directly (prompt only, no constraint, one sample
each) and save them in the constrained-result layout, like eval_api_direct.py does for the
other hosted model. Only the benchmark request text is sent.

Credentials: the Anthropic SDK's own resolution (ANTHROPIC_API_KEY or an `ant auth login`
profile); if neither is present, the key is read from the macOS Keychain (service
``anthropic-key``, account ``anthropic``). No fallback model is configured: a refusal is
recorded as a refusal, so every poem in the file was written by the named model.

Usage: uv run --with anthropic python scripts/eval_claude_direct.py <model> <out.json>
           [limit=30] [workers=6] [benchmark.jsonl]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic

from arabic_poet.config import resolve_path
from arabic_poet.generation.prompts import tashkeel_messages
from arabic_poet.generation.schemas import PoemRequest
from arabic_poet.poetry import arud_fsa as fsa
from arabic_poet.poetry.normalization import strip_diacritics
from arabic_poet.utils.reproducibility import write_json

KEEP = set("ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي" + fsa.MARKS + " \n")


def client() -> anthropic.Anthropic:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return anthropic.Anthropic()
    key = subprocess.run(
        ["security", "find-generic-password", "-a", "anthropic", "-s", "anthropic-key", "-w"],
        capture_output=True, text=True).stdout.strip()
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


def clean(text: str) -> str:
    lines = ["".join(c for c in ln if c in KEEP).strip() for ln in text.strip().split("\n")]
    return "\n".join(" ".join(ln.split()) for ln in lines)


def main() -> None:
    model, out = sys.argv[1], resolve_path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    bench_path = sys.argv[5] if len(sys.argv) > 5 else "data/benchmarks/poetry_prompts_v2.jsonl"
    bench = [json.loads(x) for x in resolve_path(bench_path).open(encoding="utf-8")]
    bench = [b for b in bench if b.get("meter") and b.get("topic")][:limit]
    api = client()
    done = {}
    if out.is_file():
        done = {x["id"]: x for x in json.loads(out.read_text(encoding="utf-8"))["items"]}

    def one(it: dict) -> dict:
        req = PoemRequest(**{k: it[k] for k in ("topic", "meter", "rhyme", "line_count", "mood")
                             if it.get(k) is not None})
        system, user = (m["content"] for m in tashkeel_messages(req))
        t0 = time.time()
        try:
            res = api.messages.create(model=model, max_tokens=16000, system=system,
                                      messages=[{"role": "user", "content": user}])
        except anthropic.RateLimitError:
            time.sleep(30)
            res = api.messages.create(model=model, max_tokens=16000, system=system,
                                      messages=[{"role": "user", "content": user}])
        raw = "".join(b.text for b in res.content if b.type == "text")
        poem = clean(raw) if res.stop_reason != "refusal" else ""
        abyat = [x.split("\n") for x in poem.split("\n\n")]
        abyat = [x for x in abyat if len(x) == 2]
        n = req.line_count
        exact = sum(fsa.complete(s, req.meter, "sadr") + fsa.complete(a, req.meter, "ajuz")
                    for s, a in abyat) / (2 * n)
        last = [strip_diacritics(a).split()[-1] if a.strip() else "" for _, a in abyat]
        rawi = (sum(1 for w in last if req.rhyme in w[-2:]) / n) if req.rhyme else None
        print(it["id"], round(exact, 3), rawi, res.stop_reason, flush=True)
        return {"id": it["id"], "request": req.model_dump(exclude_none=True),
                "free": {"poem": poem, "raw": raw, "exact": round(exact, 3), "rawi": rawi,
                         "failed": 0, "retries": 0, "seconds": round(time.time() - t0, 1),
                         "stop_reason": res.stop_reason, "model": res.model,
                         "usage": res.usage.to_dict()}}

    todo = [b for b in bench if b["id"] not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(one, todo):
            done[row["id"]] = row
            write_json(out, {"model": model, "revision": row["free"]["model"], "modes": ["free"],
                             "items": [done[b["id"]] for b in bench if b["id"] in done]})


if __name__ == "__main__":
    main()
