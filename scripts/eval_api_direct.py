"""Ask a hosted chat model for the benchmark poems directly (no draft, no constraint) and save
them in the constrained-result layout (requests run in parallel; a rerun resumes), so
rescore_constrained.py and import_constrained_run.py
apply unchanged. Only the benchmark request text is sent; nothing about the user or project.

The key is read from the macOS Keychain (service ``openai-key``, account ``openai``), never
from a file or the command line.

Usage: uv run python scripts/eval_api_direct.py <model> <out.json> [limit=30] [workers=6]
           [benchmark.jsonl]
       uv run python scripts/eval_api_direct.py --list          # model ids the key can use
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from arabic_poet.config import resolve_path
from arabic_poet.generation.prompts import tashkeel_messages
from arabic_poet.generation.schemas import PoemRequest
from arabic_poet.poetry import arud_fsa as fsa
from arabic_poet.poetry.normalization import strip_diacritics
from arabic_poet.utils.reproducibility import write_json

API = "https://api.openai.com/v1"


def call(path: str, body: dict | None = None) -> dict:
    key = subprocess.run(
        ["security", "find-generic-password", "-a", "openai", "-s", "openai-key", "-w"],
        capture_output=True, text=True, check=True).stdout.strip()
    req = urllib.request.Request(
        f"{API}/{path}", data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=2400) as r:
        return json.loads(r.read())


def clean(text: str) -> str:
    """Keep hemistich lines only: drop punctuation the acceptor does not know."""
    keep = set("ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي" + fsa.MARKS + " \n")
    lines = ["".join(c for c in ln if c in keep).strip() for ln in text.strip().split("\n")]
    return "\n".join(" ".join(ln.split()) for ln in lines)


def main() -> None:
    if sys.argv[1] == "--list":
        print("\n".join(sorted(m["id"] for m in call("models")["data"])))
        return
    model, out = sys.argv[1], resolve_path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    bench_path = sys.argv[5] if len(sys.argv) > 5 else "data/benchmarks/poetry_prompts_v2.jsonl"
    bench = [json.loads(x) for x in resolve_path(bench_path).open(encoding="utf-8")]
    bench = [b for b in bench if b.get("meter") and b.get("topic")][:limit]
    done = {}
    if out.is_file():  # resume: keep what an earlier run already produced
        done = {x["id"]: x for x in json.loads(out.read_text(encoding="utf-8"))["items"]}
    revision = [None]

    def one(it: dict) -> dict:
        req = PoemRequest(**{k: it[k] for k in ("topic", "meter", "rhyme", "line_count", "mood")
                             if it.get(k) is not None})
        t0 = time.time()
        res = call("chat/completions", {"model": model, "messages": tashkeel_messages(req)})
        revision[0] = res.get("model")
        raw = res["choices"][0]["message"]["content"]
        poem = clean(raw)
        abyat = [x.split("\n") for x in poem.split("\n\n")]
        abyat = [x for x in abyat if len(x) == 2]
        n = req.line_count
        exact = sum(fsa.complete(s, req.meter, "sadr") + fsa.complete(a, req.meter, "ajuz")
                    for s, a in abyat) / (2 * n)
        last = [strip_diacritics(a).split()[-1] if a.strip() else "" for _, a in abyat]
        rawi = (sum(1 for w in last if req.rhyme in w[-2:]) / n) if req.rhyme else None
        print(it["id"], round(exact, 3), rawi, flush=True)
        return {"id": it["id"], "request": req.model_dump(exclude_none=True),
                "free": {"poem": poem, "raw": raw, "exact": round(exact, 3), "rawi": rawi,
                         "failed": 0, "retries": 0, "seconds": round(time.time() - t0, 1),
                         "usage": res.get("usage")}}

    todo = [b for b in bench if b["id"] not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(one, b) for b in todo]):
            try:  # one slow or failed request must not lose the others
                row = fut.result()
            except Exception as exc:
                print("failed:", type(exc).__name__, exc, flush=True)
                continue
            done[row["id"]] = row
            write_json(out, {"model": model, "revision": revision[0], "modes": ["free"],
                             "items": [done[b["id"]] for b in bench if b["id"] in done]})

if __name__ == "__main__":
    main()
