"""Build what the training job needs to benchmark the tuned model under the 'arud constraint.

Writes into cloud/sagemaker/_bundle/ (uploaded by launch.py next to train_sft.py):
  poet_src.tar.gz  the arabic_poet package (decoder, acceptors)
  lexicon.json     corpus vocabulary forms (poetry/lexicon.py, min_count 2)
  bench.json       benchmark prompts that name a meter: id, request, chat messages

Usage: uv run python cloud/sagemaker/build_eval_bundle.py [benchmark.jsonl ...]
"""
from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

from arabic_poet.config import resolve_path
from arabic_poet.generation.prompts import tashkeel_messages
from arabic_poet.generation.schemas import PoemRequest
from arabic_poet.poetry.lexicon import corpus_lexicon

OUT = Path(__file__).parent / "_bundle"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    src = resolve_path("src/arabic_poet")
    with tarfile.open(OUT / "poet_src.tar.gz", "w:gz") as tar:
        tar.add(src, arcname="arabic_poet",
                filter=lambda t: None if "__pycache__" in t.name else t)
    forms = sorted(corpus_lexicon().words)
    (OUT / "lexicon.json").write_text(json.dumps(forms, ensure_ascii=False), encoding="utf-8")
    files = sys.argv[1:] or ["data/benchmarks/poetry_prompts_v2.jsonl"]
    rows = [json.loads(x) for f in files for x in resolve_path(f).open(encoding="utf-8")]
    bench = []
    for it in rows:
        if not (it.get("meter") and it.get("topic")):
            continue
        req = PoemRequest(**{k: it[k] for k in ("topic", "meter", "rhyme", "line_count", "mood")
                             if it.get(k) is not None})
        bench.append({"id": it["id"], "request": req.model_dump(exclude_none=True),
                      "messages": tashkeel_messages(req)})
    (OUT / "bench.json").write_text(json.dumps(bench, ensure_ascii=False), encoding="utf-8")
    print(f"{len(forms)} lexicon forms, {len(bench)} prompts ->", OUT)


if __name__ == "__main__":
    main()
