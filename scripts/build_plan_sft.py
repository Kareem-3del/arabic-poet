"""Build a plan-conditioned SFT set (tashkeel-v3) from a diacritized one (tashkeel-v2).

Each request gains a plan read off the real poem: per bayt, its rhyme word (the last word of
the 'ajuz) and up to ``k`` content words taken from the bayt in order. The model then learns to
write a bayt that uses given words and ends on a rhyme word fixed in advance, which is how the
classical manuals describe composition (Ibn Rashiq: no bayt without knowing its rhyme) and what
the generation literature found to keep rhyme and content together (plan words per line, rhyme
word first). At inference the plan comes from a planner, and the decoder enforces the rhyme word.

Content words: undiacritized, at least 3 letters, not among the ``stop`` most frequent forms of
the training split, not the rhyme word; ``k`` of them picked at random (seeded) so the model does
not rely on any fixed position. Also repairs the doubled tanween (ًاً) left by the diacritizer.

Usage: uv run python scripts/build_plan_sft.py <src_dir> <out_dir> [k=2] [stop=400]
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter

from arabic_poet.config import resolve_path
from arabic_poet.generation.prompts import with_bayt_plan
from arabic_poet.poetry.lexicon import norm
from arabic_poet.poetry.normalization import strip_diacritics
from arabic_poet.utils.reproducibility import utc_now, write_json


def words_of(line: str) -> list[str]:
    return [w for w in strip_diacritics(line).replace("ـ", "").split() if w]


def plan_of(poem: str, stop: frozenset[str], k: int, rng: random.Random):
    plan = []
    for bayt in poem.split("\n\n"):
        halves = bayt.split("\n")
        if len(halves) != 2:
            return None
        ws = words_of(halves[0]) + words_of(halves[1])
        if not ws:
            return None
        rhyme = ws[-1]
        seen, pool = {norm(rhyme)}, []
        for i, w in enumerate(ws[:-1]):
            n = norm(w)
            if len(n) >= 3 and n not in stop and n not in seen:
                seen.add(n)
                pool.append((i, w))
        pick = sorted(rng.sample(pool, min(k, len(pool))))
        plan.append((rhyme, [w for _, w in pick]))
    return plan


def main() -> None:
    src, out = resolve_path(sys.argv[1]), resolve_path(sys.argv[2])
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    n_stop = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    out.mkdir(parents=True, exist_ok=True)
    rows = {s: [json.loads(x) for x in (src / f"{s}.jsonl").open(encoding="utf-8")]
            for s in ("train", "valid", "test")}
    freq: Counter[str] = Counter()
    for r in rows["train"]:
        freq.update(norm(w) for w in words_of(r["messages"][2]["content"]))
    stop = frozenset(w for w, _ in freq.most_common(n_stop))
    rng = random.Random(0)
    stats = {"created_at": utc_now(), "source": sys.argv[1], "k": k, "stop": n_stop, "splits": {}}
    for split, rs in rows.items():
        kept = 0
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rs:
                sys_m, user, poem = (m["content"] for m in r["messages"])
                poem = poem.replace("ًاً", "ًا")
                plan = plan_of(poem, stop, k, rng)
                if plan is None:
                    continue
                kept += 1
                f.write(json.dumps({"messages": [
                    {"role": "system", "content": sys_m},
                    {"role": "user", "content": with_bayt_plan(user, plan)},
                    {"role": "assistant", "content": poem}]}, ensure_ascii=False) + "\n")
        stats["splits"][split] = {"in": len(rs), "out": kept}
    write_json(out / "stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
