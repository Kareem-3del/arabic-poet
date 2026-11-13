"""Recompute every figure of the paper's main table from the saved poems, with one code path.

For each system: hemistichs exactly metrical by the system's own diacritics (current acceptor),
the usual checker columns saved with the poems (strip, CATT, pyarud), requested rawi under the
strict rule (rawi, then at most one wasl letter or ha' + khuruj), unfinished hemistichs, and
hemistichs found verbatim (undiacritized, normalized) in the training corpus.

Usage: uv run python scripts/paper_tables.py   -> paper/results.json and a table on stdout
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict

from arabic_poet.config import resolve_path
from arabic_poet.poetry import arud_fsa as fsa
from arabic_poet.poetry.lexicon import _WORD, norm
from arabic_poet.poetry.normalization import strip_diacritics
from arabic_poet.utils.reproducibility import utc_now, write_json

E = "paper/data/"
SYSTEMS = [
    ("ALLaM-7B FT, free", E + "constrained_tashkeel_v2.json", "free"),
    ("ALLaM-7B FT, constrained", E + "constrained_tashkeel_v2.json", "lexicon"),
    ("ALLaM-7B FT, constrained, final acceptor", E + "constrained_tashkeel_v2_marks.json",
     "lexicon"),
    ("Jais-3-32B FT, free", E + "constrained_jais3_32b_tashkeel_v2.json", "free"),
    ("Jais-3-32B FT, constrained", E + "constrained_jais3_32b_tashkeel_v2.json", "lexicon"),
    ("Jais-3-32B FT, free, rerun", E + "constrained_jais3_32b_final.json", "free"),
    ("Jais-3-32B FT, constrained, final acceptor", E + "constrained_jais3_32b_final.json",
     "lexicon"),
    ("gpt-6.1-sol, prompt only", E + "direct_gpt61sol_scored.json", "free"),
    ("gpt-6-astra, prompt only", E + "direct_gpt6astra_scored.json", "free"),
    ("claude-fable-5-2, prompt only", E + "direct_claudefable52_scored.json", "free"),
    # ablation, ALLaM, seed 0: one constraint at a time
    ("ablation: diacritics only", E + "ablation_allam_s0.json", "form"),
    ("ablation: rhyme only", E + "ablation_allam_s0.json", "rhyme"),
    ("ablation: vocabulary only", E + "ablation_allam_s0.json", "vocab"),
    ("ablation: meter only", E + "ablation_allam_s0.json", "meter"),
    ("ablation: meter + rhyme", E + "ablation_allam_s0.json", "constrained"),
    # before fine-tuning: the base model, same prompts
    ("ALLaM-7B base, free", E + "base_allam_v2prompts.json", "free"),
    # held-out prompts, written after the acceptor was frozen
    ("held-out: ALLaM-7B FT, free", E + "heldout_allam.json", "free"),
    ("held-out: ALLaM-7B FT, constrained", E + "heldout_allam.json", "lexicon"),
    ("held-out: ALLaM-7B base, free", E + "heldout_base_allam.json", "free"),
    ("held-out: Jais-3-32B FT, free", E + "heldout_jais3_32b.json", "free"),
    ("held-out: Jais-3-32B FT, constrained", E + "heldout_jais3_32b.json", "lexicon"),
    ("held-out: gpt-6.1-sol, prompt only", E + "heldout_gpt61sol_scored.json", "free"),
    ("held-out: gpt-6-astra, prompt only", E + "heldout_gpt6astra_scored.json", "free"),
    ("held-out: claude-fable-5-2, prompt only", E + "heldout_claudefable52_scored.json", "free"),
    # two more seeds, ALLaM
    ("seed 1: free", E + "seeds_allam_s1.json", "free"),
    ("seed 1: constrained", E + "seeds_allam_s1.json", "lexicon"),
    ("seed 2: free", E + "seeds_allam_s2.json", "free"),
    ("seed 2: constrained", E + "seeds_allam_s2.json", "lexicon"),
]
_PRE = ("وال", "بال", "فال", "كال", "لل", "ال", "و", "ف", "ب", "ل", "ك")


def stem(w: str) -> str:
    w = norm(w)
    for p in _PRE:
        if w.startswith(p) and len(w) - len(p) >= 3:
            return w[len(p):]
    return w


def topic_words_used(topic: str, poem: str) -> float:
    """Share of the topic's content words whose stem occurs inside some word of the poem.
    A lexical proxy for staying on topic: it sees shared words, not meaning."""
    stems = [t for t in (stem(w) for w in _WORD.findall(topic)) if len(t) >= 3]
    words = [norm(w) for w in _WORD.findall(strip_diacritics(poem))]
    if not stems:
        return 0.0
    return sum(any(t[:4] in w for w in words) for t in stems) / len(stems)


def key(line: str) -> str:
    return " ".join(norm(w) for w in _WORD.findall(strip_diacritics(line)))


def corpus_hemistichs() -> frozenset[str] | None:
    """Hemistichs of the training corpus, or None when the corpus is not on disk (it is not
    distributed with the code); the verbatim column is then left out."""
    if not resolve_path("data/processed/wikisource-pd-v1/records").is_dir():
        return None
    out = set()
    for f in resolve_path("data/processed/wikisource-pd-v1/records").glob("*.jsonl"):
        for line in f.open(encoding="utf-8"):
            for bayt in json.loads(line)["text"].split("\n"):
                out.update(k for h in bayt.split(" ... ") if len(k := key(h)) >= 10)
    return frozenset(out)


def boot(xs: list[float], ys: list[float] | None = None, n: int = 10_000) -> list[float]:
    """95% percentile bootstrap over prompts for a mean, or for a paired difference of means.
    It describes the spread across these prompts only: not seed variance, not other prompts."""
    rng = random.Random(0)
    d = [x - y for x, y in zip(xs, ys, strict=True)] if ys is not None else xs
    k = len(d)
    means = sorted(sum(d[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [round(means[int(0.025 * n)], 3), round(means[int(0.975 * n)], 3)]


PAIRS = [
    ("ALLaM-7B FT, constrained, final acceptor", "ALLaM-7B FT, free"),
    ("gpt-6.1-sol, prompt only", "ALLaM-7B FT, constrained, final acceptor"),
    ("claude-fable-5-2, prompt only", "ALLaM-7B FT, constrained, final acceptor"),
    ("claude-fable-5-2, prompt only", "gpt-6.1-sol, prompt only"),
    ("gpt-6-astra, prompt only", "gpt-6.1-sol, prompt only"),
    ("ALLaM-7B FT, free", "ALLaM-7B base, free"),
    ("seed 1: free", "ALLaM-7B base, free"),
    ("seed 2: free", "ALLaM-7B base, free"),
    ("held-out: ALLaM-7B FT, constrained", "held-out: ALLaM-7B FT, free"),
    ("held-out: ALLaM-7B FT, free", "held-out: ALLaM-7B base, free"),
    ("held-out: claude-fable-5-2, prompt only", "held-out: ALLaM-7B FT, constrained"),
    ("held-out: gpt-6.1-sol, prompt only", "held-out: ALLaM-7B FT, constrained"),
    ("held-out: gpt-6-astra, prompt only", "held-out: ALLaM-7B FT, constrained"),
    ("Jais-3-32B FT, constrained, final acceptor", "Jais-3-32B FT, free, rerun"),
    ("Jais-3-32B FT, constrained, final acceptor", "ALLaM-7B FT, constrained, final acceptor"),
    ("held-out: Jais-3-32B FT, constrained", "held-out: Jais-3-32B FT, free"),
    ("held-out: Jais-3-32B FT, constrained", "held-out: ALLaM-7B FT, constrained"),
    ("held-out: gpt-6-astra, prompt only", "held-out: Jais-3-32B FT, constrained"),
    ("held-out: claude-fable-5-2, prompt only", "held-out: Jais-3-32B FT, constrained"),
    ("held-out: gpt-6.1-sol, prompt only", "held-out: Jais-3-32B FT, constrained"),
]


def main() -> None:
    corpus = corpus_hemistichs()
    per_prompt: dict[str, dict[str, list[float]]] = {}
    res = {"created_at": utc_now(), "corpus_hemistichs": len(corpus) if corpus else None,
           "systems": {}}
    for name, path, mode in SYSTEMS:
        if not resolve_path(path).is_file():
            continue
        items = json.loads(resolve_path(path).read_text(encoding="utf-8"))["items"]
        agg: dict[str, list[float]] = defaultdict(list)
        per_meter: dict[str, list[float]] = defaultdict(list)
        hem = copied = 0
        for it in items:
            req, r = it["request"], it[mode]
            n = req["line_count"]
            abyat = [x.split("\n") for x in r["poem"].split("\n\n")]
            abyat = [x for x in abyat if len(x) == 2]
            m = req["meter"]
            exact = sum(fsa.complete(s, m, "sadr") + fsa.complete(a, m, "ajuz")
                        for s, a in abyat) / (2 * n)
            agg["exact"].append(exact)
            agg["exact_at_decoding"].append(r["exact"])
            per_meter[req["meter"]].append(exact)
            agg["topic_words"].append(topic_words_used(req["topic"], r["poem"]))
            agg["catt_090"].append(r["catt_090"])
            agg["catt_095"].append(r["catt_095"])
            if r.get("seconds") is not None:
                agg["seconds"].append(r["seconds"])
            agg["unfinished"].append(sum(h == "…" for x in abyat for h in x))
            agg["abyat_written"].append(len(abyat) / n)
            if req.get("rhyme"):
                rx = re.compile(f"{re.escape(req['rhyme'])}(?:[اىيوه]|ه[اوي])?$")
                last = [(strip_diacritics(a).split() or [""])[-1] for _, a in abyat]
                agg["rawi_strict"].append(sum(bool(rx.search(w)) for w in last) / n)
            for x in abyat:
                for h in x:
                    if h != "…":
                        hem += 1
                        copied += bool(corpus) and key(h) in corpus
        row = {k: round(sum(v) / len(v), 3) for k, v in agg.items()}
        row["ci95"] = {k: boot(agg[k]) for k in ("exact", "catt_090", "catt_095", "rawi_strict",
                                                 "topic_words") if agg.get(k)}
        per_prompt[name] = agg
        row.update(poems=len(items), hemistichs=hem,
                   verbatim_in_corpus=copied if corpus else None,
                   exact_by_meter={m: round(sum(v) / len(v), 2)
                                   for m, v in sorted(per_meter.items())})
        res["systems"][name] = row
        print(name, json.dumps(row, ensure_ascii=False))
    res["paired_differences"] = {
        f"{a} minus {b}": {k: {"diff": round(sum(per_prompt[a][k]) / len(per_prompt[a][k])
                                               - sum(per_prompt[b][k]) / len(per_prompt[b][k]), 3),
                               "ci95": boot(per_prompt[a][k], per_prompt[b][k])}
                           for k in ("exact", "catt_090", "rawi_strict", "topic_words")}
        for a, b in PAIRS if a in per_prompt and b in per_prompt}
    for k, v in res["paired_differences"].items():
        print(k, json.dumps(v))
    write_json(resolve_path("paper/results.json"), res)


if __name__ == "__main__":
    main()
