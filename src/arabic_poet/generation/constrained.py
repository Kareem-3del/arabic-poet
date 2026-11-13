"""Meter-constrained decoding: the model writes fully diacritized verse and an 'arud acceptor
(`poetry/arud_fsa.py`) removes, at every step, the tokens that would break the requested meter.

The poem's layout is driven from here (sadr, newline, 'ajuz, blank line), so the model only
chooses words. A hemistich ends when the model picks a newline and the text is a whole legal
hemistich; an 'ajuz must also end in the requested rawi with a rhyme word not used before.
If no candidate token is feasible (a dead end), the hemistich is sampled again from its start,
up to ``max_retries`` times; a hemistich that still fails is reported, never hidden.

With ``lexicon`` (`poetry/lexicon.py`) a token is also dropped when it would finish a word
that is not in the corpus vocabulary or start one that cannot become such a word.

The loop is framework-neutral: it talks to the model through a small backend (feed tokens, get
next-token logits, snapshot and restore the KV cache). `MLXBackend_` is used on the Mac and
`TorchBackend` on a CUDA server (transformers models, e.g. Jais-3 under bitsandbytes).
"""

from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass, field

import numpy as np

from arabic_poet.poetry import arud_fsa as fsa
from arabic_poet.poetry.lexicon import norm
from arabic_poet.poetry.normalization import strip_diacritics


@dataclass
class ConstrainedResult:
    poem: str
    abyat: int
    failed_hemistichs: int
    retries: int
    dead_ends: int
    tokens: int
    seconds: float
    log: list[str] = field(default_factory=list)


class MLXBackend_:
    """KV-cache access for an mlx-lm model."""

    def __init__(self, model):
        from mlx_lm.models.cache import make_prompt_cache

        self.model = model
        self.cache = make_prompt_cache(model)

    def feed(self, ids: list[int]) -> np.ndarray:
        import mlx.core as mx

        out = self.model(mx.array(ids)[None], cache=self.cache)[0, -1, :]
        return np.array(out.astype(mx.float32))

    def snapshot(self):
        return copy.deepcopy(self.cache)

    def restore(self, snap) -> None:
        self.cache[:] = copy.deepcopy(snap)


class TorchBackend:
    """KV-cache access for a transformers causal LM. A snapshot is just the cached length:
    restoring crops the cache back to it, so no tensors are copied."""

    def __init__(self, model):
        self.model = model
        self.past = None
        self.n = 0

    def feed(self, ids: list[int]) -> np.ndarray:
        import torch

        x = torch.tensor([ids], device=self.model.device)
        with torch.no_grad():
            out = self.model(input_ids=x, past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        self.n += len(ids)
        return out.logits[0, -1].float().cpu().numpy()

    def snapshot(self) -> int:
        return self.n

    def restore(self, n: int) -> None:
        if n == self.n:
            return
        if hasattr(self.past, "crop"):
            self.past.crop(n - self.n)  # negative: drop that many tokens (all versions)
        else:  # legacy tuple-of-tuples cache
            self.past = tuple(tuple(t[:, :, :n] for t in layer) for layer in self.past)
        self.n = n


def _rhyme_ok(line: str, rawi: str | None, used: set[str], target: str | None = None) -> bool:
    words = strip_diacritics(line).split()
    if not words:
        return False
    last = words[-1]
    if target:  # the rhyme word was fixed before the bayt was written
        return norm(last) == norm(target)
    if last in used:
        return False
    if not rawi:
        return True
    # After the rawi: nothing, one wasl letter, or a ha' of wasl with its khuruj. Stripping more
    # would accept «راعيا» for a rawi of ع, where the rhyme letter is really the ya'.
    return bool(re.search(f"{re.escape(rawi)}(?:[اىيوه]|ه[اوي])?$", last))


def _overlap(text: str, earlier: list[str]) -> float:
    """Largest share of ``text``'s words already present in one earlier hemistich."""
    words = {norm(w) for w in strip_diacritics(text).split()}
    if not words:
        return 0.0
    return max((len(words & {norm(w) for w in strip_diacritics(e).split()}) / len(words)
                for e in earlier), default=0.0)


def generate(
    model,
    tokenizer,
    prompt: str,
    meter: str,
    n_abyat: int,
    rawi: str | None = None,
    *,
    temperature: float = 0.7,
    top_k: int = 40,
    wide_k: int = 400,
    max_line_tokens: int = 70,
    max_retries: int = 6,
    seed: int = 0,
    constrain: bool = True,
    lexicon=None,
    backend=None,
    prompt_ids: list[int] | None = None,
    stop_ids: frozenset[int] = frozenset(),
    rhyme_words: list[str] | None = None,
    candidates: int = 1,
    select: str = "mass",
    use_meter: bool = True,
    use_rhyme: bool = True,
) -> ConstrainedResult:
    """``backend`` defaults to MLX; pass ``TorchBackend(model)`` for a transformers model.
    ``prompt_ids`` replaces ``tokenizer.encode(prompt)`` (caller controls special tokens).
    ``stop_ids``: extra end-of-turn token ids, never written into a line.
    ``rhyme_words``: one planned rhyme word per bayt; the 'ajuz must end on it.
    ``candidates``: legal hemistichs to collect (within the same retry budget) before one is
    kept. Masking token by token is myopic: a path can stay legal while the model likes it less
    and less. ``select`` names the score that picks among them, both means per token:
    "mass" = log of the model's probability left on legal tokens (how little the mask had to
    remove; the importance weight of SMC steering), "lm" = log-probability the unmasked model
    gave the tokens actually written.
    ``use_meter`` / ``use_rhyme``: switch one constraint off (ablation). With the meter off, a
    constrained line must still be well-formed diacritized text; its length is the model's."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    meter_on, rhyme_on = constrain and use_meter, constrain and use_rhyme
    be = backend if backend is not None else MLXBackend_(model)
    feed = be.feed

    logits = feed(prompt_ids if prompt_ids is not None else tokenizer.encode(prompt))
    lines: list[str] = []
    used: set[str] = set()
    stats = {"failed": 0, "retries": 0, "dead": 0, "tokens": 0}
    log: list[str] = []

    def write_line(half: str, start_logits, target: str | None = None):
        """One attempt at a hemistich. Returns (text or None, its selection score)."""
        ids: list[int] = []
        lg = start_logits
        mass = lm = 0.0
        steps = 0
        for _ in range(max_line_tokens):
            arr = lg
            order = np.argsort(-arr)
            text = tokenizer.decode(ids) if ids else ""
            pending_ok = not text.endswith("\ufffd\ufffd\ufffd")  # never pile up loose bytes
            picked = None
            for k in (top_k, wide_k) if constrain else (top_k,):
                cand, weights = [], []
                for tid in order[:k]:
                    tid = int(tid)
                    new = tokenizer.decode([*ids, tid])
                    if constrain and "ـ" in new:
                        continue  # tatweel pads a word without changing how it scans
                    if "\n" in new or tid == tokenizer.eos_token_id or tid in stop_ids:
                        line = new.split("\n")[0]
                        ok = "\n" in new and new.split("\n", 1)[1].strip() == ""
                        if ok and meter_on:
                            ok = fsa.complete(line, meter, half)
                        elif ok and constrain:
                            ok = fsa.well_formed(line.strip()) and len(line.split()) >= 2
                        if ok and half == "ajuz" and rhyme_on:
                            ok = _rhyme_ok(line, rawi, used, target)
                        if ok and lexicon is not None:
                            ok = lexicon.feasible(line, final=True)
                        if ok:
                            cand.append((tid, True, line))
                            weights.append(arr[tid])
                    elif new.endswith("\ufffd"):
                        # Byte-level tokenizer mid-character: judged once the character is whole.
                        if constrain and new.rstrip("\ufffd") == text and not pending_ok:
                            continue
                        cand.append((tid, False, new))
                        weights.append(arr[tid])
                    elif not new.strip() or new == text:
                        continue  # nothing visible added
                    elif lexicon is not None and not lexicon.feasible(new):
                        continue
                    elif (not constrain or (fsa.feasible(new.lstrip(), meter, half) if meter_on
                                            else fsa.well_formed(new.lstrip()))):
                        cand.append((tid, False, new))
                        weights.append(arr[tid])
                if cand:
                    raw = np.array(weights, dtype=np.float64)
                    w = raw / max(temperature, 1e-3)
                    p = np.exp(w - w.max())
                    j = int(rng.choice(len(cand), p=p / p.sum()))
                    picked = cand[j]
                    if candidates > 1:
                        a64 = arr.astype(np.float64)
                        norm_c = a64.max() + np.log(np.exp(a64 - a64.max()).sum())
                        mass += raw.max() + np.log(np.exp(raw - raw.max()).sum()) - norm_c
                        lm += raw[j] - norm_c
                        steps += 1
                    break
            if picked is None:
                stats["dead"] += 1
                return None, 0.0
            tid, ends, new = picked
            stats["tokens"] += 1
            if ends:
                score = (mass if select == "mass" else lm) / max(steps, 1)
                return new.strip(), score
            ids.append(tid)
            lg = feed([tid])
        return None, 0.0

    for bayt in range(n_abyat):
        for half in ("sadr", "ajuz"):
            snapshot = be.snapshot()
            start = logits
            found: dict[str, float] = {}
            for attempt in range(max_retries + 1):
                if attempt:
                    stats["retries"] += 1
                    be.restore(snapshot)
                target = rhyme_words[bayt] if rhyme_words and half == "ajuz" else None
                got, score = write_line(half, start, target)
                if got:
                    found[got] = max(score, found.get(got, -np.inf))
                    if len(found) >= candidates:
                        break
            # The likeliest line is often a copy of an earlier one: rank overlap first.
            text = None
            if found:
                text = min(found, key=lambda t: (_overlap(t, lines) > 0.5, -found[t]))
            if not text:
                stats["failed"] += 1
                log.append(f"bayt {bayt + 1} {half}: no legal hemistich in {max_retries + 1} tries")
                text = "…"
            # Re-feed the accepted line so the cache holds exactly the poem as written.
            be.restore(snapshot)
            sep = "\n" if half == "sadr" else "\n\n"
            last = bayt == n_abyat - 1 and half == "ajuz"
            logits = feed(tokenizer.encode(text + ("" if last else sep), add_special_tokens=False))
            lines.append(text)
            if half == "ajuz" and text != "…":
                used.add(strip_diacritics(text).split()[-1])
    poem = "\n\n".join(f"{lines[i]}\n{lines[i + 1]}" for i in range(0, len(lines), 2))
    return ConstrainedResult(poem, n_abyat, stats["failed"], stats["retries"], stats["dead"],
                             stats["tokens"], round(time.time() - t0, 1), log)
