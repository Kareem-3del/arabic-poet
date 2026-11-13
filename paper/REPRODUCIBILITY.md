# Reproducibility record

What is pinned, what is published, and what a rerun can and cannot reproduce. Written
13 November 2026.

## Pinned inputs

| Item | Identifier |
|---|---|
| Raw corpus dump (Arabic Wikisource, collected 9 November 2026) | sha256 `8efe0cb44590c29c91b7e4b1f443c41477af7460d318a5a0cc70be7d27cffd8d` |
| Training set tashkeel-v2, `train.jsonl` | sha256 `c0f0fa4f35b509697d07f798ffdc88b385016513fdd33aa4da35d405eeeb8043` |
| tashkeel-v2, `valid.jsonl` | sha256 `1f1590859e48c283919b79bf50c41366b3d81d395874055450a92805dd5f2d43` |
| tashkeel-v2, `test.jsonl` | sha256 `cffc544aec4d8e224cc03dc37c7d17e8053be4c4e13e626dba5530f8f496d41c` |
| Plan-conditioned set tashkeel-v3, `train.jsonl` | sha256 `da006df90c1ed47c2729fbed10feace716eb8d086a587ac5372bc2441e172ecd` |
| Benchmark `data/benchmarks/poetry_prompts_v2.jsonl` (in this repository) | sha256 `6f81ce8e01cb0ec2364254cf319899a1f2237a191ea29cd4f111e10ce13891b6` |
| Held-out benchmark `data/benchmarks/poetry_prompts_v3_heldout.jsonl` (in this repository) | sha256 `40bc30982e76e92128ba42cb25f2331ecdca16091e764ee28e2835c8649a9da3` |
| Acceptor rules frozen before the held-out prompts were written | `paper/data/frozen_rules.txt` |
| ALLaM adapter used in every ALLaM row (MLX `adapters.safetensors`) | sha256 `cf539b0c32b481d580f2e75594c6b12e266b50f21c15b363c493312c551b12ce` |

## Models and tools

| Item | Version |
|---|---|
| ALLaM base | `humain-ai/ALLaM-7B-Instruct-preview`, revision `a28dd1e67420cde72d3629c8633a974cf7d9c366`, converted to MLX 4-bit |
| Jais base | `inception42/Jais-3-32B`, revision `c87efc1f6d94b1f5152ba1d2220aa68cdcb5f4ed`, loaded 4-bit NF4 |
| Hosted models | model ids `gpt-6.1-sol`, `gpt-6-astra` and `claude-fable-5-2`, each queried on 13 November 2026 with default settings; closed, so a later query may differ |
| Diacritizer | CATT encoder-decoder, release v2 ONNX (`abjadai/catt`), zip sha256 `75d2a6360725c7e0bbb238b5a950f621fb8e30b21ad7c65e91e0f36759bcb3ce` |
| Scansion tables and scorer | pyarud 1.0.0 |
| Local inference | mlx 0.32.4, mlx-lm 0.32.2, on an M4 Pro (48 GB) |
| Jais training and decoding | transformers 5.22.2 (>= 4.57 required), peft >= 0.17, bitsandbytes >= 0.48, on one L40S |
| Everything else | `uv.lock` in this repository |

## What is in the repository

- All code, including the acceptor, decoder, builders and scoring scripts.
- The benchmark prompts.
- Every generated poem behind the tables, with per-poem scores (`paper/data/`), including the
  raw responses of the hosted models.
- `paper/results.json`, rebuilt from those files by `scripts/paper_tables.py`.

So every figure in the report can be recomputed from the saved poems without any model.

## What is not, and why

- **Corpus text and training sets.** They are transcriptions from Arabic Wikisource under
  CC BY-SA. The hashes above identify the exact files; a fresh crawl will differ because
  Wikisource changes. Publishing the snapshot is possible under the licence and has not been
  done yet.
- **Adapter weights.** Not published yet. Without them the ALLaM and Jais rows cannot be
  regenerated, only rescored.
- **Hosted models.** Closed. The saved responses are the record.

## What a rerun reproduces

| Step | Reproducible exactly? |
|---|---|
| Scores from saved poems | Yes |
| Constrained decoding with the same adapter, seed and acceptor | Yes on the same MLX version and hardware; small numerical differences are possible elsewhere |
| Training | No bit-exact guarantee (GPU nondeterminism); same recipe and data |
| Hosted-model poems | No |
