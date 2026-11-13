# Scripts

Run from the repository root with `uv run python scripts/<name>`.


## Decoding and evaluation

- `eval_constrained.py`: Benchmark the tashkeel adapter with and without the 'arud constraint.
- `eval_planned.py`: Does the model follow a per-bayt plan under the constraint? (oracle plans)
- `rescore_constrained.py`: Rescore poems produced on a remote machine with the same measures as eval_constrained.py.
- `eval_api_direct.py`: Ask a hosted chat model for the benchmark poems directly (no draft, no constraint) and save them in the constrained-result layout (requests run in parallel; a rerun resumes), so rescore_constrained.py and import_constrained_run.py apply unchanged. Only the benchmark request text is sent; nothing about the user or project.
- `eval_claude_direct.py`: Ask a Claude model for the benchmark poems directly (prompt only, no constraint, one sample each) and save them in the constrained-result layout, like eval_api_direct.py does for the other hosted model. Only the benchmark request text is sent.
- `eval_shaer_published.py`: Score the Shaer authors' published generations with this project's strict meter checker.

## Training data

- `build_tashkeel_sft.py`: Build a fully diacritized SFT set in which every bayt scans exactly in the poem's meter.
- `tashkeel_probe.py`: Probe: how many training abyat scan in their poem's meter after full diacritization?
- `build_plan_sft.py`: Build a plan-conditioned SFT set (tashkeel-v3) from a diacritized one (tashkeel-v2).
- `build_rhyme_inventory.py`: Rhyme words attested in the training poems, per (meter, rawi), with counts.

## Human evaluation

- `build_blind_eval.py`: Assemble a blind human evaluation from the saved poems of every system.
- `build_rating_workbook.py`: Write the blind rating workbook that goes to human raters (Arabic, right-to-left).
- `human_eval.py`: Summarize the blind human ratings: per-system means for each rater, agreement between the raters, and paired contrasts with prompt-level bootstrap intervals.
- `build_prosody_gold_sheet.py`: Prepare a sheet for a prosodist: abyat to judge by hand, so the checkers can be measured against expert labels instead of against their own pass rates.

## Report

- `paper_tables.py`: Recompute every figure of the paper's main table from the saved poems, with one code path.
- `paper_figures.py`: Draw the report's charts as SVG files (English and Arabic labels) from paper/results.json and the saved experiment files. Plain SVG, no plotting library: text is laid out by the viewer, so Arabic shapes correctly.
- `build_paper_pdf.py`: Typeset the report as PDF: Markdown -> designed HTML -> headless Chrome print.
