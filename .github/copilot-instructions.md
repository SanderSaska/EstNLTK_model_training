---
description: Instructions for AI agents working on the EstNLTK model training project.
name: EstNLTK Model Training Instructions
applyTo: "**"
---

Project structure overview

- Major components:
    - `data/` — raw and processed corpora (look in `data/enc2017/processed`, `data/ud_est/processed`, `data/homonymous_word_forms/processed`).
    - `notebooks/` — experiment and training notebooks (start with `notebooks/old/04_train_on_UD_EST-EDT_treebank.ipynb` and `notebooks/homonyms_compare_vabamorf.ipynb`).
    - `scripts/` — procedural scripts and utilities (key files: `scripts/bert_morph_tagger.py`, `scripts/est_ud_morph_conv.py`).
    - `models/` — produced model artifacts (each model dir contains `model.safetensors`, `tokenizer`/`sentencepiece.bpe.model`, `config.json`).
    - `outputs/` — evaluation outputs, diagnostics and summary files (used for comparisons and tracking).

Project-specific patterns and conventions

- Data splits: processed datasets live in `*/processed/`. Do not overwrite raw files in `data/*/raw/`.
- Tokenizers: models rely on a SentencePiece BPE model (see `models/*/sentencepiece.bpe.model`). Reuse existing tokenizers when fine-tuning small changes.
- Notebooks as canonical scripts: experiments are often run from notebooks (not CLI). When converting notebook code to scripts, keep the same sequence of cells and document parameter defaults.
- Evaluation artifacts: textual diffs and HTML visualisations in `outputs/` are the primary human-facing diagnostics — preserve their structure when updating evaluation code.

Integration points & dependencies to check

- HuggingFace / transformers-style model objects (artifact layout suggests transformers-compatible checkpoints). Inspect `scripts/bert_morph_tagger.py` for exact training/eval invocations.
- External corpora: `enc2017` and `ud_est` must be present and preprocessed; preprocessing utilities live in `scripts/` and `notebooks/old/`.

Do / Don't (practical rules)

- Do:
  - keep new experiments isolated (new model dir, new output filenames).
  - document changes in `changes.md`.
  - document todo items in `todo.md` under section "Automatic TODO list for EstNLTK model training", and update it as you go.
- Don't: rewrite existing model artifacts in `models/` or overwrite processed data without adding an explicit migration note.

If you're stuck

- Open `todo.md` at the repo root for current priorities and experiment ideas.
- Inspect `notebooks/old/` and `scripts/old/` for runnable examples and parameter choices.
