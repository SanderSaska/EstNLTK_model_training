## Manual TODO list for EstNLTK model training

### Confusion matrix, random baseline score (most frequent class or uniform random)

Weighted training...
Train model on the whole dataset and try to get 100 score

How to handle different weights of different data points?

Dashboard

Test sets:

- Vabamorf and BertMorphTagger comparison test set;
- UD treebank test set, enc2017 test set, homonyms test set --- score for each dataset separately
- Gather test set from morphological and syntax different cases in Estonian using GPT to filter these sentences.
- more...

Train set on all 3 datasets

Evaluation on test set score should be higher than in train set

First train on the existing BertMorphTagger model. If time permits, try to train from scratch.

Build train set:

- 3 train sets: UD treebank test set, enc2017 test set, homonyms test set

GPT experiments:

- Morphological analysis with GPT on homonyms test set. Replace the labelled word with "see" (eluta) or "tema"/name (elus) (Find a name that has different writings in different cases) OR replace with synonym. Find or search for alternative methods to augment data.
- Test GPT on 10 sample sentences from homonyms test set and see if the score is over 90%.
- Use few-shot prompting.
- Try to limit API calls to 10€ or even less for GPT.

### Background research

Millistes keeltes käänte homonüümia üldse esineb?
How to use LLM (GPT) to predict cases for words? Morphological analysis. Search for existing research or examples using LLM based tools.
Catastrophic forgetting literature review: does it apply to this dashboard case?

## Automatic TODO list for EstNLTK model training

### Extraction step (corpora-specific and common tasks)

- **enc2017 — Create extractor module:** implement a dedicated extractor in `scripts/preprocessing/enc2017` with a CLI entrypoint.
  - Unpack `enc2017_selection_plain_texts_json.zip` and enumerate `_plain_texts_json/` files.
  - Convert each JSON to an `estnltk` text object using `estnltk.converters.json_to_text`.
  - Normalize metadata (source, doc-id, charset) and save outputs to `data/enc2017/processed/_plain_texts_csv` and JSON-based formats used downstream.
  - Add checksum, file counts and basic validation reports.

- **ud-est-edt — Wrap existing converters:** create a small wrapper in `scripts/preprocessing/ud_et_edt` that reuses `est_ud_morph_conv.py` and `est_ud_utils.py`.
  - Locate CONLLU files inside repository zips under `data/ud_et_edt/raw`.
  - Convert to the reduced Vabamorf format required by downstream pipelines and save to `data/ud_et_edt/processed/UD_converted`.
  - Provide conversion logs and per-file validation.

- **Homonymous sentences — LabelStudio -> estnltk:** implement conversion utilities in `scripts/preprocessing/homonym`.
  - Recreate extraction steps from `notebooks/homonyms_compare_vabamorf.ipynb` to convert LabelStudio JSON annotations back to `estnltk` text objects.
  - Aggregate foldered batches (`annotations/1`, `annotations/2`, ...) into a single processed dataset at `data/homonymous_word_forms/processed`.
  - Normalize annotation configs and provide mapping documentation for label schemas.

- **Common utilities and infra:**
  - Implement robust zip/tar extraction utilities used by all extractors.
  - Define and document a common output schema (fields, JSON structure, required metadata) used by all corpora.
  - Add logging, progress reporting, and resumable extraction support (skip already-processed files).
  - Add unit/functional tests exercising small sample inputs for each extractor.
  - Create a top-level extraction CLI/orchestrator to run corpus-specific extractors (`--corpus enc2017|ud|homonym`).
  - Update `changes.md` and `README.md` with extraction procedure and example commands.

- **Validation & QA:**
  - Implement automated checks: file counts, random sample round-trip conversion, schema validation and checksum comparison.
  - Produce human-readable summary reports into `outputs/` after each extraction run.

- **Deployment & CI:**
  - Add lightweight CI job to run extractor unit tests and a smoke-test that processes 1–5 sample files.
  - Document expected runtime and memory characteristics for each extractor.

### Acceptance criteria for extraction step

- Each corpus has a working extractor CLI that converts raw packaged inputs into the project's standardized processed layout under `data/`.
- Outputs validate against the common JSON schema and pass the smoke-test suite.
- Documentation updated: `README.md`, `changes.md` and `todo.md` reflect implemented steps.

### Preprocessing step (analysis, sampling, transforms)

- **Corpus analysis:**
  - Compute corpus sizes in sentences and approximate word counts for `enc2017`, `ud_et_edt`, and `homonymous_word_forms`.
  - Produce per-corpus statistics: sentence length distribution, token distribution, OOV rates relative to Vabamorf vocabulary, and label/annotation schema summaries.

- **Dataset sizing and sentence-based sampling:**
  - Allow user to specify target dataset size in words (approximate) and have the sampler extract whole sentences until the size is reached or slightly exceeded.
  - Implement sentence-count and word-count estimators used for stopping criteria.

- **Corpus location configuration:**
  - Accept corpus locations via config/CLI (`--enc2017-path`, `--ud-path`, `--homonym-path`) and validate paths before sampling.
  - Support reading from `data/` default layout.

- **Corpus weighting (α, β, γ):**
  - Allow specifying proportions for the three corpora as `(α, β, γ)` summing to 1.
  - Design sampler to respect proportions while picking whole sentences; implement fairness and rounding heuristics to handle indivisible sentence units.
  - Provide option to enforce minimum/maximum corpus contributions and to fall back to proportional sampling when a corpus runs out of data.

- **Preprocessing transforms:**
  - Tokenisation and normalisation steps consistent with EstNLTK expectations (unicode normalization, punctuation handling, whitespace collapsing).
  - Optional morphological analysis pass using Vabamorf for feature alignment.

- **Filtering and deduplication:**
  - Remove near-duplicates and filter out extremely short/long sentences based on configurable thresholds.
  - Optionally filter by sentence-language detection and simple heuristics (e.g., presence of non-Estonian scripts).

- **Splitting and dataset assembly:**
  - Create `train/val/test` splits while preserving corpus proportions and avoiding leakage (same document sentences staying in same split when possible).
  - Save outputs and metadata into `data/<corpus>/processed` and a unified `data/processed` layout for combined datasets.

- **Validation, provenance and QA:**
  - Record exact sampling decisions, random seeds, checksums, and per-file metadata for reproducibility.
  - Implement automated QA checks: final word counts, corpus proportion drift, sample round-trip checks.

- **Tooling, tests and CLI:**
  - Implement a `scripts/preprocessing/common` sampler module and a top-level `preprocess` CLI that accepts `--target-words`, `--alpha`, `--beta`, `--gamma`, and corpus paths.
  - Add unit tests for sampler logic, weighting heuristics and splitting behaviour using small synthetic corpora.
  - Document example commands and expected outputs in `README.md` and `changes.md`.

- **Acceptance criteria for preprocessing:**
  - Sampler extracts whole sentences to meet the requested approximate word count and respects `(α, β, γ)` proportions within configurable tolerances.
  - Outputs include provenance metadata and pass the QA checks; documentation and CLI examples are present.
