## Manual TODO list for EstNLTK model training

### Priority

~~Train the model on homonym dataset and try to overfit it as much as possible. Try to get the best results for this dataset only. Next step is to do train/test splitting, where we have 5000 examples for train and 2000 examples for test. After that, we evaluate the model on the test set and see if the confusion matrix shows similar results as with training on the whole dataset. The results will be a bit more pessimistic, but we want to see if the model can generalize to unseen data. Train model on the whole dataset and try to get 100 score~~

~~Inspect homonym models confusion matrix results, specifically the confusions between adt and sg g, sg n and sg g, and sg p and sg g. Gather 10 examples for each of these confusions and analyze them to see if there are any patterns or commonalities that could explain why the model is confusing these cases.~~

Overleaf: Write the Table Of Contents and start writing some paragraphs for each section if possible.

Since we have tabular results for the homonym dataset, transfer these results to the Overleaf document and write some analysis for these results.

Inspect and gather senteneces containing homonym words in the Koondkorpus and plot form distribution for these words. Check if our model predicts the most common form for these words in the Koondkorpus. If not, try to find out why.

Weighted training: samples vs weights. Should we sample more from the homonym dataset or should we give more weight to the examples from the homonym dataset? Or maybe a combination of both?

Modify GPT prompt so instead of replacing the homonym word with word "kaas" in correct form, replace it with a pronoun or a synonym.

~~Early stopping criteria: if F1-score on the validation set does not go under 95% for 2 consecutive epochs, stop training.~~

~~Might have to write report for Siim showcasing the results and the process of training the model on the homonym dataset. Explain why we chose so.~~

### BERT and GPT MLM

Do BERT masked prediction. Take TOP 10 predictions for the masked word and do morph analysis. Do case profile (for forms) analysis for these predictions. Check if the correct form is among the TOP 10 predictions and if it is, check if it has the correct case profile. You can filter out cases or labels (cases + forms), where the case does not match with the word at all. The idea is to see whether BERT can predict a word for the masked position that has the correct case profile, for gathering more data for the homonym dataset. Nevertheless, we need to check the same with GPT as well, since it is a more powerful model and might give better results.

Add more constraints to the BERT word predictions using part of speech or prediction scores. For predictions, check if the score is above a certain threshold.

### Morph-syntax conflicts dataset

Take the morph-syntax conflicts dataset and generate annotations for these sentences using Vabamorf and the BERT-based model. Check if the models can disambiguate the cases correctly.

- Save csv with columns: Sentence_id, sentence, word, span, Bert Morph v2 form, Vabamorf form
  - Basically in a way so we can easily see on a monitor without needing to scroll horizontally.
- Filter sentences where Bert Morph v2 form is not in ["n", "p"], pick randomly 20 sentences and inspect them to see if there are any patterns that could explain why the model is predicting these forms.
- Predict more than the most probable morph label for the word. If the most probable label's probability is lower than the sum of the probabilities of the labels "n" and "p", then we can predict "n" or "p" instead of the most probable label. Essentially, we want to see how sure the model is about its predictions and if it is not sure, then we can use the case profile of the word to make a better prediction. For example, if the model predicts "sg g" with 0.4 probability, "sg n" with 0.3 probability and "sg p" with 0.3 probability, then we can predict "sg n" or "sg p" instead of "sg g", since the sum of the probabilities of "sg n" and "sg p" is higher than the probability of "sg g". This way we can potentially improve the predictions for these cases where the model is not very sure about its predictions.
- Predict with Bert Morph v2 model and get the most probable label and sum the predictions of both forms n and p.
- <https://github.com/estnltk/estnltk-model-data/tree/main/morph_tagging/syntax_morph_conflicts>
  1. Take the morph-syntax conflicts dataset, where Bert and Vabamorf differ and where BERT predicts form that is not in ["n", "p"]
  2. Modify MoE model pipeline so it also adds syntax analysis layer.
  3. Now, check if the syntax for the questionable word is subject or not. If the syntax is subject, then make sure to use the baseline model for the prediction.

### Send UD Treebank diffs

Send at <kadri.muischnek@ut.ee> the expert diffs between predicted and gold labels for sentences where expert was selected on the UD treebank test set.

~~### Mixture of experts approach~~

~~We have a new model trained on the whole homonym dataset and an old model that was trained on enc2017 corpus and UD treebank. We can use the new model as an expert for the homonym dataset and the old model as an expert for the enc2017 and UD treebank datasets. We can then use a gating mechanism to decide which model to use for each example. Find a simple gating mechanism that can be implemented easily and does not require a lot of additional training.~~

### Koondkorpus

Now, that we have examples of predictions for the homonym dataset, where the model predicted wrong cases, we can use these examples to find similar sentences containing these words in the Koondkorpus. The problem is that Koondkorpus presumably does not have morphological annotations, but Siim might have some scripts or a way to get the morphological analysis for the sentences in the Koondkorpus.

### Background research

Millistes keeltes käänte homonüümia üldse esineb?
How to use LLM (GPT) to predict cases for words? Morphological analysis. Search for existing research or examples using LLM based tools.
Catastrophic forgetting literature review: does it apply to this dashboard case?

### Conflicts syntax vs morphology dataset

Filter out words with dash suffix like "laenu-" and two or more connected words such as "Monte Carlo" from the conflicts dataset. These will become problematic if we try to use LLM to replace the word with a synonym or a placeholder like "see" or "tema" depending on the subject being animate or inanimate. If the subject is a pronoun then replace the word with a name, preferably a name that has different case forms in Estonian (text is different for each case).

After filtering write a prompt task for LLM to replace the confliting word with a synonym or a placeholder.

Use Vabamorf on the conflicts dataset to find out whether Vabamorf can disambiguate the cases and if it can, are they correct.

1. For getting a morph analysis for the conflicting word, use analyze_token function to get the analysis for the word in the sentence. If the analysis has only one possible case, then Vabamorf can disambiguate the word. If it has more than one possible case, then Vabamorf cannot disambiguate the word.
2. For checking if the analysis is correct, compare the predicted case with the given case in the dataset. If they don't match, then the problem is syntaxtic and not morphological. Otherwise...?

### Notes

Weighted training...

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
