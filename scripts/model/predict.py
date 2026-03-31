"""Universal prediction script for token classification models.

This module mirrors the evaluation-style preprocessing and prediction loop,
but returns rich, word-aligned outputs so predictions can be attached back to
the original token-level dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import DataCollatorForTokenClassification

from scripts.model.utils import (
    TokenClassificationDataset,
    _is_placeholder_label,
    _normalise_placeholder_labels,
    _parse_placeholder_labels,
    _select_placeholder_output_label,
    _tokenize_word_for_output,
    initialize_model,
    load_test_df,
    prepare_shared_inputs,
    prepare_token_classification_data,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for prediction script.

    Returns:
            Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="Generate word-aligned predictions for a token classification model."
    )
    parser.add_argument("--test-set", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--second-model-path",
        type=Path,
        default=None,
        help="Path to expert model. If provided, hybrid density-gated mode is used.",
    )
    parser.add_argument(
        "--homonym-list-path",
        type=Path,
        default=Path("../data/homonymous_word_forms/processed/homonymous_words.txt"),
        help=(
            "Path to homonym word list file (required in hybrid mode). "
            "One lowercase token per line."
        ),
    )
    parser.add_argument(
        "--density-threshold",
        type=float,
        default=1.0,
        help=(
            "Sentence-level homonym density threshold T for hybrid gating. "
            "If sentence density >= T, all tokens use expert model."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--sent-id-col", type=str, default="sentence_id")
    parser.add_argument(
        "--source-col",
        type=str,
        default=None,
        help=(
            "Optional source column name. If provided (or if a 'source' column "
            "exists), sentence IDs are normalised per source before grouping."
        ),
    )
    parser.add_argument("--text-col", type=str, default="words")
    parser.add_argument("--label-col", type=str, default="labels")
    parser.add_argument(
        "--test-format",
        choices=["auto", "csv", "parquet"],
        default="auto",
    )
    parser.add_argument(
        "--placeholder-labels",
        type=str,
        default="-,NONE",
        help="Comma-separated placeholder labels (case-insensitive).",
    )
    parser.add_argument(
        "--include-all-tokens",
        action="store_true",
        help="Include full model tokens (with special tokens) in sentence-level output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information.",
    )
    return parser.parse_args()


def load_homonym_set(homonym_list_path: Path) -> set[str]:
    """Load homonymous words from text file.

    Args:
            homonym_list_path: Path to file with one token per line.

    Returns:
            Set of lower-cased homonym words.
    """

    if not homonym_list_path.exists():
        raise FileNotFoundError(
            "Hybrid mode requires a homonym list file, but path was not found: "
            f"{homonym_list_path}"
        )

    with homonym_list_path.open("r", encoding="utf-8") as file_handle:
        return {line.strip().lower() for line in file_handle if line.strip()}


def homonym_density_for_words(words: list[str], homonym_set: set[str]) -> float:
    """Compute homonym density for a sentence.

    Args:
            words: Sentence tokens.
            homonym_set: Known homonymous token set.

    Returns:
            Homonym density in range [0, 1].
    """

    if not words:
        return 0.0
    return sum(1 for word in words if str(word).lower() in homonym_set) / len(words)


def run_single_model_predict(
    model_path: Path | None,
    model_bundle: dict[str, Any] | None,
    sentences: list[tuple[list[str], list[str]]],
    batch_size: int,
    max_length: int | None,
    ignore_placeholders: bool = False,
    placeholder_labels: set[str] | None = None,
    include_all_tokens: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run word-aligned prediction loop for a single model.

    The output preserves one prediction per input word, and includes word-level
    subtoken lists to make word-token alignment explicit.

    Args:
            model_path: Optional model path (used if ``model_bundle`` is None).
            model_bundle: Optional pre-initialized model bundle.
            sentences: Sentence-level input data.
            batch_size: Batch size.
            max_length: Optional max sequence length for preprocessing.
            ignore_placeholders: Whether to encode placeholder labels as ``-100``.
            placeholder_labels: Placeholder labels for alignment behaviour.
            include_all_tokens: Whether to include full sequence tokens in output. This means the output will include the model's tokenization of the input, including special tokens, which can be useful for debugging tokenization and alignment issues.
            verbose: Print progress.

    Returns:
            Dictionary with sentence-level and row-level prediction outputs.
    """

    if verbose:
        print("Initializing model for prediction...")

    normalized_placeholder_labels = _normalise_placeholder_labels(placeholder_labels)
    placeholder_output_label = _select_placeholder_output_label(
        normalized_placeholder_labels
    )

    bundle = initialize_model(str(model_path)) if model_bundle is None else model_bundle
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    id2label = bundle["id2label"]
    label2id = bundle["label2id"]

    if verbose:
        print("Preparing encoded inputs...")

    encodings, aligned_labels = prepare_token_classification_data(
        tokenizer,
        sentences,
        label2id,
        max_length=max_length,
        ignore_placeholders=ignore_placeholders,
        placeholder_labels=normalized_placeholder_labels,
    )

    dataset = TokenClassificationDataset(encodings, aligned_labels)
    collator = DataCollatorForTokenClassification(tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    sentence_outputs: list[dict[str, Any]] = []
    row_outputs: list[dict[str, Any]] = []

    sent_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()
            pred_ids = logits.argmax(axis=-1)

            for batch_row_idx, pred_row in enumerate(pred_ids):
                words, true_labels = sentences[sent_idx]
                word_tokens = [
                    _tokenize_word_for_output(tokenizer, str(word)) for word in words
                ]

                pred_labels: list[str] = []
                aligned_true_labels: list[str] = []
                probabilities: list[float] = []

                token_pointer = 1
                for word_index, (word, true_label) in enumerate(
                    zip(words, true_labels)
                ):
                    current_word_tokens = word_tokens[word_index]
                    token_count = len(current_word_tokens)
                    is_placeholder = _is_placeholder_label(
                        true_label,
                        normalized_placeholder_labels,
                    )

                    if is_placeholder and ignore_placeholders:
                        predicted_label = placeholder_output_label
                        gold_label = placeholder_output_label
                    else:
                        if token_pointer < len(pred_row):
                            predicted_label = id2label.get(
                                int(pred_row[token_pointer]),
                                placeholder_output_label,
                            )
                        else:
                            predicted_label = placeholder_output_label
                        gold_label = str(true_label)

                    pred_labels.append(predicted_label)
                    aligned_true_labels.append(gold_label)

                    # Safely retrieve token-level probability. If the token_pointer
                    # is outside the model's sequence length for this batch row,
                    # use a default probability of 0.0 to avoid IndexError.
                    if token_pointer < logits.shape[1]:
                        probability = float(
                            torch.softmax(
                                torch.tensor(logits[batch_row_idx][token_pointer]), dim=-1
                            ).max()
                        )
                    else:
                        probability = 0.0

                    # Append to per-sentence probability list and row-level outputs.
                    probabilities.append(probability)

                    row_outputs.append(
                        {
                            "sentence_index": sent_idx,
                            "word_index": word_index,
                            "word": str(word),
                            "word_tokens": current_word_tokens,
                            "pred_label": predicted_label,
                            "true_label": gold_label,
                            "probability": probability,
                        }
                    )

                    token_pointer += token_count

                sentence_result: dict[str, Any] = {
                    "sentence_index": sent_idx,
                    "words": [str(word) for word in words],
                    "word_tokens": word_tokens,
                    "pred_labels": pred_labels,
                    "true_labels": aligned_true_labels,
                    "probabilities": probabilities,
                }

                if include_all_tokens:
                    enc = encodings[sent_idx]
                    input_ids = enc["input_ids"]
                    attention_mask = enc["attention_mask"]
                    valid_len = int(sum(attention_mask))
                    sentence_result["all_tokens"] = tokenizer.convert_ids_to_tokens(
                        input_ids[:valid_len]
                    )

                sentence_outputs.append(sentence_result)
                sent_idx += 1

    if verbose:
        print(f"Prediction finished for {len(sentence_outputs)} sentences.")

    return {
        "sentences": sentence_outputs,
        "rows": row_outputs,
    }


def run_hybrid_density_predict(
    baseline_model_path: Path | None,
    baseline_bundle: dict[str, Any] | None,
    expert_model_path: Path | None,
    expert_bundle: dict[str, Any] | None,
    sentences: list[tuple[list[str], list[str]]],
    batch_size: int,
    max_length: int | None,
    density_threshold: float,
    homonym_set: set[str],
    ignore_placeholders: bool = False,
    placeholder_labels: set[str] | None = None,
    include_all_tokens: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run hybrid density-gated prediction using baseline and expert models.

    Gating policy mirrors evaluation mode:
    - If sentence homonym density >= ``density_threshold``: use expert model for all words.
    - Else: use expert model on homonymous words and baseline on other words.

    Args:
            baseline_model_path: Optional path to baseline model.
            baseline_bundle: Optional pre-initialised baseline model bundle.
            expert_model_path: Optional path to expert model.
            expert_bundle: Optional pre-initialised expert model bundle.
            sentences: Sentence-level input data.
            batch_size: Batch size.
            max_length: Optional max sequence length for preprocessing.
            density_threshold: Hybrid gating threshold in [0, 1].
            homonym_set: Set of lower-cased homonymous words.
            ignore_placeholders: Whether to encode placeholder labels as ``-100``.
            placeholder_labels: Placeholder labels for alignment behaviour.
            include_all_tokens: Include model-level tokens in sentence output.
            verbose: Print progress.

    Returns:
            Dictionary with sentence-level outputs, row-level outputs, and gate decisions.
    """

    if verbose:
        print("Initializing baseline model for prediction...")

    normalized_placeholder_labels = _normalise_placeholder_labels(placeholder_labels)
    placeholder_output_label = _select_placeholder_output_label(
        normalized_placeholder_labels
    )

    baseline_bundle = (
        initialize_model(str(baseline_model_path))
        if baseline_bundle is None
        else baseline_bundle
    )
    model_baseline = baseline_bundle["model"]
    tokenizer_baseline = baseline_bundle["tokenizer"]
    id2label_baseline = baseline_bundle["id2label"]
    label2id_baseline = baseline_bundle["label2id"]

    if verbose:
        print("Initializing expert model for prediction...")

    expert_bundle = (
        initialize_model(str(expert_model_path))
        if expert_bundle is None
        else expert_bundle
    )
    model_expert = expert_bundle["model"]
    tokenizer_expert = expert_bundle["tokenizer"]
    id2label_expert = expert_bundle["id2label"]
    label2id_expert = expert_bundle["label2id"]

    if verbose:
        print("Preparing encoded inputs for both models...")

    encodings_baseline, aligned_labels_baseline = prepare_token_classification_data(
        tokenizer_baseline,
        sentences,
        label2id_baseline,
        max_length=max_length,
        ignore_placeholders=ignore_placeholders,
        placeholder_labels=normalized_placeholder_labels,
    )
    encodings_expert, aligned_labels_expert = prepare_token_classification_data(
        tokenizer_expert,
        sentences,
        label2id_expert,
        max_length=max_length,
        ignore_placeholders=ignore_placeholders,
        placeholder_labels=normalized_placeholder_labels,
    )

    dataset_baseline = TokenClassificationDataset(
        encodings_baseline,
        aligned_labels_baseline,
    )
    dataset_expert = TokenClassificationDataset(
        encodings_expert,
        aligned_labels_expert,
    )

    collator_baseline = DataCollatorForTokenClassification(tokenizer_baseline)
    collator_expert = DataCollatorForTokenClassification(tokenizer_expert)

    dataloader_baseline = torch.utils.data.DataLoader(
        dataset_baseline,
        batch_size=batch_size,
        collate_fn=collator_baseline,
    )
    dataloader_expert = torch.utils.data.DataLoader(
        dataset_expert,
        batch_size=batch_size,
        collate_fn=collator_expert,
    )

    densities = [
        homonym_density_for_words(words, homonym_set) for words, _ in sentences
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_baseline.to(device)
    model_expert.to(device)
    model_baseline.eval()
    model_expert.eval()

    sentence_outputs: list[dict[str, Any]] = []
    row_outputs: list[dict[str, Any]] = []
    gate_decisions: list[dict[str, float | int]] = []

    sent_idx = 0
    with torch.no_grad():
        for batch_baseline, batch_expert in zip(dataloader_baseline, dataloader_expert):
            batch_baseline = {
                key: value.to(device) for key, value in batch_baseline.items()
            }
            batch_expert = {
                key: value.to(device) for key, value in batch_expert.items()
            }

            outputs_baseline = model_baseline(**batch_baseline)
            outputs_expert = model_expert(**batch_expert)

            logits_baseline = outputs_baseline.logits.detach().cpu().numpy()
            logits_expert = outputs_expert.logits.detach().cpu().numpy()

            for baseline_row, expert_row in zip(logits_baseline, logits_expert):
                baseline_pred_ids = baseline_row.argmax(axis=-1)
                expert_pred_ids = expert_row.argmax(axis=-1)

                words, true_labels = sentences[sent_idx]
                density = densities[sent_idx]

                word_tokens = [
                    _tokenize_word_for_output(tokenizer_baseline, str(word))
                    for word in words
                ]

                pred_labels: list[str] = []
                aligned_true_labels: list[str] = []
                sources: list[str] = []
                baseline_probabilities: list[float] = []
                expert_probabilities: list[float] = []
                probabilities: list[float] = []

                baseline_pointer = 1
                expert_pointer = 1
                for word_index, (word, true_label) in enumerate(
                    zip(words, true_labels)
                ):
                    baseline_token_count = len(word_tokens[word_index])
                    expert_token_count = len(
                        _tokenize_word_for_output(tokenizer_expert, str(word))
                    )

                    is_placeholder = _is_placeholder_label(
                        true_label,
                        normalized_placeholder_labels,
                    )

                    if is_placeholder and ignore_placeholders:
                        predicted_label = placeholder_output_label
                        gold_label = placeholder_output_label
                        source = "placeholder"
                    else:
                        if baseline_pointer < len(baseline_pred_ids):
                            baseline_pred = id2label_baseline.get(
                                int(baseline_pred_ids[baseline_pointer]),
                                placeholder_output_label,
                            )
                        else:
                            baseline_pred = placeholder_output_label

                        if expert_pointer < len(expert_pred_ids):
                            expert_pred = id2label_expert.get(
                                int(expert_pred_ids[expert_pointer]),
                                placeholder_output_label,
                            )
                        else:
                            expert_pred = placeholder_output_label

                        if density >= density_threshold:
                            predicted_label = expert_pred
                            source = "expert"
                        else:
                            current_word = str(word).lower()
                            if current_word in homonym_set:
                                predicted_label = expert_pred
                                source = "expert"
                            else:
                                predicted_label = baseline_pred
                                source = "baseline"

                        gold_label = str(true_label)

                    pred_labels.append(predicted_label)
                    aligned_true_labels.append(gold_label)
                    sources.append(source)

                    # Compute token-level probabilities for both models (safe bounds).
                    if is_placeholder and ignore_placeholders:
                        baseline_prob = 0.0
                        expert_prob = 0.0
                    else:
                        if baseline_pointer < baseline_row.shape[0]:
                            baseline_prob = float(
                                torch.softmax(
                                    torch.tensor(baseline_row[baseline_pointer]), dim=-1
                                ).max()
                            )
                        else:
                            baseline_prob = 0.0

                        if expert_pointer < expert_row.shape[0]:
                            expert_prob = float(
                                torch.softmax(
                                    torch.tensor(expert_row[expert_pointer]), dim=-1
                                ).max()
                            )
                        else:
                            expert_prob = 0.0

                    # Choose probability according to the selected source.
                    if source == "expert":
                        chosen_prob = expert_prob
                    elif source == "baseline":
                        chosen_prob = baseline_prob
                    else:
                        chosen_prob = 0.0

                    baseline_probabilities.append(baseline_prob)
                    expert_probabilities.append(expert_prob)
                    probabilities.append(chosen_prob)

                    row_outputs.append(
                        {
                            "sentence_index": sent_idx,
                            "word_index": word_index,
                            "word": str(word),
                            "word_tokens": word_tokens[word_index],
                            "pred_label": predicted_label,
                            "true_label": gold_label,
                            "source": source,
                            "density": float(density),
                            "probability": chosen_prob,
                            "baseline_probability": baseline_prob,
                            "expert_probability": expert_prob,
                        }
                    )

                    baseline_pointer += baseline_token_count
                    expert_pointer += expert_token_count

                sentence_result: dict[str, Any] = {
                    "sentence_index": sent_idx,
                    "words": [str(word) for word in words],
                    "word_tokens": word_tokens,
                    "pred_labels": pred_labels,
                    "true_labels": aligned_true_labels,
                    "sources": sources,
                    "density": float(density),
                    "probabilities": probabilities,
                    "baseline_probabilities": baseline_probabilities,
                    "expert_probabilities": expert_probabilities,
                }

                if include_all_tokens:
                    baseline_enc = encodings_baseline[sent_idx]
                    baseline_input_ids = baseline_enc["input_ids"]
                    baseline_attention_mask = baseline_enc["attention_mask"]
                    baseline_valid_len = int(sum(baseline_attention_mask))
                    sentence_result["all_tokens_baseline"] = (
                        tokenizer_baseline.convert_ids_to_tokens(
                            baseline_input_ids[:baseline_valid_len]
                        )
                    )

                    expert_enc = encodings_expert[sent_idx]
                    expert_input_ids = expert_enc["input_ids"]
                    expert_attention_mask = expert_enc["attention_mask"]
                    expert_valid_len = int(sum(expert_attention_mask))
                    sentence_result["all_tokens_expert"] = (
                        tokenizer_expert.convert_ids_to_tokens(
                            expert_input_ids[:expert_valid_len]
                        )
                    )

                sentence_outputs.append(sentence_result)

                gate_decisions.append(
                    {
                        "expert_tokens": sources.count("expert"),
                        "baseline_tokens": sources.count("baseline"),
                        "placeholder_tokens": sources.count("placeholder"),
                        "density": float(density),
                    }
                )

                sent_idx += 1

    if verbose:
        print(f"Hybrid prediction finished for {len(sentence_outputs)} sentences.")

    return {
        "sentences": sentence_outputs,
        "rows": row_outputs,
        "gate_decisions": gate_decisions,
    }


def predictions_to_dataframe(prediction_output: dict[str, Any]) -> pd.DataFrame:
    """Convert row-wise prediction output to dataframe.

    Args:
            prediction_output: Output of ``run_single_model_predict``.

    Returns:
            DataFrame with one row per original word.
    """

    rows = prediction_output.get("rows", [])
    return pd.DataFrame(rows)


def save_prediction_output(output: dict[str, Any], output_path: Path) -> None:
    """Save sentence-level prediction output to JSON.

    Args:
            output: Prediction output dictionary.
            output_path: Target JSON file path.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(output, file_handle, ensure_ascii=False, indent=2)


def main() -> None:
    """CLI entrypoint for prediction generation."""

    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if args.second_model_path is not None and not (
        0.0 <= args.density_threshold <= 1.0
    ):
        raise ValueError("--density-threshold must be between 0 and 1 in hybrid mode.")

    placeholder_labels = _parse_placeholder_labels(args.placeholder_labels)

    test_df = load_test_df(args.test_set, args.test_format)
    sentences = prepare_shared_inputs(
        test_df,
        sent_id_col=args.sent_id_col,
        text_col=args.text_col,
        label_col=args.label_col,
        source_col=args.source_col,
    )

    if args.second_model_path is None:
        output = run_single_model_predict(
            model_path=args.model_path,
            model_bundle=None,
            sentences=sentences,
            batch_size=args.batch_size,
            max_length=args.max_length,
            ignore_placeholders=False,
            placeholder_labels=placeholder_labels,
            include_all_tokens=args.include_all_tokens,
            verbose=args.verbose,
        )
    else:
        homonym_set = load_homonym_set(args.homonym_list_path)
        output = run_hybrid_density_predict(
            baseline_model_path=args.model_path,
            baseline_bundle=None,
            expert_model_path=args.second_model_path,
            expert_bundle=None,
            sentences=sentences,
            batch_size=args.batch_size,
            max_length=args.max_length,
            density_threshold=args.density_threshold,
            homonym_set=homonym_set,
            ignore_placeholders=False,
            placeholder_labels=placeholder_labels,
            include_all_tokens=args.include_all_tokens,
            verbose=args.verbose,
        )

    print(f"Predicted sentences: {len(output['sentences'])}")
    print(f"Predicted word rows: {len(output['rows'])}")

    if args.output is not None:
        save_prediction_output(output, args.output)
        print(f"Saved sentence-level output to: {args.output}")


if __name__ == "__main__":
    main()
