"""Universal evaluation script for token classification models.

This module provides two evaluation modes:

1) Single-model evaluation (baseline loop)
2) Two-model hybrid token-level density-based gating evaluation

The implementation intentionally reuses existing helper functions from
``scripts.model.utils`` for model initialisation and preprocessing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import evaluate
import pandas as pd
import torch
from transformers import DataCollatorForTokenClassification

from scripts.model.utils import (
    TokenClassificationDataset,
    _group_sentences_from_df,
    _is_placeholder_label,
    _normalise_placeholder_labels,
    _parse_placeholder_labels,
    _select_placeholder_output_label,
    _token_count_for_word,
    initialize_model,
    load_df,
    prepare_shared_inputs,
    prepare_token_classification_data,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluation script.

    Returns:
            Parsed argument namespace.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate one or two token classification models on a test set."
    )
    parser.add_argument(
        "--test-set",
        type=Path,
        required=True,
        help="Path to test dataset file (CSV or Parquet).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to first model (single-model mode if --second-model-path is omitted).",
    )
    parser.add_argument(
        "--second-model-path",
        type=Path,
        default=None,
        help="Path to second model. If provided, two-model hybrid mode is used.",
    )
    parser.add_argument(
        "--homonym-list-path",
        type=Path,
        default=Path("../data/homonymous_word_forms/processed/homonymous_words.txt"),
        help=(
            "Path to homonym word list file (required in two-model mode). "
            "One lowercase token per line."
        ),
    )
    parser.add_argument(
        "--density-threshold",
        type=float,
        default=1.0,
        help=(
            "Sentence-level homonym density threshold T for two-model hybrid gating. "
            "If sentence density >= T, all tokens use second model."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional max tokenisation length passed to preprocessing.",
    )
    parser.add_argument(
        "--sent-id-col",
        type=str,
        default="sentence_id",
        help="Sentence ID column name in test set.",
    )
    parser.add_argument(
        "--source-col",
        type=str,
        default=None,
        help=(
            "Optional source column name. If provided (or if a 'source' column "
            "exists), sentence IDs are normalised per source before grouping."
        ),
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="words",
        help="Token/text column name in test set.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="labels",
        help="Label column name in test set.",
    )
    parser.add_argument(
        "--test-format",
        choices=["auto", "csv", "parquet"],
        default="auto",
        help="Input file format (auto-detected by suffix by default).",
    )
    parser.add_argument(
        "--placeholder-labels",
        type=str,
        default="-,NONE",
        help="Comma-separated placeholder labels (case-insensitive).",
    )
    parser.add_argument(
        "--average",
        choices=["macro avg", "weighted avg"],
        default="macro avg",
        help="Averaging method for evaluation metrics.",
    )
    return parser.parse_args()


def custom_metrics(
    preds: list[list[str]],
    labels: list[list[str]],
    poseval_metric: Any,
) -> dict[str, Any]:
    """Compute poseval metrics.

    Args:
            preds: Predicted labels per sentence.
            labels: Gold labels per sentence.
            poseval_metric: Loaded Hugging Face poseval metric object.

    Returns:
            Poseval result dictionary.
    """

    return poseval_metric.compute(predictions=preds, references=labels, zero_division=0)


def print_metrics(
    results: dict[str, Any], average: Literal["macro avg", "weighted avg"] = "macro avg"
) -> None:
    """Print weighted aggregate evaluation metrics in notebook-compatible style.

    Args:
            results: Poseval metric result dictionary.
            average: The averaging method for the metrics.
    """

    print(f"Accuracy: \t{results['accuracy']:.2%}")
    print(f"Precision: \t{results[average]['precision']:.2%}")
    print(f"Recall: \t{results[average]['recall']:.2%}")
    print(f"F1-score: \t{results[average]['f1-score']:.2%}")


def get_metrics(
    results: dict[str, Any], average: Literal["macro avg", "weighted avg"] = "macro avg"
) -> dict[str, float]:
    """Extract weighted aggregate evaluation metrics into a dictionary.

    Args:
            results: Poseval metric result dictionary.
            average: The averaging method for the metrics.

    Returns:
            Dictionary with accuracy, precision, recall and F1-score.
    """

    return {
        "accuracy": results["accuracy"],
        "precision": results[average]["precision"],
        "recall": results[average]["recall"],
        "f1-score": results[average]["f1-score"],
    }


def run_single_model_eval(
    model_path: Path | None,
    model_bundle: dict[str, Any] | None,
    sentences: list[tuple[list[str], list[str]]],
    batch_size: int,
    max_length: int | None,
    poseval_metric: Any,
    ignore_placeholders: bool = True,
    placeholder_labels: set[str] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run notebook-equivalent single-model evaluation loop.

    Args:
            model_path: Optional Path to model checkpoint directory.
            model_bundle: Optional pre-initialized model bundle dict (model, tokenizer, id2label, label2id) to reuse across multiple runs.
            sentences: Sentence-level data.
            batch_size: Batch size.
            max_length: Optional max tokenisation length.
            poseval_metric: Loaded poseval metric.
            ignore_placeholders: If True, ignore placeholder tokens (label id -100) in evaluation.
            placeholder_labels: Placeholder labels used by preprocessing and
                output alignment logic.
            verbose: If True, print verbose progress information.

    Returns:
            Dict with predictions, labels and metrics.
    """

    if verbose:
        print("Initializing model...")

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
        if model_path is not None:
            print(f"Model loaded from {model_path} with {len(label2id)} labels.")
        if model_bundle is not None:
            print(f"Model bundle provided with {len(label2id)} labels.")
        print("Preparing data...")

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
        dataset, batch_size=batch_size, collate_fn=collator
    )

    if verbose:
        print(f"Data prepared with {len(dataset)} samples, batch size {batch_size}.")
        print("Running evaluation loop...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds_all: list[list[str]] = []
    labels_all: list[list[str]] = []
    sent_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()
            pred_ids = logits.argmax(axis=-1)

            for pred_row in pred_ids:
                words, sentence_gold_labels = sentences[sent_idx]
                sent_idx += 1

                pred_labels: list[str] = []
                gold_labels: list[str] = []

                token_pointer = 1
                for word, gold_word_label in zip(words, sentence_gold_labels):
                    token_count = _token_count_for_word(tokenizer, str(word))
                    is_placeholder = _is_placeholder_label(
                        gold_word_label,
                        normalized_placeholder_labels,
                    )

                    if is_placeholder and ignore_placeholders:
                        # Do not include placeholder tokens in metric inputs.
                        # Keep pointer movement so token-word alignment remains correct.
                        token_pointer += token_count
                        continue
                    elif is_placeholder and not ignore_placeholders:
                        # For evaluation purposes, treat placeholders as a special case of token-level gating:
                        # if the gold label is a placeholder, use the model prediction for that token regardless of token count.
                        pred_labels.append(placeholder_output_label)
                        gold_labels.append(placeholder_output_label)
                    else:
                        if token_pointer < len(pred_row):
                            pred_label = id2label.get(
                                int(pred_row[token_pointer]),
                                placeholder_output_label,
                            )
                        else:
                            pred_label = placeholder_output_label
                        pred_labels.append(pred_label)
                        gold_labels.append(str(gold_word_label))

                    token_pointer += token_count

                preds_all.append(pred_labels)
                labels_all.append(gold_labels)

    if verbose:
        print("Evaluation loop completed. Computing metrics...")

    results = custom_metrics(preds_all, labels_all, poseval_metric)

    if verbose:
        print("Metrics computed. Evaluation finished.")

    return {"preds": preds_all, "labels": labels_all, "metrics": results}


def load_homonym_set(homonym_list_path: Path) -> set[str]:
    """Load homonymous words from text file.

    Args:
            homonym_list_path: Path to file with one token per line.

    Returns:
            Set of lower-cased homonym words.
    """

    if not homonym_list_path.exists():
        raise FileNotFoundError(
            "Two-model mode requires a homonym list file, but path was not found: "
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


def run_hybrid_density_eval(
    baseline_model_path: Path | None,
    baseline_bundle: dict[str, Any] | None,
    expert_model_path: Path | None,
    expert_bundle: dict[str, Any] | None,
    sentences: list[tuple[list[str], list[str]]],
    batch_size: int,
    max_length: int | None,
    density_threshold: float,
    homonym_set: set[str],
    poseval_metric: Any,
    ignore_placeholders: bool = True,
    placeholder_labels: set[str] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    For each token, if it is a homonymous word (i.e., it appears in the homonym set), use the expert model's prediction for that token; otherwise, use the baseline model's prediction.
    <br>
    Threshold T is used to decide when to apply token-level gating vs. sentence-level gating. For sentences with density above T, apply sentence-level expert selection. For sentences with density below T, apply token-level gating (expert for homonyms, baseline for non-homonyms).
    <br>
        Gating policy:
    <ul>
        <li> If sentence density >= T: use expert model for all sentence tokens.
        <li> Else: use expert only on homonymous words, baseline on other words.
    </ul>

    Args:
            baseline_model_path: Optional Path to baseline model.
            baseline_bundle: Optional pre-initialized baseline model bundle dict (model, tokenizer, id2label, label2id) to reuse across multiple runs.
            expert_model_path: Optional Path to expert model.
            expert_bundle: Optional pre-initialized expert model bundle dict (model, tokenizer, id2label, label2id) to reuse across multiple runs.
            sentences: Sentence-level data.
            batch_size: Batch size.
            max_length: Optional max tokenisation length.
            density_threshold: Gating threshold T.
            homonym_set: Set of homonymous words.
            poseval_metric: Loaded poseval metric.
            ignore_placeholders: If True, ignore placeholder tokens (label id -100) in evaluation.
            placeholder_labels: Placeholder labels used by preprocessing and
                output alignment logic.
            verbose: Whether to print verbose output.

    Returns:
            Dict with predictions, labels, per-token source selections and metrics.
    """

    if verbose:
        print("Initializing baseline model...")

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
        print(
            f"Baseline model loaded from {baseline_model_path} with {len(label2id_baseline)} labels."
        )
        print("Initializing expert model...")

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
        print(
            f"Expert model loaded from {expert_model_path} with {len(label2id_expert)} labels."
        )
        print("Preparing data for both models...")

    encodings_baseline, aligned_labels_baseline = prepare_token_classification_data(
        tokenizer_baseline,
        sentences,
        label2id_baseline,
        max_length=max_length,
        ignore_placeholders=ignore_placeholders,
        placeholder_labels=normalized_placeholder_labels,
    )

    if verbose:
        print("Baseline data prepared. Preparing expert data...")

    encodings_expert, aligned_labels_expert = prepare_token_classification_data(
        tokenizer_expert,
        sentences,
        label2id_expert,
        max_length=max_length,
        ignore_placeholders=ignore_placeholders,
        placeholder_labels=normalized_placeholder_labels,
    )

    if verbose:
        print("Expert data prepared. Creating datasets and dataloaders...")

    dataset_baseline = TokenClassificationDataset(
        encodings_baseline, aligned_labels_baseline
    )
    dataset_expert = TokenClassificationDataset(encodings_expert, aligned_labels_expert)

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

    if verbose:
        print(
            f"Data prepared with {len(dataset_baseline)} samples, batch size {batch_size}."
        )
        print("Gathering homonym density information for gating decisions...")

    densities = [
        homonym_density_for_words(words, homonym_set) for words, _ in sentences
    ]

    if verbose:
        print("Homonym densities computed. Running hybrid evaluation loop...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_baseline.to(device)
    model_expert.to(device)
    model_baseline.eval()
    model_expert.eval()

    preds_all: list[list[str]] = []
    labels_all: list[list[str]] = []
    sources_all: list[list[str]] = []
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

                sentence_words, sentence_gold_word_labels = sentences[sent_idx]
                density = densities[sent_idx]
                sent_idx += 1

                selected_preds: list[str] = []
                sentence_labels: list[str] = []
                sentence_sources: list[str] = []
                placeholder_token_count = 0

                baseline_pointer = 1
                expert_pointer = 1
                for word, gold_word_label in zip(
                    sentence_words, sentence_gold_word_labels
                ):
                    baseline_token_count = _token_count_for_word(
                        tokenizer_baseline,
                        str(word),
                    )
                    expert_token_count = _token_count_for_word(
                        tokenizer_expert,
                        str(word),
                    )

                    is_placeholder = _is_placeholder_label(
                        gold_word_label,
                        normalized_placeholder_labels,
                    )
                    if is_placeholder and ignore_placeholders:
                        # Do not include placeholder tokens in metric inputs.
                        # Keep separate count for gate diagnostics.
                        placeholder_token_count += 1
                    elif is_placeholder and not ignore_placeholders:
                        # For evaluation purposes, treat placeholders as a special case of token-level gating:
                        # if the gold label is a placeholder, use the model prediction for that token regardless of token count or homonym status.
                        selected_preds.append(placeholder_output_label)
                        sentence_labels.append(placeholder_output_label)
                        sentence_sources.append("placeholder")
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
                            selected_preds.append(expert_pred)
                            sentence_sources.append("expert")
                        else:
                            current_word = str(word).lower()
                            if current_word in homonym_set:
                                selected_preds.append(expert_pred)
                                sentence_sources.append("expert")
                            else:
                                selected_preds.append(baseline_pred)
                                sentence_sources.append("baseline")

                        sentence_labels.append(str(gold_word_label))

                    baseline_pointer += baseline_token_count
                    expert_pointer += expert_token_count

                preds_all.append(selected_preds)
                labels_all.append(sentence_labels)
                sources_all.append(sentence_sources)
                if ignore_placeholders:
                    gate_decisions.append(
                        {
                            "expert_tokens": sentence_sources.count("expert"),
                            "baseline_tokens": sentence_sources.count("baseline"),
                            "placeholder_tokens": placeholder_token_count,
                            "density": float(density),
                        }
                    )
                else:
                    gate_decisions.append(
                        {
                            "expert_tokens": sentence_sources.count("expert"),
                            "baseline_tokens": sentence_sources.count("baseline"),
                            "placeholder_tokens": sentence_sources.count("placeholder"),
                            "density": float(density),
                        }
                    )

    if verbose:
        print("Hybrid evaluation loop completed. Computing metrics...")

    results = custom_metrics(preds_all, labels_all, poseval_metric)

    if verbose:
        print("Metrics computed. Evaluation finished.")

    return {
        "preds": preds_all,
        "labels": labels_all,
        "sources": sources_all,
        "gate_decisions": gate_decisions,
        "metrics": results,
    }


def print_hybrid_gate_stats(
    gate_decisions: list[dict[str, float | int]],
    density_threshold: float,
) -> None:
    """Print gate summary for hybrid evaluation mode.

    Args:
            gate_decisions: Per-sentence gate summary list.
            density_threshold: Configured gate threshold.
    """

    total_sentences = len(gate_decisions)
    total_tokens = sum(
        int(decision["expert_tokens"])
        + int(decision["baseline_tokens"])
        + int(decision.get("placeholder_tokens", 0))
        for decision in gate_decisions
    )
    total_expert = sum(int(decision["expert_tokens"]) for decision in gate_decisions)
    total_baseline = sum(
        int(decision["baseline_tokens"]) for decision in gate_decisions
    )
    total_placeholder = sum(
        int(decision.get("placeholder_tokens", 0)) for decision in gate_decisions
    )
    expert_share = (total_expert / total_tokens) if total_tokens else 0.0
    baseline_share = (total_baseline / total_tokens) if total_tokens else 0.0
    placeholder_share = (total_placeholder / total_tokens) if total_tokens else 0.0

    print("\n=== Hybrid Gate Stats ===")
    print(f"Threshold T: {density_threshold}")
    print(f"Total sentences: {total_sentences}")
    print(f"Total tokens: {total_tokens}")
    print(f"Expert tokens: {total_expert} ({expert_share:.1%})")
    print(f"Baseline tokens: {total_baseline} ({baseline_share:.1%})")
    print(f"Placeholder tokens: {total_placeholder} ({placeholder_share:.1%})")


def main() -> None:
    """Entry point for CLI execution."""

    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if args.second_model_path is not None and not (
        0.0 <= args.density_threshold <= 1.0
    ):
        raise ValueError(
            "--density-threshold must be between 0 and 1 in two-model mode."
        )

    poseval_metric = evaluate.load("evaluate-metric/poseval", module_type="metric")
    placeholder_labels = _parse_placeholder_labels(args.placeholder_labels)

    test_df = load_df(args.test_set, args.test_format)
    sentences = prepare_shared_inputs(
        test_df,
        sent_id_col=args.sent_id_col,
        text_col=args.text_col,
        label_col=args.label_col,
        source_col=args.source_col,
    )

    if args.second_model_path is None:
        output = run_single_model_eval(
            model_path=args.model_path,
            model_bundle=None,
            sentences=sentences,
            batch_size=args.batch_size,
            max_length=args.max_length,
            poseval_metric=poseval_metric,
            placeholder_labels=placeholder_labels,
        )
        print_metrics(output["metrics"])
    else:
        homonym_set = load_homonym_set(args.homonym_list_path)
        output = run_hybrid_density_eval(
            baseline_model_path=args.model_path,
            baseline_bundle=None,
            expert_model_path=args.second_model_path,
            expert_bundle=None,
            sentences=sentences,
            batch_size=args.batch_size,
            max_length=args.max_length,
            density_threshold=args.density_threshold,
            homonym_set=homonym_set,
            poseval_metric=poseval_metric,
            placeholder_labels=placeholder_labels,
        )
        print_metrics(output["metrics"])
        print_hybrid_gate_stats(
            gate_decisions=output["gate_decisions"],
            density_threshold=args.density_threshold,
        )


if __name__ == "__main__":
    main()
