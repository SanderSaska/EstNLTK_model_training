"""Universal evaluation script for token classification models.

This module provides two evaluation modes:

1) Single-model evaluation (baseline loop)
2) Two-model hybrid token-level density-based gating evaluation

The implementation intentionally reuses existing helper functions from
``scripts.model.model`` for model initialisation and preprocessing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import evaluate
import pandas as pd
import torch
from transformers import DataCollatorForTokenClassification

from scripts.model.model import (
    TokenClassificationDataset,
    _group_sentences_from_df,
    initialize_model,
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
    return parser.parse_args()


def load_test_df(path: Path, file_format: str) -> pd.DataFrame:
    """Load evaluation dataset from CSV or Parquet.

    Args:
            path: Path to input file.
            file_format: One of ``auto``, ``csv`` or ``parquet``.

    Returns:
            Loaded dataframe.
    """

    if not path.exists():
        raise FileNotFoundError(f"Test set not found: {path}")

    resolved_format = file_format
    if file_format == "auto":
        suffix = path.suffix.lower()
        if suffix == ".csv":
            resolved_format = "csv"
        elif suffix in {".parquet", ".pq"}:
            resolved_format = "parquet"
        else:
            raise ValueError(
                f"Could not infer file format from suffix '{suffix}'. "
                "Use --test-format explicitly."
            )

    if resolved_format == "csv":
        return pd.read_csv(path)
    if resolved_format == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {resolved_format}")


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


def print_metrics(results: dict[str, Any]) -> None:
    """Print weighted aggregate evaluation metrics in notebook-compatible style.

    Args:
            results: Poseval metric result dictionary.
    """

    print(f"Accuracy: \t{results['accuracy']:.2%}")
    print(f"Precision: \t{results['weighted avg']['precision']:.2%}")
    print(f"Recall: \t{results['weighted avg']['recall']:.2%}")
    print(f"F1-score: \t{results['weighted avg']['f1-score']:.2%}")


def prepare_shared_inputs(
    test_df: pd.DataFrame,
    sent_id_col: str,
    text_col: str,
    label_col: str,
) -> list[tuple[list[str], list[str]]]:
    """Build sentence-level inputs used by both evaluation modes.

    Args:
            test_df: Token-level dataframe.
            sent_id_col: Sentence id column.
            text_col: Token text column.
            label_col: Label column.

    Returns:
            List of tuples ``(words, labels)`` grouped by sentence.
    """

    missing_cols = [
        col for col in (sent_id_col, text_col, label_col) if col not in test_df.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing required test-set columns: {missing_cols}")

    eval_df = test_df[[sent_id_col, text_col, label_col]].copy()
    eval_df.columns = ["sentence_id", "words", "labels"]
    return _group_sentences_from_df(eval_df)


def run_single_model_eval(
    model_path: Path | None,
    model_bundle: dict[str, Any] | None,
    sentences: list[tuple[list[str], list[str]]],
    batch_size: int,
    max_length: int | None,
    poseval_metric: Any,
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
            verbose: If True, print verbose progress information.

    Returns:
            Dict with predictions, labels and metrics.
    """

    if verbose:
        print("Initializing model...")

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
        ignore_placeholders=False,
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
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = batch["labels"].detach().cpu().numpy()
            pred_ids = logits.argmax(axis=-1)

            for pred_row, label_row in zip(pred_ids, label_ids):
                pred_labels: list[str] = []
                gold_labels: list[str] = []
                for pred_id, gold_id in zip(pred_row, label_row):
                    if gold_id == -100:
                        continue
                    pred_labels.append(id2label[int(pred_id)])
                    gold_labels.append(id2label[int(gold_id)])
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
            verbose: Whether to print verbose output.

    Returns:
            Dict with predictions, labels, per-token source selections and metrics.
    """

    if verbose:
        print("Initializing baseline model...")

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
        ignore_placeholders=False,
    )

    if verbose:
        print("Baseline data prepared. Preparing expert data...")

    encodings_expert, aligned_labels_expert = prepare_token_classification_data(
        tokenizer_expert,
        sentences,
        label2id_expert,
        max_length=max_length,
        ignore_placeholders=False,
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
            label_ids = batch_baseline["labels"].detach().cpu().numpy()

            for baseline_row, expert_row, label_row in zip(
                logits_baseline, logits_expert, label_ids
            ):
                baseline_pred_ids = baseline_row.argmax(axis=-1)
                expert_pred_ids = expert_row.argmax(axis=-1)

                sentence_words = sentences[sent_idx][0]
                density = densities[sent_idx]
                sent_idx += 1

                selected_preds: list[str] = []
                sentence_labels: list[str] = []
                sentence_sources: list[str] = []

                word_pos = 0
                for baseline_pred_id, expert_pred_id, gold_id in zip(
                    baseline_pred_ids, expert_pred_ids, label_row
                ):
                    if gold_id == -100:
                        continue

                    baseline_pred = id2label_baseline[int(baseline_pred_id)]
                    expert_pred = id2label_expert[int(expert_pred_id)]
                    gold_label = id2label_baseline[int(gold_id)]

                    if density >= density_threshold:
                        selected_preds.append(expert_pred)
                        sentence_sources.append("expert")
                    else:
                        current_word = str(sentence_words[word_pos]).lower()
                        if current_word in homonym_set:
                            selected_preds.append(expert_pred)
                            sentence_sources.append("expert")
                        else:
                            selected_preds.append(baseline_pred)
                            sentence_sources.append("baseline")

                    sentence_labels.append(gold_label)
                    word_pos += 1

                preds_all.append(selected_preds)
                labels_all.append(sentence_labels)
                sources_all.append(sentence_sources)
                gate_decisions.append(
                    {
                        "expert_tokens": sentence_sources.count("expert"),
                        "baseline_tokens": sentence_sources.count("baseline"),
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
        int(decision["expert_tokens"]) + int(decision["baseline_tokens"])
        for decision in gate_decisions
    )
    total_expert = sum(int(decision["expert_tokens"]) for decision in gate_decisions)
    total_baseline = sum(
        int(decision["baseline_tokens"]) for decision in gate_decisions
    )
    expert_share = (total_expert / total_tokens) if total_tokens else 0.0
    baseline_share = (total_baseline / total_tokens) if total_tokens else 0.0

    print("\n=== Hybrid Gate Stats ===")
    print(f"Threshold T: {density_threshold}")
    print(f"Total sentences: {total_sentences}")
    print(f"Total tokens: {total_tokens}")
    print(f"Expert tokens: {total_expert} ({expert_share:.1%})")
    print(f"Baseline tokens: {total_baseline} ({baseline_share:.1%})")


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

    test_df = load_test_df(args.test_set, args.test_format)
    sentences = prepare_shared_inputs(
        test_df,
        sent_id_col=args.sent_id_col,
        text_col=args.text_col,
        label_col=args.label_col,
    )

    if args.second_model_path is None:
        output = run_single_model_eval(
            model_path=args.model_path,
            sentences=sentences,
            batch_size=args.batch_size,
            max_length=args.max_length,
            poseval_metric=poseval_metric,
        )
        print_metrics(output["metrics"])
    else:
        homonym_set = load_homonym_set(args.homonym_list_path)
        output = run_hybrid_density_eval(
            baseline_model_path=args.model_path,
            expert_model_path=args.second_model_path,
            sentences=sentences,
            batch_size=args.batch_size,
            max_length=args.max_length,
            density_threshold=args.density_threshold,
            homonym_set=homonym_set,
            poseval_metric=poseval_metric,
        )
        print_metrics(output["metrics"])
        print_hybrid_gate_stats(
            gate_decisions=output["gate_decisions"],
            density_threshold=args.density_threshold,
        )


if __name__ == "__main__":
    main()
