import re
import pandas as pd
import numpy as np
import estnltk
import pathlib
import sklearn
import sklearn.metrics

from typing import Any, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from tqdm import tqdm
from scripts.model.bert_morph_tagger import BertMorphTagger
from IPython.display import display


# Additional helper functions included before the main APIs:
def _validate_columns(
    results_df: pd.DataFrame,
    pred_col: str,
    true_col: str,
    group_col: str | None = None,
) -> None:
    """Validate that required columns exist in the results dataframe.

    Parameters
    ----------
    results_df : pd.DataFrame
        Dataframe containing evaluation results.
    pred_col : str
        Column name for predicted labels.
    true_col : str
        Column name for true labels.
    group_col : str | None, optional
        Optional column name used for grouped reporting/plotting.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    required_columns: set[str] = {pred_col, true_col}
    if group_col is not None:
        required_columns.add(group_col)

    missing_columns = [col for col in required_columns if col not in results_df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in results dataframe: {missing_columns}"
        )


def _build_group_save_path(save_path: str, group_value: Any) -> str:
    """Build a group-specific save path by appending a safe suffix.

    Parameters
    ----------
    save_path : str
        Base save path from the user.
    group_value : Any
        Group value used in the filename suffix.

    Returns
    -------
    str
        Save path with a group-specific suffix.
    """
    path = pathlib.Path(save_path)
    safe_group = re.sub(r"[^\w\-.]+", "_", str(group_value))
    return str(path.with_name(f"{path.stem}_{safe_group}{path.suffix}"))


def _normalise_true_label(label_value: Any) -> str:
    """Normalise true-label value to a single string label.

    Parameters
    ----------
    label_value : Any
        Raw label value from dataframe row.

    Returns
    -------
    str
        Cleaned label string.
    """
    if label_value is None or pd.isna(label_value):
        return "missing_true_label"

    if isinstance(label_value, tuple):
        if not label_value:
            return "missing_true_label"
        return str(label_value[0])

    if isinstance(label_value, list):
        if not label_value:
            return "missing_true_label"
        return str(label_value[0])

    label_text = str(label_value)
    if label_text.startswith("['") and label_text.endswith("']"):
        return label_text[2:-2]
    if label_text.startswith('["') and label_text.endswith('"]'):
        return label_text[2:-2]

    return label_text


def _extract_prediction_from_layer(
    text: estnltk.Text, layer_name: str, target_span: tuple[int, int]
) -> str | None:
    """Extract first form prediction from an EstNLTK layer for target span.

    Parameters
    ----------
    text : estnltk.Text
        Text object with tagged layers.
    layer_name : str
        Layer name to inspect (e.g., "bert_morph_tagging", "morph_analysis").
    target_span : tuple[int, int]
        Span tuple in format '(start, end)'.

    Returns
    -------
    str | None
        Predicted form label or None if no matching annotation is found.
    """
    layer = getattr(text, layer_name, None)
    if layer is None:
        return None

    for annotation in layer:
        annotation_span = tuple([annotation.start, annotation.end])
        if annotation_span == target_span:
            prediction = annotation.form[0]
            if isinstance(prediction, list):
                prediction = prediction[0] if prediction else None
            return prediction

    return None


def annotate_sentences_with_model(
    input_df: pd.DataFrame,
    model_name: str,
    output_csv_path: str | None = None,
    progress_desc: str = "Evaluating model",
    sentence_col: str = "sentence",
    word_col: str = "word",
    word_span_col: str = "word_span",
    true_label_col: str = "label",
    num_col: str = "num",
    inflection_type_col: str = "inflection_type",
    model_prediction_col: str = "pred_label",
) -> pd.DataFrame:
    """Annotate sentences with selected model or Vabamorf.

    Parameters
    ----------
    input_df : pd.DataFrame
        Input dataframe containing sentence-level data to evaluate.
    model_name : str
        Model path for BertMorphTagger or string "Vabamorf".
    output_csv_path : str | None, optional
        If provided, save results dataframe to this CSV path.
    progress_desc : str, optional
        Description shown in tqdm progress bar.
    sentence_col : str, optional
        Column name containing sentence text.
    word_col : str, optional
        Column name containing analysed word text.
    word_span_col : str, optional
        Column name containing analysed word span.
    true_label_col : str, optional
        Column name containing true label.
    num_col : str, optional
        Column name containing source/group id.
    inflection_type_col : str, optional
        Column name containing inflection type.
    model_prediction_col : str, optional
        Name of output column for selected model predictions.

    Returns
    -------
    pd.DataFrame
        Evaluation results dataframe with prediction columns.
    """
    required_columns = [sentence_col, word_col, word_span_col, true_label_col]
    optional_columns = [num_col, inflection_type_col]
    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in input_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required input columns: {missing_columns}")

    from estnltk.default_resolver import make_resolver

    resolver = make_resolver()
    use_vabamorf_as_model = model_name.strip().lower() == "vabamorf"
    bmt_model = None
    if not use_vabamorf_as_model:
        bmt_model = BertMorphTagger(model_location=str(model_name))

    results: list[dict[str, Any]] = []
    outer = tqdm(
        input_df.iterrows(),
        total=len(input_df),
        desc=progress_desc,
    )

    for _, row in outer:
        sentence_text = row[sentence_col]
        target_span = tuple(row[word_span_col])

        text = estnltk.Text(sentence_text)
        text.tag_layer("sentences")

        if use_vabamorf_as_model:
            text.tag_layer(resolver=resolver)
            model_prediction = _extract_prediction_from_layer(
                text=text,
                layer_name="morph_analysis",
                target_span=target_span,
            )
        else:
            bmt_model.tag(text)
            model_prediction = _extract_prediction_from_layer(
                text=text,
                layer_name="bert_morph_tagging",
                target_span=target_span,
            )

        result_row: dict[str, Any] = {
            "sentence": sentence_text,
            "word": row[word_col],
            "true_label": _normalise_true_label(row[true_label_col]),
            model_prediction_col: model_prediction,
        }

        for optional_column in optional_columns:
            if optional_column in input_df.columns:
                result_row[optional_column] = row[optional_column]

        results.append(result_row)
        outer.refresh()

    results_df = pd.DataFrame(results)

    if output_csv_path is not None:
        output_path = pathlib.Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)

    return results_df


def _print_metrics(y_true: pd.Series, y_pred: pd.Series, title: str) -> None:
    """Compute and print weighted metrics for a prediction task.

    Parameters
    ----------
    y_true : pd.Series
        True labels.
    y_pred : pd.Series
        Predicted labels.
    title : str
        Header title for the printed block.
    """
    accuracy = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    precision, recall, f1_score, _ = sklearn.metrics.precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        average="weighted",
        zero_division=0,
    )

    print(title)
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-score:  {f1_score:.2%}")


def display_metrics_and_classification_report(
    results_df: pd.DataFrame,
    pred_col: str,
    true_col: str,
    group_col: str | None = None,
    show_metrics: bool = True,
    show_classification_report: bool = True,
) -> None:
    """Display evaluation metrics and/or classification reports.

    Parameters
    ----------
    results_df : pd.DataFrame
        Dataframe containing evaluation results.
    pred_col : str
        Column name from which predicted labels are read.
    true_col : str
        Column name from which true labels are read.
    group_col : str | None, optional
        Optional grouping column for per-group results.
    show_metrics : bool, optional
        If True, print weighted metrics (accuracy, precision, recall, F1).
    show_classification_report : bool, optional
        If True, print sklearn classification report.
    """
    _validate_columns(results_df, pred_col, true_col, group_col)

    if not show_metrics and not show_classification_report:
        print(
            "Nothing to display: both show_metrics and show_classification_report are False."
        )
        return

    # Work on a copy and normalise missing values to avoid sklearn errors.
    data = results_df.copy()
    data[pred_col] = data[pred_col].fillna("-")
    data[true_col] = data[true_col].fillna("-")

    def _display_block(block_df: pd.DataFrame, title: str) -> None:
        y_true = block_df[true_col]
        y_pred = block_df[pred_col]

        if show_metrics:
            _print_metrics(y_true=y_true, y_pred=y_pred, title=title)

        if show_classification_report:
            print(f"{title} - Classification Report")
            print(
                sklearn.metrics.classification_report(
                    y_true=y_true,
                    y_pred=y_pred,
                    zero_division=0,
                )
            )

    if group_col is None:
        _display_block(data, "Overall results")
        return

    for group_value in sorted(data[group_col].dropna().unique()):
        group_data = data[data[group_col] == group_value]
        _display_block(group_data, f"{group_col}={group_value}")


def plot_confusion_matrices(
    results_df: pd.DataFrame,
    pred_col: str,
    true_col: str,
    group_col: str | None = None,
    save_path: str | None = None,
    significant_pred_threshold_pct: float = 1.0,
    show_excluded_predictions: bool = True,
) -> None:
    """Plot normalised confusion matrix/matrices with filtered rows.

    Rows always contain only labels present in true labels.
    Predicted-only labels are shown as additional columns only when their
    prediction share is >= `significant_pred_threshold_pct`.
    """
    if significant_pred_threshold_pct < 0:
        raise ValueError("significant_pred_threshold_pct must be >= 0")

    _validate_columns(results_df, pred_col, true_col, group_col)

    data = results_df.copy()
    data[pred_col] = data[pred_col].fillna("-")
    data[true_col] = data[true_col].fillna("-")

    def _show_predicted_only_summary(
        block_df: pd.DataFrame,
        pred_only_labels: list[str],
        significant_pred_only_labels: list[str],
        title: str,
    ) -> None:
        if not show_excluded_predictions or not pred_only_labels:
            return

        pred_counts = block_df[pred_col].value_counts(dropna=False)
        total_count = max(len(block_df), 1)
        summary_df = pd.DataFrame(
            {
                "predicted_only_label": pred_only_labels,
                "count": [int(pred_counts.get(label, 0)) for label in pred_only_labels],
            }
        )
        summary_df["share_pct"] = (summary_df["count"] / total_count * 100).round(2)
        summary_df["included_in_plot"] = summary_df["predicted_only_label"].isin(
            significant_pred_only_labels
        )
        summary_df["threshold_pct"] = significant_pred_threshold_pct
        summary_df = summary_df.sort_values(
            by="share_pct", ascending=False
        ).reset_index(drop=True)

        print(f"{title} - Predicted-only labels (not present in true labels):")
        try:
            print(summary_df)
            print(f"Sum of true labels count: {total_count}")
        except NameError:
            print(summary_df.to_string(index=False))

    def _plot_one(
        block_df: pd.DataFrame, title: str, one_save_path: str | None
    ) -> None:
        y_true = block_df[true_col].astype(str)
        y_pred = block_df[pred_col].astype(str)

        true_labels = sorted(y_true.unique().tolist())
        predicted_labels = sorted(y_pred.unique().tolist())
        pred_only_labels = [
            label for label in predicted_labels if label not in set(true_labels)
        ]

        pred_distribution_pct = y_pred.value_counts(normalize=True).mul(100)
        significant_pred_only_labels = [
            label
            for label in pred_only_labels
            if float(pred_distribution_pct.get(label, 0.0))
            >= significant_pred_threshold_pct
        ]

        plot_pred_labels = true_labels + [
            label for label in significant_pred_only_labels if label not in true_labels
        ]

        cm_counts = pd.crosstab(y_true, y_pred).reindex(
            index=true_labels,
            columns=plot_pred_labels,
            fill_value=0,
        )
        cm = cm_counts.to_numpy(dtype=float)

        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_normalised = np.divide(cm, row_sums, where=row_sums != 0)
        cm_normalised = np.nan_to_num(cm_normalised, nan=0.0)

        plt.figure(figsize=(8, 8))
        sns.heatmap(
            cm_normalised,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=plot_pred_labels,
            yticklabels=true_labels,
        )
        plt.title(
            f"{title} (extra predicted-only columns >= {significant_pred_threshold_pct:.2f}%)"
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        if one_save_path is not None:
            output_path = pathlib.Path(one_save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300)

        plt.show()

        _show_predicted_only_summary(
            block_df=block_df,
            pred_only_labels=pred_only_labels,
            significant_pred_only_labels=significant_pred_only_labels,
            title=title,
        )

    if group_col is None:
        _plot_one(
            block_df=data,
            title=f"{pred_col} vs {true_col}",
            one_save_path=save_path,
        )
        return

    for group_value in sorted(data[group_col].dropna().unique()):
        group_data = data[data[group_col] == group_value]
        group_save_path = None
        if save_path is not None:
            group_save_path = _build_group_save_path(
                save_path=save_path, group_value=group_value
            )

        _plot_one(
            block_df=group_data,
            title=f"Confusion Matrix ({group_col}={group_value}): {pred_col} vs {true_col}",
            one_save_path=group_save_path,
        )


def plot_true_vs_pred_by_inflection(
    df: pd.DataFrame,
    label_col: str = "true_label",
    pred_col: str = "pred_label",
    inflection_col: str = "inflection_type",
    percent: bool = True,
    figsize: Tuple[int, int] = (12, 5),
    width: float = 0.8,
    pred_width_ratio: float = 0.75,
    alpha: float = 0.6,
    hatch_true: str = "/",
    hatch_pred: str = "\\\\",
    color_true: str | None = None,
    color_pred: str | None = None,
    rotate_xticks: int = 45,
    top_n: int | None = None,
    save_dir: Optional[str] = None,
    save_prefix: str = "_true_vs_pred",
    dpi: int = 200,
    save_ext: str = "png",
    plot_per_inflection: bool = True,
    plot_overall: bool = True,
) -> None:
    """
    Plot true vs predicted label distributions.

    Behavior:
    - When `plot_per_inflection` is True, creates one overlaid true-vs-pred
      bar plot per distinct value in `inflection_col` (same behaviour as before).
    - When `plot_overall` is True, additionally creates an overall plot
      (single figure) showing the label distribution across the entire `df`,
      not separated by inflection type.

    Parameters
    ----------
    df
        DataFrame with ground-truth and predictions.
    label_col
        Column name with the true labels.
    pred_col
        Column name with predicted labels.
    inflection_col
        Column name used to split per-inflection plots.
    percent
        If True, show percentages on the y-axis (within each plotted block).
    figsize
        Figure size for each plot.
    width
        Width of the "true" bars.
    pred_width_ratio
        Predicted bars width = width * pred_width_ratio (drawn on top).
    alpha
        Transparency for bars.
    hatch_true, hatch_pred
        Hatch patterns for true and predicted bars (default opposing 45°).
        Note: '\\\\' in source represents a single backslash hatch.
    color_true, color_pred
        Optional colours for true/pred bars; if None, uses a seaborn palette.
    rotate_xticks
        Rotation angle for x tick labels.
    top_n
        If set, limits labels to top-N by combined frequency.
    save_dir
        If provided, each plot will be saved into this directory.
    save_prefix
        Prefix appended to saved filenames.
    dpi
        Save resolution.
    save_ext
        File extension/format for saves (e.g., 'png', 'pdf').
    plot_per_inflection
        If True, produce separate plots per `inflection_col` value.
    plot_overall
        If True, produce an overall plot across the whole dataframe.
    """
    sns.set_style("whitegrid")
    palette = sns.color_palette("muted")
    color_true = color_true or palette[0]
    color_pred = color_pred or palette[1]
    ext = save_ext.lstrip(".")
    savedir_path: Optional[pathlib.Path] = pathlib.Path(save_dir) if save_dir else None
    if savedir_path:
        savedir_path.mkdir(parents=True, exist_ok=True)

    # Work on a copy and normalise missing labels
    data = df.copy()
    data[pred_col] = data[pred_col].fillna("-").astype(str)
    data[label_col] = data[label_col].fillna("-").astype(str)

    def _safe_name(s: str) -> str:
        return re.sub(r"[^\w\-_. ]", "_", str(s))

    def _make_and_show_plot(
        labels: list[str],
        true_vals: pd.Series,
        pred_vals: pd.Series,
        title: str,
        save_name: Optional[str],
    ) -> None:
        """Render and optionally save one overlaid true/pred bar plot."""
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=figsize)

        # Draw true bars (full width)
        ax.bar(
            x,
            true_vals.values,
            width=width,
            color=color_true,
            label="True",
            alpha=alpha,
            hatch=hatch_true,
            edgecolor="black",
            linewidth=0.4,
        )

        # Draw predicted bars slightly narrower on top
        pred_width = width * float(pred_width_ratio)
        ax.bar(
            x,
            pred_vals.values,
            width=pred_width,
            color=color_pred,
            label="Predicted",
            alpha=alpha,
            hatch=hatch_pred,
            edgecolor="black",
            linewidth=0.4,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=rotate_xticks, ha="right")
        ax.set_ylabel("Percentage (%)" if percent else "Count")
        ax.set_title(title)
        if percent:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.legend()
        plt.tight_layout()

        if save_name and savedir_path:
            out_path = savedir_path / save_name
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

        plt.show()
        plt.close(fig)

    def _prepare_counts(block: pd.DataFrame) -> tuple[list[str], pd.Series, pd.Series]:
        """Return labels, true_vals, pred_vals aligned to those labels."""
        true_counts = block[label_col].value_counts()
        pred_counts = block[pred_col].value_counts()
        combined = (true_counts.add(pred_counts, fill_value=0)).sort_values(
            ascending=False
        )
        if top_n is not None:
            combined = combined.iloc[:top_n]
        labels = list(combined.index)
        true_vals = true_counts.reindex(labels, fill_value=0).astype(float)
        pred_vals = pred_counts.reindex(labels, fill_value=0).astype(float)
        if percent:
            total = max(1, len(block))
            true_vals = (true_vals / total) * 100.0
            pred_vals = (pred_vals / total) * 100.0
        return labels, true_vals, pred_vals

    # Overall plot across the whole dataframe (single combined plot)
    if plot_overall:
        labels, true_vals, pred_vals = _prepare_counts(data)
        safe_fn = f"{save_prefix}_overall.{ext}"
        _make_and_show_plot(
            labels=labels,
            true_vals=true_vals,
            pred_vals=pred_vals,
            title=f"Overall True vs Predicted label distribution",
            save_name=safe_fn if savedir_path else None,
        )

    # Per-inflection plots
    if plot_per_inflection:
        inflections = sorted(data[inflection_col].dropna().unique())
        for inf in inflections:
            block = data[data[inflection_col] == inf]
            if block.empty:
                continue
            labels, true_vals, pred_vals = _prepare_counts(block)
            safe_inf = _safe_name(inf)
            safe_fn = f"{save_prefix}_infl_{safe_inf}.{ext}"
            _make_and_show_plot(
                labels=labels,
                true_vals=true_vals,
                pred_vals=pred_vals,
                title=f"True vs Predicted label distribution — {inflection_col} = {inf}",
                save_name=safe_fn if savedir_path else None,
            )


def display_examples(
    dataset: pd.DataFrame,
    pred_col: str | None = None,
    true_col: str | None = None,
    num_examples: int = 5,
    type_of_examples: str = "both",
    display_or_print: str = "display",
    list_of_columns_to_show: list[str] | None = None,
):
    """Display example rows from the dataset for correct and incorrect predictions.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame containing the evaluation results.
    pred_col : str, optional
        Column name for predicted labels. If None, the function will not filter by correctness and will just show the first `num_examples` rows.
    true_col : str, optional
        Column name for true labels. Required if `pred_col` is provided.
    num_examples : int, optional
        Number of examples to show for each category (correct/incorrect).
    type_of_examples : str, optional
        Which examples to show: "correct", "incorrect", or "both". Applies only if `pred_col` and `true_col` are provided.
    list_of_columns_to_show : list[str] | None, optional
        Optional list of column names to include in the displayed examples.
        If None, defaults to showing all columns.
    """

    if list_of_columns_to_show is not None:
        missing_cols = [
            col for col in list_of_columns_to_show if col not in dataset.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Columns specified in list_of_columns_to_show are missing from dataset: {missing_cols}"
            )

    columns_to_show = list_of_columns_to_show or dataset.columns.tolist()

    def _pretty_print_df(df: pd.DataFrame) -> None:
        """Print the DataFrame in a readable format."""
        for i, (idx, row) in enumerate(df.iterrows()):
            if i >= num_examples:
                break
            print(f"Example {i + 1} (index {idx}):")
            for col in columns_to_show:
                print(f"  {col}: {row[col]}")
            print("-" * 40)

    if pred_col and true_col:
        correct_predictions = dataset[dataset[pred_col] == dataset[true_col]]
        incorrect_predictions = dataset[dataset[pred_col] != dataset[true_col]]

        if type_of_examples in ("correct", "both"):
            print(f"\nCorrect Predictions (showing up to {num_examples} examples):")
            if display_or_print == "display":
                display(correct_predictions[columns_to_show].head(num_examples))
            else:
                _pretty_print_df(
                    correct_predictions[columns_to_show].head(num_examples)
                )

        if type_of_examples in ("incorrect", "both"):
            print(f"\nIncorrect Predictions (showing up to {num_examples} examples):")
            if display_or_print == "display":
                display(incorrect_predictions[columns_to_show].head(num_examples))
            else:
                _pretty_print_df(
                    incorrect_predictions[columns_to_show].head(num_examples)
                )
    else:
        if display_or_print == "display":
            display(dataset[columns_to_show].head(num_examples))
        else:
            _pretty_print_df(dataset[columns_to_show].head(num_examples))
