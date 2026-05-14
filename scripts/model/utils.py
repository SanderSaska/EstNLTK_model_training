import gc
import json
import typing
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    RobertaForTokenClassification,
    RobertaTokenizer,
)


def initialize_model(
    model_name: str,
    unique_labels: typing.Optional[typing.List[str]] = None,
    use_Roberta_tokenizer: bool = False,
    use_fast_tokenizer: typing.Optional[bool] = None,
    cleanup: bool = True,
) -> typing.Dict[str, typing.Any]:
    """
    Initialise a token-classification model (for NER) using Hugging Face transformers.

    Args:
        model_name: Pretrained model name or local path (e.g. 'camembert-base').
        unique_labels: All possible label strings the model will predict.
        use_Roberta_tokenizer: If True, uses RobertaTokenizerFast instead of AutoTokenizer (for better handling of RoBERTa-like tokenization).
        use_fast_tokenizer: Whether to use a fast tokenizer implementation. If None,
            defaults to SimpleTransformers-compatible behaviour for CamemBERT
            checkpoints (`False`), and `True` for other model types.
        cleanup: If True, attempts to free previous model memory before initialisation.

    Returns:
        A dict with keys:
        - 'model': AutoModelForTokenClassification
        - 'tokenizer': AutoTokenizer
        - 'config': AutoConfig
        - 'label2id': dict[str,int]
        - 'id2label': dict[int,str]
        - 'device': torch.device
        - 'training_args': small dict with training hyperparameters
    """
    if cleanup and "model" in globals():
        try:
            del globals()["model"]
        except Exception:
            pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build label mappings expected by transformers config if provided
    # `label2id` maps label string -> integer id (used by the model)
    # `id2label` maps integer id -> label string (used for decoding predictions)
    label2id = None
    id2label = None
    if unique_labels is not None:
        label2id = {label: i for i, label in enumerate(unique_labels)}
        id2label = {i: label for label, i in label2id.items()}

    # Load config, tokenizer and model
    # Prefer loading existing config from the checkpoint and update label mappings if needed.
    try:
        config = AutoConfig.from_pretrained(model_name)
    except Exception:
        # Fallback: create config with desired number of labels
        # If reading a config from the checkpoint fails, the intention here is
        # to at least create a config that has the correct `num_labels` so
        # downstream code can construct a classification head. Using
        # `AutoConfig.from_pretrained` again may still try to load metadata
        # from the checkpoint; consider constructing a fresh config if needed.
        config = AutoConfig.from_pretrained(model_name, num_labels=len(unique_labels))

    cfg_num_labels = getattr(config, "num_labels", None)
    cfg_id2label = getattr(config, "id2label", None)
    cfg_label2id = getattr(config, "label2id", None)

    # Canonicalise label mappings: prefer mappings stored in the checkpoint
    # but fall back to `unique_labels` when necessary. The goal is to
    # produce two canonical dicts used by HF models: `id2label: {int: str}`
    # and `label2id: {str: int}` regardless of how they were stored.
    id2label: typing.Dict[int, str] = {}
    label2id: typing.Dict[str, int] = {}

    if isinstance(cfg_id2label, dict) and len(cfg_id2label) > 0:
        # `config.id2label` is the most explicit source: it directly maps
        # numeric ids to labels. When saved to JSON the keys may be strings,
        # so convert them back to ints here and build the inverse mapping.
        try:
            id2label = {int(k): str(v) for k, v in cfg_id2label.items()}
            label2id = {str(v): int(k) for k, v in cfg_id2label.items()}
            warnings.warn("Using label mapping from model `id2label` in checkpoint.")
        except Exception as exc:
            raise ValueError(
                "config.id2label exists but has unexpected format"
            ) from exc
    elif isinstance(cfg_label2id, dict) and len(cfg_label2id) > 0:
        # `config.label2id` is the alternate form (label -> id). Convert
        # its values to ints and build the reverse mapping. This branch is
        # used when `id2label` is not present in the checkpoint.
        try:
            label2id = {str(k): int(v) for k, v in cfg_label2id.items()}
            id2label = {int(v): str(k) for k, v in cfg_label2id.items()}
            warnings.warn("Using label mapping from model `label2id` in checkpoint.")
        except Exception as exc:
            raise ValueError(
                "config.label2id exists but has unexpected format"
            ) from exc
    else:
        # No mapping in checkpoint: require unique_labels to be provided
        if unique_labels is None:
            raise ValueError(
                "Model checkpoint has no label mapping and no unique_labels were provided."
            )
        label2id = {str(lbl): i for i, lbl in enumerate(unique_labels)}
        id2label = {i: lbl for lbl, i in label2id.items()}

    # Ensure the config reflects the chosen mapping and number of labels.
    # HF's JSON representation uses string keys, so write the mappings back
    # using `str(key)` for compatibility with `save_pretrained()`.
    config.num_labels = len(id2label)
    config.id2label = {str(k): v for k, v in id2label.items()}
    config.label2id = {str(k): v for k, v in label2id.items()}

    if use_fast_tokenizer is None:
        model_type = str(getattr(config, "model_type", "")).lower()
        use_fast_tokenizer = model_type != "camembert"

    if use_Roberta_tokenizer:
        tokenizer = RobertaTokenizer.from_pretrained(
            model_name, use_fast=use_fast_tokenizer
        )
        model = RobertaForTokenClassification.from_pretrained(model_name, config=config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=use_fast_tokenizer
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, config=config
        )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "label2id": label2id,
        "id2label": id2label,
        "device": device,
    }


class TokenClassificationDataset(Dataset):
    def __init__(
        self, encodings: typing.List[dict], labels: typing.List[typing.List[int]]
    ):
        # encodings: list of dicts produced by tokenizer (input_ids, attention_mask, etc.)
        # labels: parallel list of label id sequences (aligned to tokens)
        # We store them as-is and convert to tensors in `__getitem__` so that the
        # dataset itself remains lightweight and serialization-friendly.
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        # Number of examples equals number of encoding dicts
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict:
        # Convert lists to tensors for the DataLoader. Only include the
        # commonly used keys expected by transformers models; keep `labels`
        # as a tensor under the `labels` key so the model's forward() will
        # compute loss when provided.
        item = {
            k: torch.tensor(v)
            for k, v in self.encodings[idx].items()
            if k in ("input_ids", "attention_mask", "token_type_ids")
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def _group_sentences_from_df(
    df,
    text_col: str = "words",
    label_col: str = "labels",
    sent_id_col: str = "sentence_id",
    source_col: typing.Optional[str] = None,
):
    """
    Convert DataFrame with token-level rows into list-of-tokens and list-of-labels per sentence.

    Args:
        df: Token-level DataFrame.
        text_col: Token text column name.
        label_col: Label column name.
        sent_id_col: Sentence ID column name.
        source_col: Optional source column name. If provided and present in
            ``df``, grouping is done by ``(source_col, sent_id_col)`` to avoid
            mixing sentences that share the same sentence ID across sources.
    """
    grouped = []
    group_cols: typing.Union[str, typing.List[str]] = sent_id_col
    if source_col is not None and source_col in df.columns:
        group_cols = [source_col, sent_id_col]

    for _, group in df.groupby(group_cols, sort=False):
        tokens = list(group[text_col].astype(str))
        tags = list(group[label_col].astype(str))
        grouped.append((tokens, tags))
    return grouped


# ---------------------------------------------------------------------------
# Shared placeholder-label helpers
# ---------------------------------------------------------------------------


def _parse_placeholder_labels(raw_labels: str) -> typing.Set[str]:
    """Parse comma-separated placeholder labels from a string.

    Converts the ``--placeholder-labels`` CLI argument (a comma-separated
    string such as ``"-,NONE"``) into a ``set`` of label strings suitable for
    passing to preprocessing and evaluation/prediction functions.

    Args:
        raw_labels: Comma-separated label strings, e.g. ``"-,NONE"``.

    Returns:
        Parsed non-empty label set.  Falls back to ``{"-", "NONE"}`` when the
        input produces no non-empty parts after splitting.
    """
    parsed = {part.strip() for part in raw_labels.split(",") if part.strip()}
    if not parsed:
        return {"-", "NONE"}
    return parsed


def _normalise_placeholder_labels(
    placeholder_labels: typing.Optional[typing.Set[str]],
) -> typing.Set[str]:
    """Normalise placeholder labels to a consistent upper-cased set.

    Args:
        placeholder_labels: Optional user-provided placeholder label set.
            Defaults to ``{"-", "NONE"}`` when ``None``.

    Returns:
        Upper-cased, stripped label set with empty strings removed.
    """
    if placeholder_labels is None:
        placeholder_labels = {"-", "NONE"}
    return {
        str(label).strip().upper() for label in placeholder_labels if str(label).strip()
    }


def _select_placeholder_output_label(
    normalized_placeholder_labels: typing.Set[str],
) -> str:
    """Pick a deterministic placeholder label for output rows.

    Args:
        normalized_placeholder_labels: Normalised placeholder label set.

    Returns:
        The lexicographically first placeholder label, or ``"-"`` when the
        set is empty.
    """
    if not normalized_placeholder_labels:
        return "-"
    return sorted(normalized_placeholder_labels)[0]


def _is_placeholder_label(
    label: typing.Optional[str],
    normalized_placeholder_labels: typing.Set[str],
) -> bool:
    """Check if a label value should be treated as a placeholder.

    Args:
        label: Candidate label value.
        normalized_placeholder_labels: Set of normalised (upper-cased)
            placeholder labels to match against.

    Returns:
        ``True`` when the label is ``None``, empty, or matches a normalised
        placeholder label; ``False`` otherwise.
    """
    if label is None:
        return True
    normalized = str(label).strip()
    return normalized == "" or normalized.upper() in normalized_placeholder_labels


def _tokenize_word_for_output(tokenizer: typing.Any, word: str) -> typing.List[str]:
    """Tokenise a word and return the subtoken list, with fallback for empty results.

    When the tokenizer produces no tokens for a word (e.g. purely punctuation
    on some tokenizers), falls back to the tokenizer's ``unk_token`` or the
    literal string ``"[UNK]"`` so callers never receive an empty list.

    Args:
        tokenizer: Hugging Face tokenizer instance.
        word: Input word to tokenise.

    Returns:
        Non-empty list of subtoken strings.
    """
    word_tokens = tokenizer.tokenize(str(word))
    if word_tokens:
        return word_tokens
    unk_token = getattr(tokenizer, "unk_token", None)
    if unk_token is None:
        return ["[UNK]"]
    fallback_tokens = tokenizer.tokenize(unk_token)
    if fallback_tokens:
        return fallback_tokens
    return [unk_token]


def _token_count_for_word(tokenizer: typing.Any, word: str) -> int:
    """Compute the number of subtokens a tokenizer produces for a word.

    Uses the same unk-token fallback logic as :func:`_tokenize_word_for_output`
    so the count is always at least 1.  This is used to advance word-level
    token pointers in evaluation and prediction loops.

    Args:
        tokenizer: Hugging Face tokenizer instance.
        word: Input word.

    Returns:
        Number of subtokens (at least 1).
    """
    return len(_tokenize_word_for_output(tokenizer, word))


# ---------------------------------------------------------------------------
# Data loading helpers (shared across eval, predict and training scripts)
# ---------------------------------------------------------------------------


def load_df(path: Path, file_format: str) -> pd.DataFrame:
    """Load a dataset file from disk into a pandas DataFrame.

    Supports CSV and Parquet formats.  When ``file_format`` is ``"auto"``, the
    format is inferred from the file suffix.  This function is shared across
    evaluation, prediction, and training data-loading workflows.

    Args:
        path: Path to the input file.
        file_format: One of ``"auto"``, ``"csv"``, or ``"parquet"``.

    Returns:
        Loaded dataset as a DataFrame.

    Raises:
        FileNotFoundError: When the file does not exist.
        ValueError: When the format cannot be inferred or is unsupported.
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


def load_test_df(path: Path, file_format: str) -> pd.DataFrame:
    """Backward-compatible alias for :func:`load_df`.

    Args:
        path: Path to the input file.
        file_format: One of ``"auto"``, ``"csv"``, or ``"parquet"``.

    Returns:
        Loaded dataset as a DataFrame.
    """
    return load_df(path, file_format)


def prepare_shared_inputs(
    test_df: pd.DataFrame,
    sent_id_col: str,
    text_col: str,
    label_col: str,
    source_col: typing.Optional[str] = None,
) -> typing.List[typing.Tuple[typing.List[str], typing.List[str]]]:
    """Convert a token-level DataFrame into sentence-level ``(words, labels)`` tuples.

    Validates that the required columns exist, then groups token rows by
    sentence ID and returns the list of tuples used as input to preprocessing,
    evaluation, and prediction loops.

    Args:
        test_df: Token-level DataFrame with one row per word.
        sent_id_col: Name of the sentence identifier column.
        text_col: Name of the word/token text column.
        label_col: Name of the label column.

    Returns:
        List of ``(words, labels)`` tuples, one per sentence.

    Raises:
        ValueError: When any required column is absent from ``test_df``.
    """

    def _normalize_sentence_ids_by_source(
        df: pd.DataFrame,
        sentence_id_col: str,
        src_col: str,
    ) -> pd.DataFrame:
        """Map ``(source, sentence_id)`` pairs to a global consecutive sentence_id.

        The mapping preserves first-seen order of unique pairs in ``df``.
        """
        unique_pairs = (
            df[[src_col, sentence_id_col]].drop_duplicates().reset_index(drop=True)
        )
        unique_pairs["_normalized_sentence_id"] = unique_pairs.index.astype(int)

        merged = df.merge(
            unique_pairs,
            on=[src_col, sentence_id_col],
            how="left",
            sort=False,
        )
        merged[sentence_id_col] = merged["_normalized_sentence_id"].astype(int)
        return merged.drop(columns=["_normalized_sentence_id"])

    missing_cols = [
        col for col in (sent_id_col, text_col, label_col) if col not in test_df.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing required test-set columns: {missing_cols}")

    processed_source_col = source_col

    source_df = test_df
    if processed_source_col is not None:
        if processed_source_col not in test_df.columns:
            raise ValueError(
                f"Source column '{processed_source_col}' is missing from test set."
            )
        source_df = _normalize_sentence_ids_by_source(
            test_df,
            sentence_id_col=sent_id_col,
            src_col=processed_source_col,
        )

    processed_df: pd.DataFrame = pd.DataFrame(
        {
            "sentence_id": source_df[sent_id_col],
            "words": source_df[text_col],
            "labels": source_df[label_col],
        }
    )
    return _group_sentences_from_df(processed_df)


def prepare_token_classification_data(
    tokenizer,
    sentences: typing.List[typing.Tuple[typing.List[str], typing.List[str]]],
    label2id: typing.Dict[str, int],
    max_length: typing.Optional[int] = None,
    ignore_placeholders: bool = False,
    pad_token_label_id: int = -100,
    placeholder_labels: typing.Optional[typing.Set[str]] = None,
) -> typing.Tuple[typing.List[dict], typing.List[typing.List[int]]]:
    """
    Tokenize and align labels using a SimpleTransformers-compatible pipeline.

    This mirrors the older `simpletransformers.ner.convert_example_to_feature`
    behaviour used by the historic checkpoints in this repository:
    tokenise each word independently, assign the gold label only to the first
    produced subtoken, add special tokens manually, and then truncate/pad to
    `max_length`.

    Returns list of per-example encodings (dicts) and label id lists. Uses
    `pad_token_label_id` for special tokens, padded positions, and ignored
    subtokens so evaluation/training behaves like the original pipeline.

    Args:
        tokenizer: Hugging Face tokenizer instance.
        sentences: List of sentence tuples ``(words, labels)``.
        label2id: Mapping from label string to integer id.
        max_length: Optional max sequence length.
        ignore_placeholders: If True, placeholder labels are encoded as
            ``pad_token_label_id`` instead of label ids.
        pad_token_label_id: Special ignore id used by token classification.
        placeholder_labels: Optional set of placeholder label strings.
            Defaults to ``{"-", "NONE"}``. Matching is case-insensitive.
    """
    # Validate and clean U+2581 markers before tokenization
    # The tokenizer uses U+2581 (▁) to mark word beginnings.
    cleaned_sentences = []
    for words, labels in sentences:
        # Remove any existing U+2581 from input to avoid conflicts.
        cleaned_words = [word.replace("\u2581", " ").strip() for word in words]
        cleaned_sentences.append((cleaned_words, labels))

    encodings = []
    all_label_ids: typing.List[typing.List[int]] = []

    normalized_placeholder_labels = _normalise_placeholder_labels(placeholder_labels)

    tokenizer_signature = " ".join(
        [
            tokenizer.__class__.__name__.lower(),
            str(getattr(tokenizer, "name_or_path", "")).lower(),
        ]
    )
    sep_token_extra = any(
        model_name in tokenizer_signature
        for model_name in ("roberta", "camembert", "xlmroberta", "longformer", "mpnet")
    )

    cls_token = getattr(tokenizer, "cls_token", None)
    sep_token = getattr(tokenizer, "sep_token", None)
    unk_token = getattr(tokenizer, "unk_token", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0

    if cls_token is None or sep_token is None:
        raise ValueError(
            "Tokenizer must define cls_token and sep_token for token classification preprocessing."
        )

    def _tokenize_word(word: str) -> typing.List[str]:
        word_tokens = tokenizer.tokenize(str(word))
        if word_tokens:
            return word_tokens

        if unk_token is None:
            raise ValueError(
                "Tokenizer produced no tokens for a word and has no unk_token fallback."
            )

        fallback_tokens = tokenizer.tokenize(unk_token)
        return fallback_tokens or [unk_token]

    for words, labels in cleaned_sentences:
        if len(words) != len(labels):
            raise ValueError(
                "Each sentence must contain the same number of words and labels."
            )

        tokens: typing.List[str] = []
        label_ids: typing.List[int] = []

        for word, label in zip(words, labels):
            word_tokens = _tokenize_word(str(word))
            tokens.extend(word_tokens)

            if ignore_placeholders and _is_placeholder_label(
                label, normalized_placeholder_labels
            ):
                first_label_id = pad_token_label_id
            else:
                first_label_id = label2id.get(str(label), pad_token_label_id)

            label_ids.extend(
                [first_label_id] + [pad_token_label_id] * (len(word_tokens) - 1)
            )

        if max_length is not None:
            special_tokens_count = 3 if sep_token_extra else 2
            max_token_count = max_length - special_tokens_count
            if len(tokens) > max_token_count:
                tokens = tokens[:max_token_count]
                label_ids = label_ids[:max_token_count]

        tokens = tokens + [sep_token]
        label_ids = label_ids + [pad_token_label_id]
        if sep_token_extra:
            tokens = tokens + [sep_token]
            label_ids = label_ids + [pad_token_label_id]

        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        token_type_ids = None
        if "token_type_ids" in getattr(tokenizer, "model_input_names", []):
            token_type_ids = [0] * len(input_ids)

        if max_length is not None:
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
                if token_type_ids is not None:
                    token_type_ids = token_type_ids + [0] * pad_len
                label_ids = label_ids + [pad_token_label_id] * pad_len

        enc_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            enc_dict["token_type_ids"] = token_type_ids

        encodings.append(enc_dict)
        all_label_ids.append(label_ids)

    return encodings, all_label_ids


def compare_label_lists(labels_json_path: str, model_dir: str):
    with open(labels_json_path, "r", encoding="utf-8") as f:
        user_labels = list(json.load(f))  # list of label strings

    cfg = AutoConfig.from_pretrained(model_dir)
    cfg_id2label = getattr(cfg, "id2label", None)
    cfg_label2id = getattr(cfg, "label2id", None)

    # Build model label list in index order if possible
    model_labels = []
    model_label2id = {}
    model_id2label = {}
    if cfg_id2label:
        model_id2label = {int(k): str(v) for k, v in cfg_id2label.items()}
        model_labels = [model_id2label[i] for i in sorted(model_id2label.keys())]
        model_label2id = {v: k for k, v in model_id2label.items()}
    elif cfg_label2id:
        model_label2id = {str(k): int(v) for k, v in cfg_label2id.items()}
        # invert to ordered list if contiguous
        max_id = max(model_label2id.values())
        model_labels = [None] * (max_id + 1)
        for lab, idx in model_label2id.items():
            model_labels[idx] = lab
        model_id2label = {
            i: lab for i, lab in enumerate(model_labels) if lab is not None
        }
    else:
        print("Model config has no id2label/label2id mapping.")
        model_labels = []
        model_label2id = {}
        model_id2label = {}

    set_user = set(user_labels)
    set_model = set([l for l in model_labels if l is not None])

    only_in_user = sorted(set_user - set_model)
    only_in_model = sorted(set_model - set_user)
    in_both = sorted(set_user & set_model)

    mismatched_ids = []
    for lab in in_both:
        uid = user_labels.index(lab)
        mid = model_label2id.get(lab)
        if mid is None or mid != uid:
            mismatched_ids.append((lab, uid, mid))

    print(f"user labels: {len(user_labels)}, model labels: {len(model_labels)}")
    print("Only in user (sample):", only_in_user[:30])
    print("Only in model (sample):", only_in_model[:30])
    if mismatched_ids:
        print("Labels with different ids (label, user_index, model_index):")
        for t in mismatched_ids[:50]:
            print(t)
    else:
        print("No differing id assignments for common labels.")
    return {
        "only_in_user": only_in_user,
        "only_in_model": only_in_model,
        "mismatched_ids": mismatched_ids,
        "user_len": len(user_labels),
        "model_len": len(model_labels),
    }
