import os
import gc
import pandas as pd
import numpy as np
import math
import typing
import logging
import warnings
import random
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    RobertaForTokenClassification,
    get_linear_schedule_with_warmup,
    DataCollatorForTokenClassification,
    RobertaTokenizer,
)
import warnings
import evaluate

# Use the Poseval metric (Hugging Face evaluate-metric/poseval)
try:
    poseval = evaluate.load("evaluate-metric/poseval", module_type="metric")
except Exception:
    raise ImportError(
        "Poseval metric is required for evaluation but could not be loaded. "
        "Please install it via `pip install evaluate` or check your environment."
    )
from tqdm.auto import tqdm


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
):
    """
    Convert DataFrame with token-level rows into list-of-tokens and list-of-labels per sentence.
    """
    grouped = []
    for sid, group in df.groupby(sent_id_col):
        tokens = list(group[text_col].astype(str))
        tags = list(group[label_col].astype(str))
        grouped.append((tokens, tags))
    return grouped


def prepare_token_classification_data(
    tokenizer,
    sentences: typing.List[typing.Tuple[typing.List[str], typing.List[str]]],
    label2id: typing.Dict[str, int],
    max_length: typing.Optional[int] = None,
    ignore_placeholders: bool = False,
    pad_token_label_id: int = -100,
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
    """
    encodings = []
    all_label_ids: typing.List[typing.List[int]] = []

    def _is_placeholder_label(label: typing.Optional[str]) -> bool:
        if label is None:
            return True
        normalized = str(label).strip()
        return normalized == "" or normalized.upper() in {"NONE", "-"}

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

    for words, labels in sentences:
        if len(words) != len(labels):
            raise ValueError(
                "Each sentence must contain the same number of words and labels."
            )

        tokens: typing.List[str] = []
        label_ids: typing.List[int] = []

        for word, label in zip(words, labels):
            word_tokens = _tokenize_word(str(word))
            tokens.extend(word_tokens)

            if ignore_placeholders and _is_placeholder_label(label):
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


def train_token_classification(
    model: torch.nn.Module,
    tokenizer,
    train_df,
    label_list: typing.List[str],
    output_dir: str,
    *,
    eval_df: typing.Optional[typing.Any] = None,
    num_train_epochs: int = 10,
    train_batch_size: int = 8,
    learning_rate: float = 5e-5,
    max_grad_norm: float = 1.0,
    evaluate_during_training: bool = False,
    use_early_stopping: bool = True,
    patience_n: int = 2,
    early_stopping_method: str = "f1",
    early_stopping_threshold: float = 0.0,
    eval_every_n_epochs: int = 1,
    best_model_dir: typing.Optional[str] = None,
    save_model_every_epoch: bool = False,
    save_steps: int = -1,
    device: typing.Optional[torch.device] = None,
    silent: bool = False,
    gradient_accumulation_steps: int = 1,
    max_length: typing.Optional[int] = None,
    ignore_placeholders: bool = False,
    seed: int = 42,
    dry_run: bool = False,
) -> dict:
    """
    Train a transformers token-classification model (no simpletransformers dependency).

    Args:
      model: transformers AutoModelForTokenClassification (already initialised).
      tokenizer: the matching tokenizer (fast tokenizer recommended).
      train_df: pandas DataFrame with token-level rows and columns `sentence_id`, `words`, `labels`.
      label_list: list of label strings (ordered).
      output_dir: path to save model and tokenizer.
      eval_df: optional DataFrame for evaluation (same format as train_df).
      evaluate_during_training: if True, evaluate after each epoch (required for early stopping).
      use_early_stopping: requires evaluation; uses best validation f1 to stop.
      patience_n: number of epochs to wait for improvement before stopping (if use_early_stopping).
      early_stopping_method: "f1" or "loss".
      early_stopping_threshold: minimum improvement or maximum loss for early stopping
      best_model_dir: where to save best model (if provided).
      save_model_every_epoch: save a checkpoint every epoch (if True).
      save_steps: not used here because checkpointing is epoch-based; keep for API compatibility.
      device: torch.device; auto-detected if None.
      silent: if True reduce tqdm output.
      gradient_accumulation_steps: number of steps to accumulate gradients before updating model.
      max_length: optional max tokenization length (passed to tokenizer).
      ignore_placeholders: if True, ignores placeholder labels (e.g. '-', 'NONE', '') in label mapping validation and optionally in training (only_target_token).
      seed: random seed for reproducibility.
      dry_run: if True, runs through the motions without saving any models or checkpoints (useful for testing).

    Returns:
      dict with training statistics and last evaluation report.
    """

    # ---------- Reproducibility ----------
    # Seed Python, NumPy and PyTorch RNGs so runs are repeatable when possible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If multiple GPUs are available set their seed too.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---------- Device placement ----------
    # If `device` not provided, auto-select CUDA when available otherwise CPU.
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---------- Label mapping ----------
    # Accept either a list of labels (ordered) or a dict mapping label->id.
    # Ensure keys/values are strings/ints and canonicalise both mappings.
    if isinstance(label_list, dict):
        label2id = {str(k): int(v) for k, v in label_list.items()}
        id2label = {int(v): str(k) for k, v in label_list.items()}
    else:
        label2id = {str(lbl): i for i, lbl in enumerate(label_list)}
        id2label = {i: lbl for lbl, i in label2id.items()}

    # ---------- Validate dataset labels against model mapping ----------
    def _collect_unique_labels(df):
        # Collect unique label strings from a token-level DataFrame column `labels`.
        if df is None:
            return set()
        return set(df["labels"].astype(str).unique())

    train_labels_in_data = _collect_unique_labels(train_df)
    eval_labels_in_data = _collect_unique_labels(eval_df)

    # Some corpora use placeholder labels for non-target tokens (e.g. '-', 'NONE', '').
    # When `ignore_placeholders` is enabled, exclude these from validation so they
    # don't trigger an error when the model's label mapping deliberately omits them.
    def _is_non_target_label(l: str) -> bool:
        if l is None:
            return True
        nl = str(l).strip()
        return nl == "" or nl.upper() in {"NONE", "-"}

    if ignore_placeholders:
        train_labels_in_data = {
            l for l in train_labels_in_data if not _is_non_target_label(l)
        }
        eval_labels_in_data = {
            l for l in eval_labels_in_data if not _is_non_target_label(l)
        }

    # Determine labels that appear in data but are not present in the model mapping.
    missing_in_model = sorted(
        list((train_labels_in_data | eval_labels_in_data) - set(label2id.keys()))
    )
    if missing_in_model:
        # Fail early with a helpful message so user can supply a correct mapping.
        raise ValueError(
            f"The following labels are present in the dataset but missing in model label mapping: {missing_in_model[:50]}."
            " Provide a mapping that includes these labels or update the model config before training."
        )

    # ---------- Prepare train examples & dataloaders ----------
    # Group token-level DataFrame rows into sentence tuples (words, labels).
    train_sentences = _group_sentences_from_df(train_df)
    if len(train_sentences) == 0:
        raise ValueError("No training sentences found in train_df")

    # Tokenize and align labels to tokens (produces encodings + label ids per example).
    encodings_train, labels_train = prepare_token_classification_data(
        tokenizer,
        train_sentences,
        label2id,
        max_length=max_length,
        ignore_placeholders=ignore_placeholders,
    )

    # Data collator handles padding of variable-length sequences into batches and
    # prepares the `labels` tensor expected by the model (works with label ids -100).
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Wrap into a Dataset that converts lists to tensors on-the-fly in __getitem__.
    train_dataset = TokenClassificationDataset(encodings_train, labels_train)

    # RandomSampler for shuffled training; DataLoader with collate_fn for batching.
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_batch_size,
        collate_fn=data_collator,  # pads and stacks examples into tensors
        num_workers=0,  # set >0 to parallelise loading if desired
        pin_memory=True,  # can improve GPU transfer performance
    )

    # Optional evaluation dataset (created similarly but no shuffling).
    if evaluate_during_training and eval_df is not None:
        eval_sentences = _group_sentences_from_df(eval_df)
        encodings_eval, labels_eval = prepare_token_classification_data(
            tokenizer, eval_sentences, label2id, max_length=max_length
        )
        eval_dataset = TokenClassificationDataset(encodings_eval, labels_eval)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=train_batch_size,
            collate_fn=data_collator,
            num_workers=0,
        )
    else:
        eval_loader = None

    # ---------- Optimiser & learning-rate scheduler ----------
    # Apply weight decay to most parameters but exclude
    # bias and LayerNorm weights (they typically should not be decayed).
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    # AdamW optimiser (decoupled weight decay) with chosen learning rate.
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Total training steps (accounts for gradient accumulation): number of optimizer
    # updates = (batches per epoch // gradient_accumulation_steps) * epochs
    t_total = len(train_loader) // gradient_accumulation_steps * num_train_epochs

    # Linear schedule with warmup (0 warmup steps here) for learning-rate decay.
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=t_total
    )

    # helper: compute metrics on a dataloader
    # poseval (via evaluate) for metric computation

    def _compute_metrics(dataloader):
        """
        Compute loss and token-level predictions for all examples in `dataloader`.

        This helper runs the model in evaluation mode and accumulates:
        - `total_loss`: average of model-reported batch losses
        - `preds_all`: list of predicted label sequences (strings) per example
        - `labels_all`: list of gold label sequences (strings) per example
        - token-level accuracy computed by comparing predicted and gold ids

        Notes:
        - The dataloader is expected to yield dicts with tensors including
          `input_ids`, `attention_mask`, and `labels` (labels use -100 for ignored
          positions). The `labels` tensor is moved to `device` along with inputs.
        - `id2label` from the outer scope is used to convert integer ids to
          label strings for metric computation (poseval expects string labels).

        Args:
            dataloader: PyTorch DataLoader yielding batches compatible with the model.

        Returns:
            dict with keys: loss, accuracy, precision, recall, f1, report, poseval.
        """

        model.eval()
        preds_all = []
        labels_all = []
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        # Disable gradient computation for evaluation for speed/memory
        with torch.no_grad():
            for batch in dataloader:
                # Move batch tensors to device (CPU/GPU)
                b = {k: v.to(device) for k, v in batch.items()}

                # Forward pass: model returns a loss and logits when `labels` included
                outputs = model(**b)
                total_loss += outputs.loss.item()

                # logits: (batch, seq_len, num_labels) -> convert to CPU numpy
                logits = outputs.logits.detach().cpu().numpy()

                # label_ids: padded label ids tensor on device -> CPU numpy
                label_ids = b["labels"].detach().cpu().numpy()

                # predicted label ids per token (argmax over last dim)
                pred_ids = logits.argmax(axis=-1)

                # Iterate examples in the batch and convert to string labels,
                # skipping positions marked with -100 (ignored in loss/metrics).
                for p_row, l_row in zip(pred_ids, label_ids):
                    preds = []
                    labs = []
                    for p, l in zip(p_row, l_row):
                        # Skip ignored positions (-100), e.g. subtokens and special tokens
                        if l == -100:
                            continue
                        # Map integer ids to label strings using `id2label`
                        preds.append(id2label[int(p)])
                        labs.append(id2label[int(l)])

                        # Update simple accuracy counters (token-level)
                        total_tokens += 1
                        if int(p) == int(l):
                            total_correct += 1

                    # Collect sequence-wise predictions/references for external metrics
                    preds_all.append(preds)
                    labels_all.append(labs)

        # Average loss across batches (safe divide)
        avg_loss = total_loss / (len(dataloader) if len(dataloader) else 1)
        accuracy = total_correct / total_tokens if total_tokens else 0.0

        # Use poseval to compute precision/recall/F1 on sequence labels
        poseval_result = poseval.compute(
            predictions=preds_all, references=labels_all, zero_division=0
        )

        # Extract common metrics
        accuracy = poseval_result.get("accuracy", accuracy)
        prec = poseval_result.get("weighted avg", {}).get("precision", 0.0)
        rec = poseval_result.get("weighted avg", {}).get("recall", 0.0)
        f1 = poseval_result.get("weighted avg", {}).get("f1-score", 0.0)
        report = poseval_result

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "weighted avg precision": prec,
            "weighted avg recall": rec,
            "weighted avg f1-score": f1,
            "report": report,
        }

    # Training loop
    # best metric depends on early_stopping_method
    if early_stopping_method == "loss":
        best_val_metric = float("inf")
    else:
        best_val_metric = -float("inf")
    best_state = None
    patience = patience_n if use_early_stopping else None
    patience_counter = 0

    global_step = 0
    training_stats = {"train_loss": [], "train_metrics": [], "eval_metrics": []}

    for epoch in range(int(num_train_epochs)):
        # Set model to training mode (enables dropout, etc.)
        model.train()
        epoch_loss = 0.0

        # tqdm progress bar over batches
        progress = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_train_epochs}", disable=silent
        )

        # Iterate over batches. `step` counts batches, not optimizer updates.
        for step, batch in enumerate(progress):
            # Move batch tensors to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass returns a loss when `labels` are provided in the batch
            outputs = model(**batch)
            loss = outputs.loss

            # Support gradient accumulation: scale loss before backward so that
            # gradients represent the sum over `gradient_accumulation_steps` mini-batches.
            loss = loss / gradient_accumulation_steps
            loss.backward()

            # Keep epoch-level loss (note: averaged/scaled loss is accumulated)
            epoch_loss += loss.item()

            # Update running train loss shown in the progress bar for visibility
            if not silent:
                try:
                    current_avg = epoch_loss / (step + 1)
                    progress.set_postfix({"train_loss": f"{current_avg:.4f}"})
                except Exception:
                    pass

            # Perform optimizer step only after `gradient_accumulation_steps` batches
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # Update parameters and learning rate schedule
                optimizer.step()
                scheduler.step()

                # Reset gradients for next accumulation window
                optimizer.zero_grad()
                global_step += 1

            # Optional checkpointing by step: save model+tokenizer at given steps
            if save_steps > 0 and global_step % save_steps == 0:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-step-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                if not dry_run:
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

        # End of epoch: compute average loss and store training stats
        avg_epoch_loss = epoch_loss / (len(train_loader) if len(train_loader) else 1)
        training_stats["train_loss"].append(avg_epoch_loss)

        # Show epoch-level train loss on the progress bar
        if not silent:
            try:
                progress.set_postfix({"train_loss": f"{avg_epoch_loss:.4f}"})
            except Exception:
                pass

        # ---------- Evaluation ----------
        # Compute metrics on the training set (useful for monitoring overfitting)
        eval_report = None
        train_metrics = _compute_metrics(train_loader)
        training_stats["train_metrics"].append(train_metrics)

        # Optionally compute metrics on the validation set for early stopping.
        # The `eval_every_n_epochs` parameter controls how often (in epochs)
        # evaluation is performed to allow cheaper monitoring when desired.
        if (
            evaluate_during_training
            and eval_loader is not None
            and (eval_every_n_epochs <= 1 or (epoch + 1) % eval_every_n_epochs == 0)
        ):
            # Run the model over the eval set and obtain loss/precision/recall/f1
            eval_metrics = _compute_metrics(eval_loader)
            training_stats["eval_metrics"].append(eval_metrics)

            # Shorthand report containing the most-used fields
            eval_report = {
                "f1": eval_metrics["weighted avg f1-score"],
                "loss": eval_metrics["loss"],
                "report": eval_metrics["report"],
            }

            # ---------- Early stopping decision ----------
            # Choose which metric to monitor: either 'loss' (lower is better)
            # or another metric such as 'f1' (higher is better). If the chosen
            # metric is not present in eval_metrics, fall back to F1.
            metric_name = early_stopping_method
            if metric_name not in eval_metrics:
                metric_value = eval_metrics.get("f1", 0.0)
            else:
                metric_value = eval_metrics[metric_name]

            # If no threshold specified, use the traditional improvement-based
            # early stopping: require `metric_value` to improve over `best_val_metric`.
            if early_stopping_threshold is None:
                improved = (
                    metric_value < best_val_metric
                    if early_stopping_method == "loss"
                    else metric_value > best_val_metric
                )

                if improved:
                    # Found a new best; reset patience and optionally save checkpoint
                    best_val_metric = metric_value
                    patience_counter = 0
                    if best_model_dir and not dry_run:
                        os.makedirs(best_model_dir, exist_ok=True)
                        model.save_pretrained(best_model_dir)
                        tokenizer.save_pretrained(best_model_dir)
                    # Keep best model weights in-memory to allow restoring later
                    best_state = deepcopy(model.state_dict())
                else:
                    # No improvement: increase patience and stop when exceeded
                    if use_early_stopping:
                        patience_counter += 1
                        if patience is not None and patience_counter >= patience:
                            # Restore the best model state if available
                            if best_state is not None:
                                model.load_state_dict(best_state)
                            break
            else:
                # Threshold-based early stopping mode: continue training while the
                # monitored metric remains above (or loss below) the threshold.
                if early_stopping_method == "loss":
                    threshold_ok = metric_value <= early_stopping_threshold
                else:
                    threshold_ok = metric_value >= early_stopping_threshold

                if threshold_ok:
                    # Metric passes the threshold: reset patience. Also update
                    # best_state if this epoch improved the best seen metric.
                    patience_counter = 0
                    improved = (
                        metric_value < best_val_metric
                        if early_stopping_method == "loss"
                        else metric_value > best_val_metric
                    )
                    if improved:
                        best_val_metric = metric_value
                        if best_model_dir and not dry_run:
                            os.makedirs(best_model_dir, exist_ok=True)
                            model.save_pretrained(best_model_dir)
                            tokenizer.save_pretrained(best_model_dir)
                        best_state = deepcopy(model.state_dict())
                else:
                    # Metric failed the threshold check: increase patience and
                    # stop if the allowed patience is exhausted.
                    if use_early_stopping:
                        patience_counter += 1
                        if patience is not None and patience_counter >= patience:
                            if best_state is not None:
                                model.load_state_dict(best_state)
                            break

            # Update progress bar with key metrics for quick visual feedback
            if not silent:
                try:
                    # Print eval metrics to console for visibility
                    print(
                        f"Epoch {epoch + 1} eval metrics: "
                        f"loss={eval_metrics['loss']:.4f}, "
                        f"f1={eval_metrics['f1']:.4f}, "
                        f"accuracy={eval_metrics['accuracy']:.4f}"
                    )
                    postfix = {
                        "train_loss": f"{avg_epoch_loss:.4f}",
                        "train_f1": f"{train_metrics['f1']:.4f}",
                        "train_acc": f"{train_metrics['accuracy']:.4f}",
                        "eval_f1": f"{eval_metrics['f1']:.4f}",
                        "eval_acc": f"{eval_metrics['accuracy']:.4f}",
                        "eval_loss": f"{eval_metrics['loss']:.4f}",
                    }
                    progress.set_postfix(postfix)
                except Exception:
                    pass

            # Ensure model returned to training mode after a purely-readonly eval
            # so subsequent training (or checkpointing) behaves normally.
            try:
                model.train()
            except Exception:
                pass

        # ---------- Epoch checkpointing ----------
        # Optionally save a checkpoint after every epoch
        if save_model_every_epoch:
            ckpt_dir = os.path.join(output_dir, f"epoch-{epoch + 1}")
            if not dry_run:
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

    # Final save (overwrite or save into output_dir)
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    # free memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"training_stats": training_stats, "last_eval": eval_report}


def evaluate_model(
    model: torch.nn.Module,
    tokenizer,
    device: typing.Optional[torch.device],
    eval_df_or_path: typing.Union[str, "pd.DataFrame"],
    batch_size: int = 8,
    max_length: typing.Optional[int] = None,
):
    """
    Evaluate a token-classification model on a dataset provided as a pandas
    DataFrame (token-level rows) or a path to a CSV file with token-level rows.

    Args:
        model: transformers AutoModelForTokenClassification
        tokenizer: matching tokenizer (fast tokenizer recommended)
        device: torch.device or None (auto-detected)
        eval_df_or_path: pandas DataFrame or path to CSV containing columns
            `sentence_id`, `words`, `labels` (token-level rows)
        batch_size: batch size for evaluation
        max_length: optional max tokenisation length (passed to tokenizer)

    Returns:
        dict with keys: loss, accuracy, precision, recall, f1, report, poseval
    """

    import pandas as pd
    import os

    # Normalize device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load eval DataFrame if a path was provided
    if isinstance(eval_df_or_path, str):
        if not os.path.exists(eval_df_or_path):
            raise FileNotFoundError(f"Eval CSV not found: {eval_df_or_path}")
        eval_df = pd.read_csv(eval_df_or_path)
    else:
        eval_df = eval_df_or_path

    # Group into sentence-level pairs (words, labels)
    eval_sentences = _group_sentences_from_df(eval_df)

    # Extract label mapping from model config
    cfg_label2id = getattr(model.config, "label2id", None)
    cfg_id2label = getattr(model.config, "id2label", None)

    label2id: typing.Dict[str, int] = {}
    id2label: typing.Dict[int, str] = {}

    if isinstance(cfg_label2id, dict) and len(cfg_label2id) > 0:
        # Ensure ints for ids
        label2id = {str(k): int(v) for k, v in cfg_label2id.items()}
        id2label = {int(v): str(k) for k, v in cfg_label2id.items()}
    elif isinstance(cfg_id2label, dict) and len(cfg_id2label) > 0:
        id2label = {int(k): str(v) for k, v in cfg_id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
    else:
        raise ValueError(
            "Model config does not contain label mappings (id2label/label2id)"
        )

    # Prepare tokenised encodings and aligned labels
    encodings_eval, labels_eval = prepare_token_classification_data(
        tokenizer, eval_sentences, label2id, max_length=max_length
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    eval_dataset = TokenClassificationDataset(encodings_eval, labels_eval)
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=data_collator, num_workers=0
    )

    # Run evaluation loop (same logic as training _compute_metrics)
    model.eval()
    preds_all = []
    labels_all = []
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for batch in eval_loader:
            b = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**b)
            total_loss += outputs.loss.item()
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = b["labels"].detach().cpu().numpy()
            pred_ids = logits.argmax(axis=-1)

            for p_row, l_row in zip(pred_ids, label_ids):
                preds = []
                labs = []
                for p, l in zip(p_row, l_row):
                    if l == -100:
                        continue
                    preds.append(id2label[int(p)])
                    labs.append(id2label[int(l)])
                    total_tokens += 1
                    if int(p) == int(l):
                        total_correct += 1
                preds_all.append(preds)
                labels_all.append(labs)

    avg_loss = total_loss / (len(eval_loader) if len(eval_loader) else 1)
    accuracy = total_correct / total_tokens if total_tokens else 0.0

    poseval_result = None
    if poseval is not None and preds_all:
        try:
            poseval_result = poseval.compute(
                predictions=preds_all, references=labels_all
            )
        except Exception:
            poseval_result = None

    if poseval_result:
        prec = poseval_result.get("precision", poseval_result.get("prec", 0.0))
        rec = poseval_result.get("recall", poseval_result.get("rec", 0.0))
        f1 = poseval_result.get("f1", poseval_result.get("fscore", 0.0))
        report = poseval_result
    else:
        prec = 0.0
        rec = 0.0
        f1 = 0.0
        report = {}

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "report": report,
        "poseval": poseval_result,
    }
