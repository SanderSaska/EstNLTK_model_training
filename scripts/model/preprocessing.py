import os
import gc
import math
import typing
import logging
import warnings
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
    RobertaTokenizerFast,
)
from seqeval.metrics import classification_report, f1_score
from tqdm.auto import tqdm


def initialize_model(
    model_name: str,
    unique_labels: typing.Optional[typing.List[str]] = None,
    no_progress_bars: bool = False,
    use_Roberta_tokenizer: bool = False,
    cleanup: bool = True,
) -> typing.Dict[str, typing.Any]:
    """
    Initialise a token-classification model (for NER) using Hugging Face transformers.

    Args:
        model_name: Pretrained model name or local path (e.g. 'camembert-base').
        unique_labels: All possible label strings the model will predict.
        no_progress_bars: If True, reduces HF logging verbosity.
        use_Roberta_tokenizer: If True, uses RobertaTokenizerFast instead of AutoTokenizer (for better handling of RoBERTa-like tokenization).
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

    # Logging / warnings configuration
    # logging.getLogger("transformers").setLevel(logging.ERROR)
    # if no_progress_bars:
    #     from transformers.utils import logging as hf_logging

    #     hf_logging.set_verbosity_error()
    # warnings.filterwarnings("ignore", category=UserWarning)

    # Build label mappings expected by transformers config if provided
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
        config = AutoConfig.from_pretrained(model_name, num_labels=len(unique_labels))

    cfg_num_labels = getattr(config, "num_labels", None)
    cfg_id2label = getattr(config, "id2label", None)
    cfg_label2id = getattr(config, "label2id", None)

    # If the checkpoint already contains a label mapping, prefer it to preserve the head weights.
    # Only fall back to provided unique_labels when the checkpoint has no mappings.
    if cfg_id2label and isinstance(cfg_id2label, dict):
        # Use model mapping
        try:
            id2label = {int(k): str(v) for k, v in cfg_id2label.items()}
            label2id = {str(v): int(k) for k, v in cfg_id2label.items()}
        except Exception:
            id2label = {i: label for i, label in enumerate(unique_labels or [])}
            label2id = {label: i for i, label in enumerate(unique_labels or [])}
        config.num_labels = len(id2label)
    elif cfg_label2id and isinstance(cfg_label2id, dict):
        # normalize keys to ints for id2label
        try:
            id2label = {int(k): str(v) for k, v in cfg_id2label.items()}
            label2id = {str(v): int(k) for k, v in cfg_id2label.items()}
        except Exception:
            id2label = {i: label for i, label in enumerate(unique_labels)}
            label2id = {label: i for i, label in enumerate(unique_labels)}
        config.num_labels = len(id2label)
    elif (
        cfg_label2id
        and isinstance(cfg_label2id, dict)
        and len(cfg_label2id) == len(unique_labels)
    ):
        try:
            label2id = {str(k): int(v) for k, v in cfg_label2id.items()}
            id2label = {int(v): str(k) for k, v in cfg_label2id.items()}
        except Exception:
            id2label = {i: label for i, label in enumerate(unique_labels)}
            label2id = {label: i for i, label in enumerate(unique_labels)}
        config.num_labels = len(label2id)
    else:
        # No mapping in checkpoint: require unique_labels to be provided
        if unique_labels is None:
            raise ValueError(
                "Model checkpoint has no label mapping and no unique_labels were provided."
            )
        config.num_labels = len(unique_labels)
        id2label = {i: label for i, label in enumerate(unique_labels)}
        label2id = {label: i for i, label in enumerate(unique_labels)}

    # store mappings on config
    config.id2label = {str(k): v for k, v in id2label.items()}
    config.label2id = {str(k): v for k, v in label2id.items()}

    if use_Roberta_tokenizer:
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name, use_fast=True)
        model = RobertaForTokenClassification.from_pretrained(model_name, config=config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, config=config
        )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = {
        "train_batch_size": 8,
        "learning_rate": 5e-5,
        "num_train_epochs": 10,
        "use_cuda": torch.cuda.is_available(),
    }

    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "label2id": label2id,
        "id2label": id2label,
        "device": device,
        "training_args": training_args,
    }


class TokenClassificationDataset(Dataset):
    def __init__(
        self, encodings: typing.List[dict], labels: typing.List[typing.List[int]]
    ):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict:
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
    max_length: int = None,
    only_target_token: bool = False,
    pad_token_label_id: int = -100,
) -> typing.Tuple[typing.List[dict], typing.List[typing.List[int]]]:
    """
    Tokenize and align labels. Returns list of per-example encodings (dicts) and label id lists.
    Uses -100 for subtokens we want to ignore for loss computation.
    Requires a fast tokenizer (use_fast=True).
    """
    encodings = []
    all_label_ids: typing.List[typing.List[int]] = []

    for words, labels in sentences:
        # tokenise with word-level input
        enc = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        word_ids = enc.word_ids()  # available for fast tokenizers
        # Determine target word indices if only_target_token is enabled
        target_word_indices = None
        if only_target_token:

            def _is_target_label(l: str) -> bool:
                if l is None:
                    return False
                nl = str(l).strip()
                return nl != "" and nl.upper() not in {"NONE", "-"}

            target_word_indices = {
                i for i, lab in enumerate(labels) if _is_target_label(lab)
            }
            # If none found, fall back to include all
            if not target_word_indices:
                target_word_indices = None

        label_ids = []
        previous_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                label_ids.append(pad_token_label_id)
            else:
                # If only training on target token(s), mask other words completely
                if (
                    target_word_indices is not None
                    and word_idx not in target_word_indices
                ):
                    label_ids.append(pad_token_label_id)
                else:
                    label = labels[word_idx]
                    if word_idx != previous_word_idx:
                        label_ids.append(label2id[label])
                    else:
                        # For sub-word tokens of the same word: ignore for loss
                        label_ids.append(-100)
                previous_word_idx = word_idx

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        token_type_ids = enc.get("token_type_ids")

        # If max_length is set, truncate or pad to that length (like convert_examples_to_features)
        if max_length is not None:
            # truncate if necessary
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                if token_type_ids is not None:
                    token_type_ids = token_type_ids[:max_length]
                label_ids = label_ids[:max_length]

            # pad if necessary
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                pad_id = (
                    getattr(tokenizer, "pad_token_id", None)
                    or getattr(tokenizer, "eos_token_id", None)
                    or 0
                )
                input_ids = input_ids + [pad_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
                if token_type_ids is not None:
                    pad_type = getattr(tokenizer, "pad_token_type_id", 0)
                    token_type_ids = token_type_ids + [pad_type] * pad_len
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
    best_model_dir: typing.Optional[str] = None,
    save_model_every_epoch: bool = False,
    save_steps: int = -1,
    device: typing.Optional[torch.device] = None,
    silent: bool = False,
    gradient_accumulation_steps: int = 1,
    max_length: typing.Optional[int] = None,
    only_target_token: bool = False,
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
      best_model_dir: where to save best model (if provided).
      save_model_every_epoch: save a checkpoint every epoch (if True).
      save_steps: not used here because checkpointing is epoch-based; keep for API compatibility.
      device: torch.device; auto-detected if None.
      silent: if True reduce tqdm output.

    Returns:
      dict with training statistics and last evaluation report.
    """
    import random
    import numpy as np

    # reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # label maps: allow passing either a list of labels or a mapping label->id
    if isinstance(label_list, dict):
        label2id = {str(k): int(v) for k, v in label_list.items()}
        id2label = {int(v): str(k) for k, v in label_list.items()}
    else:
        label2id = {str(lbl): i for i, lbl in enumerate(label_list)}
        id2label = {i: lbl for lbl, i in label2id.items()}

    # Validate that all labels in the training (and eval) data exist in label2id
    def _collect_unique_labels(df):
        if df is None:
            return set()
        return set(df["labels"].astype(str).unique())

    train_labels_in_data = _collect_unique_labels(train_df)
    eval_labels_in_data = _collect_unique_labels(eval_df)

    # When training only on the target token(s), some datasets use placeholder labels
    # (e.g. '-', 'NONE' or empty string) for non-target tokens. Exclude those from
    # the mapping validation so they don't cause spurious errors.
    def _is_non_target_label(l: str) -> bool:
        if l is None:
            return True
        nl = str(l).strip()
        return nl == "" or nl.upper() in {"NONE", "-"}

    if only_target_token:
        train_labels_in_data = {
            l for l in train_labels_in_data if not _is_non_target_label(l)
        }
        eval_labels_in_data = {
            l for l in eval_labels_in_data if not _is_non_target_label(l)
        }

    missing_in_model = sorted(
        list((train_labels_in_data | eval_labels_in_data) - set(label2id.keys()))
    )
    if missing_in_model:
        raise ValueError(
            f"The following labels are present in the dataset but missing in model label mapping: {missing_in_model[:50]}."
            " Provide a mapping that includes these labels or update the model config before training."
        )

    # Prepare train examples
    train_sentences = _group_sentences_from_df(train_df)
    if len(train_sentences) == 0:
        raise ValueError("No training sentences found in train_df")

    encodings_train, labels_train = prepare_token_classification_data(
        tokenizer,
        train_sentences,
        label2id,
        max_length=max_length,
        only_target_token=only_target_token,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    train_dataset = TokenClassificationDataset(encodings_train, labels_train)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_batch_size,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True,
    )

    # Optional evaluation dataset
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

    # Optimiser & scheduler
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    t_total = len(train_loader) // gradient_accumulation_steps * num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=t_total
    )

    # Training loop
    best_val_f1 = -float("inf")
    best_state = None
    patience = 3 if use_early_stopping else None
    patience_counter = 0

    global_step = 0
    training_stats = {"train_loss": [], "eval_f1": []}

    for epoch in range(int(num_train_epochs)):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_train_epochs}", disable=silent
        )
        for step, batch in enumerate(progress):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item()

            # Update running train loss in the progress bar for visibility
            if not silent:
                try:
                    current_avg = epoch_loss / (step + 1)
                    progress.set_postfix({"train_loss": f"{current_avg:.4f}"})
                except Exception:
                    pass

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if save_steps > 0 and global_step % save_steps == 0:
                # optional checkpointing by step (not often used for epoch-based workflows)
                ckpt_dir = os.path.join(output_dir, f"checkpoint-step-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                if not dry_run:
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

        avg_epoch_loss = epoch_loss / (len(train_loader) if len(train_loader) else 1)
        training_stats["train_loss"].append(avg_epoch_loss)

        # Show epoch-level train loss on the progress bar
        if not silent:
            try:
                progress.set_postfix({"train_loss": f"{avg_epoch_loss:.4f}"})
            except Exception:
                pass

        # Evaluation
        eval_report = None
        if evaluate_during_training and eval_loader is not None:
            model.eval()
            preds_all = []
            labels_all = []
            eval_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(eval_loader, desc="Evaluating", disable=silent):
                    b = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**b)
                    eval_loss += outputs.loss.item()
                    logits = outputs.logits.detach().cpu().numpy()
                    label_ids = b["labels"].detach().cpu().numpy()
                    # convert logits -> predicted label ids (per token)
                    pred_ids = logits.argmax(axis=-1)
                    # collect token-level predictions and labels, ignoring -100
                    for p_row, l_row in zip(pred_ids, label_ids):
                        preds = []
                        labs = []
                        for p, l in zip(p_row, l_row):
                            if l == -100:
                                continue
                            preds.append(id2label[int(p)])
                            labs.append(id2label[int(l)])
                        preds_all.append(preds)
                        labels_all.append(labs)

            avg_eval_loss = eval_loss / (len(eval_loader) if len(eval_loader) else 1)
            f1 = f1_score(labels_all, preds_all)
            eval_report = {
                "f1": f1,
                "loss": avg_eval_loss,
                "report": classification_report(labels_all, preds_all),
            }
            training_stats["eval_f1"].append(f1)

            # Early stopping / best-model logic uses validation F1
            if best_model_dir and not dry_run:
                os.makedirs(best_model_dir, exist_ok=True)
            if f1 > best_val_f1:
                best_val_f1 = f1
                patience_counter = 0
                if best_model_dir:
                    # save best model
                    if not dry_run:
                        model.save_pretrained(best_model_dir)
                        tokenizer.save_pretrained(best_model_dir)
                # also keep copy in memory
                best_state = deepcopy(model.state_dict())
            else:
                if use_early_stopping:
                    patience_counter += 1
                    if patience is not None and patience_counter >= patience:
                        # Early stop
                        if best_state is not None:
                            model.load_state_dict(best_state)
                        break

            # After evaluation, update the progress bar with epoch-level metrics
            if not silent:
                try:
                    postfix = {
                        "train_loss": f"{avg_epoch_loss:.4f}",
                        "eval_f1": f"{f1:.4f}",
                        "eval_loss": f"{avg_eval_loss:.4f}",
                    }
                    progress.set_postfix(postfix)
                except Exception:
                    pass

        # epoch checkpointing
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
