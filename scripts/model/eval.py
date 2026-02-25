#!/usr/bin/env python3
"""Evaluation CLI for token-classification models saved with Hugging Face format.

Usage: python eval.py --model-dir out_model --eval-csv eval.csv
"""

import argparse
import os
import pandas as pd
import torch

from preprocessing import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    _group_sentences_from_df,
    prepare_token_classification_data,
    TokenClassificationDataset,
    DataCollatorForTokenClassification,
)

from torch.utils.data import DataLoader
from seqeval.metrics import classification_report, f1_score


def load_model_and_tokenizer(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def evaluate_model(
    model_dir: str, eval_csv: str, batch_size: int = 8, max_length: int = None
):
    df = pd.read_csv(eval_csv)
    model, tokenizer, device = load_model_and_tokenizer(model_dir)
    # Prefer label mapping from model config when available
    cfg_label2id = getattr(model.config, "label2id", None)
    cfg_id2label = getattr(model.config, "id2label", None)
    if cfg_label2id:
        # ensure keys are str labels -> int ids
        label2id = {str(k): int(v) for k, v in cfg_label2id.items()}
        id2label = {int(v): str(k) for k, v in cfg_label2id.items()}
    elif cfg_id2label:
        # invert id2label
        id2label = {int(k): str(v) for k, v in cfg_id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
    else:
        # fallback to labels found in CSV
        unique_labels = sorted(df.labels.unique())
        label2id = {str(l): i for i, l in enumerate(unique_labels)}
        id2label = {i: str(l) for l, i in label2id.items()}

    sentences = _group_sentences_from_df(df)
    encodings, labels = prepare_token_classification_data(
        tokenizer,
        sentences,
        label2id,
        max_length=max_length,
    )
    dataset = TokenClassificationDataset(encodings, labels)
    collator = DataCollatorForTokenClassification(tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    model.eval()
    preds_all = []
    labels_all = []
    # ensure id2label mapping uses ints
    model_id2label = None
    if cfg_id2label:
        model_id2label = {int(k): str(v) for k, v in cfg_id2label.items()}
    elif cfg_label2id:
        model_id2label = {int(v): str(k) for k, v in cfg_label2id.items()}
    else:
        model_id2label = id2label

    with torch.no_grad():
        for batch in loader:
            b = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**b)
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = b["labels"].detach().cpu().numpy()
            pred_ids = logits.argmax(axis=-1)

            # convert ids to label strings and remove -100
            for p_row, l_row in zip(pred_ids, label_ids):
                preds = []
                labs = []
                for p, l in zip(p_row, l_row):
                    if l == -100:
                        continue
                    pred_label = model_id2label.get(int(p), str(int(p)))
                    true_label = model_id2label.get(int(l), str(int(l))) if model_id2label is not None else None
                    preds.append(pred_label)
                    labs.append(true_label if true_label is not None else str(int(l)))
                preds_all.append(preds)
                labels_all.append(labs)

    # Both lists contain ids as strings; return simple metrics
    f1 = f1_score(labels_all, preds_all)
    report = classification_report(labels_all, preds_all)
    return {"f1": f1, "report": report}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--eval-csv", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    res = evaluate_model(args.model_dir, args.eval_csv, batch_size=args.batch_size)
    print(res["report"])


if __name__ == "__main__":
    main()
