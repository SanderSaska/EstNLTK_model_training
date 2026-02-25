#!/usr/bin/env python3
"""Training CLI for token-classification using transformers (no simpletransformers).

Usage: python train.py --train-csv train.csv --output-dir out_model --model-name camembert-base
"""

import argparse
import os
import pandas as pd

from preprocessing import initialize_model, train_token_classification


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--eval-csv", required=False)
    p.add_argument("--model-name", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--labels-json", required=False, help="Optional JSON file with list of labels to override model mapping")
    p.add_argument("--only-target-token", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Run training but do not save checkpoints or overwrite model")
    args = p.parse_args()

    train_df = pd.read_csv(args.train_csv)
    eval_df = pd.read_csv(args.eval_csv) if args.eval_csv else None

    # By default prefer the model's own label mapping to preserve the classifier head.
    # If you want to override, provide --labels-json path to a JSON list of labels.
    if args.labels_json:
        import json

        with open(args.labels_json, "r", encoding="utf-8") as fh:
            labels = json.load(fh)
        model_bundle = initialize_model(args.model_name, labels, no_progress_bars=args.no_progress)
    else:
        model_bundle = initialize_model(args.model_name, None, no_progress_bars=args.no_progress)

    # Decide which label mapping to pass to trainer: model mapping or provided labels
    if args.labels_json:
        label_arg = labels
    else:
        label_arg = model_bundle["label2id"]

    res = train_token_classification(
        model=model_bundle["model"],
        tokenizer=model_bundle["tokenizer"],
        train_df=train_df,
        label_list=label_arg,
        output_dir=args.output_dir,
        eval_df=eval_df,
        num_train_epochs=model_bundle["training_args"]["num_train_epochs"],
        train_batch_size=model_bundle["training_args"]["train_batch_size"],
        learning_rate=model_bundle["training_args"]["learning_rate"],
        evaluate_during_training=False if eval_df is None else True,
        use_early_stopping=False,
        best_model_dir=os.path.join(args.output_dir, "best_model"),
        save_model_every_epoch=False,
        save_steps=-1,
        device=model_bundle["device"],
        silent=args.no_progress,
        only_target_token=args.only_target_token,
        dry_run=args.dry_run,
    )

    print("Training finished. Results:")
    print(res)


if __name__ == "__main__":
    main()
