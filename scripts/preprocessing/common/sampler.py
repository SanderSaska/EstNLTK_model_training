"""Corpus sampler CLI for EstNLTK_model_training.

Selects whole sources or individual sentences from three corpora to
meet a target number of words while respecting corpus proportions
(alpha, beta, gamma). Writes sampled Parquet and a JSON report with
provenance metadata.

Usage (example):
 python sampler.py --target-words 5000000 \
     --enc-path ../data/enc2017/processed/enc2017_morph_analysis_updated.parquet \
     --ud-path ../data/ud_et_edt/processed/ud_edt_morph_analysis_updated.parquet \
     --hom-path ../data/homonymous_word_forms/processed/homonyms_overall.parquet \
     --mode source --alpha 0.6 --beta 0.3 --gamma 0.1 --output-dir ../../data/processed/sampled
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import pathlib
import random
import time
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

# use enc2017-specific stratified sampler when available
try:
    from scripts.preprocessing.enc2017.preprocess import Preprocessor as EncPreprocessor
except Exception:
    EncPreprocessor = None


@dataclass
class Item:
    id: str
    size: int
    meta: dict


def load_word_rows(path: str | pathlib.Path) -> pd.DataFrame:
    """Load a Parquet file containing word-level rows (one row per word).

    Expected columns (at least): `sentence_id`, `words`, `source`.
    """
    df = pd.read_parquet(path)
    if df.empty:
        return df
    # ensure required columns exist
    for col in ("sentence_id", "words", "source"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing in {path}")
    return df


def sentence_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with per-sentence sizes and source.

    Columns: `source`, `sentence_id`, `words_count` (int).
    """
    # each row is a word, so group by source + sentence_id
    grp = df.groupby(["source", "sentence_id"], sort=False).size().rename("words_count")
    res = grp.reset_index()
    return res


def source_sizes_from_sentences(sent_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentence-level sizes to per-source sizes."""
    grp = (
        sent_df.groupby("source", sort=False)["words_count"].sum().rename("words_count")
    )
    counts = sent_df.groupby("source", sort=False).size().rename("sent_count")
    res = pd.concat([grp, counts], axis=1).reset_index()
    return res


def greedy_pack(
    items: List[Item], target: int, seed: int | None = None
) -> Tuple[List[Item], int, int]:
    """Greedy packing similar to stratified_sample_by_type: select items to reach >= target.

    Returns (selected_items, accumulated, rounds_used)
    """
    rnd = random.Random(seed)
    pool = items.copy()
    selected: List[Item] = []
    accumulated = 0
    rounds = 0

    # Sort pool by size ascending for bisect
    pool.sort(key=lambda it: it.size)

    while accumulated < target:
        if not pool:
            # exhaustion: allow replacement by refilling pool
            pool = items.copy()
            pool.sort(key=lambda it: it.size)
            rounds += 1
            # shuffle to avoid deterministic repetition
            rnd.shuffle(pool)

        sizes = [it.size for it in pool]
        rem = target - accumulated
        idx_eq = bisect.bisect_left(sizes, rem)
        if idx_eq < len(sizes) and sizes[idx_eq] == rem:
            pick_idx = idx_eq
        else:
            idx_le = bisect.bisect_right(sizes, rem) - 1
            if idx_le >= 0:
                pick_idx = idx_le
            else:
                # pick smallest if all items are larger than rem
                pick_idx = 0

        it = pool.pop(pick_idx)
        selected.append(it)
        accumulated += it.size

    return selected, accumulated, rounds


def sample_corpus(
    df: pd.DataFrame,
    mode: str,
    target_words: int,
    seed: int | None = None,
) -> Tuple[pd.DataFrame, dict]:
    """Sample from a single corpus DataFrame until `target_words` reached.

    mode: 'source' or 'sentence'. Returns sampled rows and provenance info.
    """
    assert mode in ("source", "sentence")

    sent_df = sentence_sizes(df)

    if mode == "sentence":
        items = [
            Item(
                id=f"{row.source}|||{row.sentence_id}",
                size=int(row.words_count),
                meta={"source": row.source, "sentence_id": int(row.sentence_id)},
            )
            for row in sent_df.itertuples(index=False)
        ]
    else:
        src_df = source_sizes_from_sentences(sent_df)
        items = [
            Item(
                id=str(row.source),
                size=int(row.words_count),
                meta={"source": row.source},
            )
            for row in src_df.itertuples(index=False)
        ]

    selected_items, accumulated, rounds = greedy_pack(items, target_words, seed=seed)

    # build sampled rows
    if mode == "sentence":
        # parse ids back
        keys = [(it.meta["source"], it.meta["sentence_id"]) for it in selected_items]
        # filter df matching any (source,sentence_id)
        mask = pd.Series(False, index=df.index)
        for src, sid in keys:
            mask |= (df["source"] == src) & (df["sentence_id"] == sid)
        sampled = df[mask].copy()
    else:
        srcs = [it.meta["source"] for it in selected_items]
        sampled = df[df["source"].isin(srcs)].copy()

    prov = {
        "requested_target_words": int(target_words),
        "achieved_words": int(accumulated),
        "rounds_used": int(rounds),
        "selected_items": len(selected_items),
    }
    return sampled.reset_index(drop=True), prov


def sample_enc2017(
    df: pd.DataFrame, mode: str, target_words: int, seed: int | None = None
) -> Tuple[pd.DataFrame, dict]:
    """Use enc2017-specific stratified sampling by `type` when in source mode.

    Falls back to `sample_corpus` for sentence mode or when enc-specific sampler
    isn't available.
    """
    if mode == "source" and EncPreprocessor is not None:
        # shuffle by source blocks to preserve ordering and pass to stratified sampler
        shuffled = EncPreprocessor.shuffle_blocks_by_source(
            df, source_col="source", seed=seed
        )
        sampled_df = EncPreprocessor.stratified_sample_by_type(
            shuffled,
            N=int(target_words),
            source_col="source",
            type_col="type",
            seed=seed,
        )
        prov = {
            "requested_target_words": int(target_words),
            "achieved_words": int(len(sampled_df)),
            "rounds_used": 0,
            "selected_items": sampled_df["source"].nunique()
            if not sampled_df.empty
            else 0,
            "method": "enc2017_stratified_by_type",
        }
        return sampled_df.reset_index(drop=True), prov

    # fallback
    return sample_corpus(df, mode, target_words, seed=seed)


def deduplicate_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate sentences across corpora by comparing concatenated word sequences.

    Keeps the first occurrence and marks later duplicates.
    Returns DataFrame with an added `duplicated_sentence` boolean column.
    """
    # build sentence-level keys
    grp = df.groupby(["orig_corpus", "source", "sentence_id"], sort=False)
    records = []
    for (orig_corpus, source, sid), idx in grp.groups.items():
        words = df.loc[idx, "words"].astype(str).tolist()
        sent_text = " ".join(words)
        records.append(
            {
                "orig_corpus": orig_corpus,
                "source": source,
                "sentence_id": sid,
                "sentence_text": sent_text,
                "rows_idx": idx,
            }
        )

    sent_df = pd.DataFrame(records)
    # find duplicates by sentence_text
    sent_df["keep"] = ~sent_df["sentence_text"].duplicated(keep="first")

    # build mask for rows to keep
    keep_mask = pd.Series(False, index=df.index)
    dup_mask = pd.Series(False, index=df.index)
    for _, row in sent_df.iterrows():
        if row["keep"]:
            keep_mask.loc[row["rows_idx"]] = True
        else:
            dup_mask.loc[row["rows_idx"]] = True

    df = df.copy()
    df["duplicated_sentence"] = dup_mask
    return df


def assemble_and_write(
    sampled_list: List[Tuple[str, pd.DataFrame, dict]],
    output_dir: str | pathlib.Path,
    seed: int | None,
    sampling_mode: str,
    targets: dict,
):
    os.makedirs(output_dir, exist_ok=True)
    # attach provenance columns and concat
    parts = []
    for corpus_name, df, prov in sampled_list:
        df = df.copy()
        df["orig_corpus"] = corpus_name
        df["sample_round"] = prov.get("rounds_used", 0)
        df["seed"] = int(seed) if seed is not None else None
        df["sampling_mode"] = sampling_mode
        parts.append(df)

    if not parts:
        raise SystemExit("No sampled data to write")

    combined = pd.concat(parts, ignore_index=True)

    # deduplicate by sentence_text across corpora by default (caller ensures this behaviour)
    combined = deduplicate_sentences(combined)

    # assign global sentence id by grouping orig_corpus+source+sentence_id
    combined["global_sentence_id"] = combined.groupby(
        ["orig_corpus", "source", "sentence_id"]
    ).ngroup()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_parquet = pathlib.Path(output_dir) / f"sampled_dataset_{timestamp}.parquet"
    combined.to_parquet(out_parquet, index=False)

    report = {
        "timestamp": timestamp,
        "output_parquet": str(out_parquet),
        "seed": seed,
        "sampling_mode": sampling_mode,
        "targets": targets,
        "total_sentences": int(combined["global_sentence_id"].nunique()),
        "total_words": int(len(combined)),
    }
    report_path = pathlib.Path(output_dir) / f"sampling_report_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Wrote sampled dataset to {out_parquet}")
    print(f"Wrote sampling report to {report_path}")


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Sample corpora to reach a target number of words"
    )
    p.add_argument(
        "--target-words", type=int, required=True, help="Total target words (required)"
    )
    p.add_argument("--enc-path", type=str, required=True)
    p.add_argument("--ud-path", type=str, required=True)
    p.add_argument("--hom-path", type=str, required=True)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--mode", choices=("source", "sentence"), default="source")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, required=True)

    args = p.parse_args(list(argv) if argv is not None else None)

    target = args.target_words
    seed = args.seed
    mode = args.mode

    # load corpora
    enc_df = load_word_rows(args.enc_path)
    ud_df = load_word_rows(args.ud_path)
    hom_df = load_word_rows(args.hom_path)

    # default proportions: uniform if not provided
    alpha = args.alpha if args.alpha is not None else 1.0 / 3.0
    beta = args.beta if args.beta is not None else 1.0 / 3.0
    gamma = args.gamma if args.gamma is not None else 1.0 / 3.0

    # validate proportions
    s = alpha + beta + gamma
    if abs(s - 1.0) > 1e-6:
        raise SystemExit("alpha+beta+gamma must sum to 1.0")

    targets = {
        "enc2017": int(round(target * alpha)),
        "ud_et_edt": int(round(target * beta)),
        "homonyms": int(round(target * gamma)),
    }

    sampled_list = []

    enc_sampled, enc_prov = sample_enc2017(enc_df, mode, targets["enc2017"], seed=seed)
    sampled_list.append(("enc2017", enc_sampled, enc_prov))

    ud_sampled, ud_prov = sample_corpus(
        ud_df, mode, targets["ud_et_edt"], seed=seed + 1 if seed is not None else None
    )
    sampled_list.append(("ud_et_edt", ud_sampled, ud_prov))

    hom_sampled, hom_prov = sample_corpus(
        hom_df, mode, targets["homonyms"], seed=seed + 2 if seed is not None else None
    )
    sampled_list.append(("homonyms", hom_sampled, hom_prov))

    assemble_and_write(sampled_list, args.output_dir, seed, mode, targets)


if __name__ == "__main__":
    main()
