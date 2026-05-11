"""Stable public API for model package utilities.

Keep imports in other modules pointing to ``scripts.model`` where practical,
so internal file moves only require updates in this file.
"""

from scripts.model.utils import (
    TokenClassificationDataset,
    _group_sentences_from_df,
    _is_placeholder_label,
    _normalise_placeholder_labels,
    _parse_placeholder_labels,
    _select_placeholder_output_label,
    _tokenize_word_for_output,
    compare_label_lists,
    initialize_model,
    load_test_df,
    load_df,
    prepare_shared_inputs,
    prepare_token_classification_data,
)

from scripts.model.predict import (
    predictions_to_dataframe,
    run_hybrid_density_predict,
    run_single_model_predict,
)

from scripts.model.eval import (
    load_homonym_set,
    print_hybrid_gate_stats,
    print_metrics,
    get_metrics,
    run_hybrid_density_eval,
    run_single_model_eval,
)

from scripts.model.train import (
    train_token_classification,
)

from scripts.model.bert_morph_tagger import BertMorphTagger

__all__ = [
    # utils
    "TokenClassificationDataset",
    "_group_sentences_from_df",
    "_is_placeholder_label",
    "_normalise_placeholder_labels",
    "_parse_placeholder_labels",
    "_select_placeholder_output_label",
    "_tokenize_word_for_output",
    "compare_label_lists",
    "initialize_model",
    "load_test_df",
    "load_df",
    "prepare_shared_inputs",
    "prepare_token_classification_data",
    # predict
    "run_hybrid_density_predict",
    "run_single_model_predict",
    "predictions_to_dataframe",
    # eval
    "load_homonym_set",
    "print_hybrid_gate_stats",
    "print_metrics",
    "get_metrics",
    "run_hybrid_density_eval",
    "run_single_model_eval",
    # train
    "train_token_classification",
    # bert_morph_tagger
    "BertMorphTagger",
]
