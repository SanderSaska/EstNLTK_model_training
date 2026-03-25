import typing
import pandas as pd
import numpy as np
import tqdm
import estnltk
from pathlib import Path
from estnltk.converters.label_studio.labelling_configurations import (
    PhraseClassificationConfiguration,
)
from estnltk.converters.label_studio.labelling_tasks import PhraseClassificationTask


class Preprocessor:
    """
    Preprocessor class for homonym disambiguation dataset.
    TODO: Add description of the class and its methods.
    """

    def __init__(self):
        pass

    @staticmethod
    def create_sentences_df(
        input_files: typing.List[typing.Tuple],
        output_dir: typing.Union[str, Path],
        do_overall_df: bool = True,
        do_individual_dfs: bool = False,
    ):
        # Use specific annotation configurations that were used in homonyms dataset
        annotation_confs = {
            1: PhraseClassificationConfiguration(
                phrase_labels=["analüüsitav sõna"],
                class_labels={"sg n": "sg n", "sg g": "sg g"},
                header="Vali sõna morfoloogiline vorm (sg n - ainsuse nimetav, sg g -- ainsuse omastav):",
                header_placement="middle",
            ),
            16: PhraseClassificationConfiguration(
                phrase_labels=["analüüsitav sõna"],
                class_labels={"sg n": "sg n", "sg g": "sg g"},
                header="Vali sõna morfoloogiline vorm (sg n - ainsuse nimetav, sg g -- ainsuse omastav):",
                header_placement="middle",
            ),
            17: PhraseClassificationConfiguration(
                phrase_labels=["analüüsitav sõna"],
                class_labels={"sg n": "sg n", "sg g": "sg g", "sg p": "sg p"},
                header="Vali sõna morfoloogiline vorm (sg n - ainsuse nimetav, sg g - ainsuse omastav, sg p - ainsuse osastav):",
                header_placement="middle",
            ),
            19: PhraseClassificationConfiguration(
                phrase_labels=["analüüsitav sõna"],
                class_labels={"sg g": "sg g", "sg p": "sg p", "adt": "adt"},
                header="Vali sõna morfoloogiline vorm (sg g - ainsuse omastav, sg p - ainsuse osastav, adt - lühike sisseütlev):",
                header_placement="middle",
            ),
        }

        overall_data = []
        # Extract data from input files
        for infl_type, input_file in input_files:
            print(f"Processing file: {input_file} (inflection type {infl_type})")
            num = int(input_file.parent.stem)
            annotation_conf = annotation_confs[infl_type]
            with open(input_file, "r", encoding="utf-8") as f:
                raw = f.read()
            task = PhraseClassificationTask(
                annotation_conf,
                input_layer="morph",
                output_layer="morph",
                label_attribute="label",
            )
            classified_sentences = task.import_data(raw)

            # Create dataframe from classified sentences
            data = []
            for sentence in classified_sentences:
                for annotation in sentence.morph:
                    if (
                        "class_label" in sentence.meta
                        and sentence.meta["class_label"] is not None
                    ):
                        data.append(
                            {
                                "num": num,
                                "inflection_type": infl_type,
                                "sentence": sentence.text,
                                "word": annotation.text,
                                "word_span": (annotation.start, annotation.end),
                                "label": sentence.meta["class_label"],
                            }
                        )
            if do_individual_dfs:
                df = pd.DataFrame(data)
                output_parquet = output_dir / Path(f"homonyms_sentences_infltype_{num}_{infl_type}.parquet")
                df.to_parquet(output_parquet, index=False)
                print(f"Saved processed data to {output_parquet}")
            overall_data.extend(data)

        if do_overall_df:
            # Create overall dataframe
            overall_df = pd.DataFrame(overall_data)
            overall_output_parquet = output_dir / Path("homonyms_overall_sentences.parquet")
            overall_df.to_parquet(overall_output_parquet, index=False)
            print(f"Saved overall processed data to {overall_output_parquet}")

    @staticmethod
    def create_model_df(
        input_files: typing.List[typing.Tuple],
        output_dir: typing.Union[str, Path],
        do_individual_dfs: bool = False,
        do_overall_df: bool = True,
    ):
        # Use specific annotation configurations that were used in homonyms dataset
        annotation_confs = {
            1: PhraseClassificationConfiguration(
                phrase_labels=["analüüsitav sõna"],
                class_labels={"sg n": "sg n", "sg g": "sg g"},
                header="Vali sõna morfoloogiline vorm (sg n - ainsuse nimetav, sg g -- ainsuse omastav):",
                header_placement="middle",
            ),
            16: PhraseClassificationConfiguration(
                phrase_labels=["analüüsitav sõna"],
                class_labels={"sg n": "sg n", "sg g": "sg g"},
                header="Vali sõna morfoloogiline vorm (sg n - ainsuse nimetav, sg g -- ainsuse omastav):",
                header_placement="middle",
            ),
            17: PhraseClassificationConfiguration(
                phrase_labels=["analüüsitav sõna"],
                class_labels={"sg n": "sg n", "sg g": "sg g", "sg p": "sg p"},
                header="Vali sõna morfoloogiline vorm (sg n - ainsuse nimetav, sg g - ainsuse omastav, sg p - ainsuse osastav):",
                header_placement="middle",
            ),
            19: PhraseClassificationConfiguration(
                phrase_labels=["analüüsitav sõna"],
                class_labels={"sg g": "sg g", "sg p": "sg p", "adt": "adt"},
                header="Vali sõna morfoloogiline vorm (sg g - ainsuse omastav, sg p - ainsuse osastav, adt - lühike sisseütlev):",
                header_placement="middle",
            ),
        }

        overall_data = []
        # Extract data from input files
        for i, (infl_type, input_file) in enumerate(input_files):
            num = int(input_file.parent.stem)  # Numbers (V)1 or (V)2 in filenames
            annotation_conf = annotation_confs[infl_type]
            with open(input_file, "r", encoding="utf-8") as f:
                raw = f.read()
            task = PhraseClassificationTask(
                annotation_conf,
                input_layer="morph",
                output_layer="morph",
                label_attribute="label",
            )
            classified_sentences = task.import_data(raw)

            # Create dataframe from classified sentences
            data = []
            inner = tqdm.tqdm(
                classified_sentences,
                desc="Processing sentences",
                leave=False,
                position=0,
            )
            for sentence_id, sentence in enumerate(inner):
                # Create estnltk Text object from sentence text
                text = estnltk.Text(sentence.text)
                # Apply morphological analysis to the text
                text.tag_layer("morph_analysis")
                # Check if annotation has the expected label and meta information
                label = None
                if (
                    "class_label" in sentence.meta
                    and sentence.meta["class_label"] is not None
                ):
                    label = sentence.meta["class_label"]
                if label is None:
                    # print(
                    #     f"Warning: No valid label found for sentence {sentence_id} in file {input_file.name}. Skipping this sentence."
                    # )
                    continue
                # Get the span of the annotated word (assuming it's the only labelled word in the sentence)
                labelled_word_span = (sentence.morph[0].start, sentence.morph[0].end)
                for ma in text.morph_analysis:
                    pos = ma.partofspeech[0]  # Take the first analysis
                    form = ma.form[0]  # Take the first analysis
                    # Check if the morphological analysis span matches the annotated word span
                    if (
                        ma.start == labelled_word_span[0]
                        and ma.end == labelled_word_span[1]
                        and label is not None
                    ):
                        if isinstance(label, list):
                            form = label[
                                0
                            ]  # ["sg n"] or ["sg g"] etc. from the annotation
                        if isinstance(label, str):
                            form = label[
                                2:-2
                            ]  # Remove quotes and possible trailing characters from the label string
                        label = form + "_" + pos
                        # Use the label from the annotation for this word
                        data.append(
                            {
                                "sentence_id": sentence_id,
                                "words": ma.text,
                                "form": form,
                                "pos": pos,
                                "labels": label,
                                "infl_type": infl_type,
                                "source": input_file.name,
                            }
                        )
                    else:
                        # If the morphological analysis span does not match the annotated word span,
                        # add word without form and pos information
                        data.append(
                            {
                                "sentence_id": sentence_id,
                                "words": ma.text,
                                "form": "-",
                                "pos": "-",
                                "labels": "-",
                                "infl_type": infl_type,
                                "source": input_file.name,
                            }
                        )
                inner.set_postfix({"Files processed": i})
                inner.refresh()
            inner.set_postfix({"Files processed": i + 1})
            inner.clear()
            inner.close()
            if do_individual_dfs:
                # Create and save individual dataframe for this file
                df = pd.DataFrame(data)
                output_parquet = output_dir / Path(
                    f"homonyms_infltype_{num}_{infl_type}.parquet"
                )
                df.to_parquet(output_parquet, index=False)
                print(f"Saved processed data to {output_parquet}")
            overall_data.extend(data)

        if do_overall_df:
            # Create overall dataframe
            overall_df = pd.DataFrame(overall_data)
            overall_output_parquet = output_dir / Path("homonyms_overall.parquet")
            overall_df.to_parquet(overall_output_parquet, index=False)
            print(f"Saved overall processed data to {overall_output_parquet}")

    @staticmethod
    def build_model_dataset_from_homonyms(
        homonym_dataset: pd.DataFrame,
        save_parquet: typing.Union[str, Path, None] = None,
    ) -> pd.DataFrame:
        """
        Build a token-level DataFrame suitable for model training from a homonym
        annotation DataFrame.

        The returned DataFrame has columns: `sentence_id`, `words`, `labels`.
        Labels use the homonym annotation (normalized) when available, otherwise
        a placeholder '-' which can be ignored during training.

        Args:
            homonym_dataset: DataFrame containing at least columns `sentence`,
                `label` and `word_span` (word_span may be a tuple or a string).
            save_parquet: optional path to save the resulting DataFrame as parquet.

        Returns:
            DataFrame with token-level rows for model consumption.
        """
        processed_sentences: list[dict] = []

        def _norm_label(lab) -> str:
            # Normalize label value coming either as a list or a string
            if isinstance(lab, list) and lab:
                core = str(lab[0])
            elif isinstance(lab, str):
                s = lab.strip()
                if s.startswith("['") and s.endswith("']"):
                    core = s[2:-2]
                else:
                    core = s
            else:
                core = ""
            return core

        for sentence_id, (_, row) in enumerate(homonym_dataset.iterrows()):
            sentence_text = row["sentence"]
            homonym_label = row.get("label")
            labelled_word_span = row.get("word_span")

            # make a reproducible string representation of the annotated span
            if isinstance(labelled_word_span, (tuple, list)):
                target_span_str = str(tuple(labelled_word_span))
            else:
                target_span_str = str(labelled_word_span)

            # Create an estnltk Text object and ensure morph_analysis layer
            text = estnltk.Text(sentence_text)
            text.tag_layer("morph_analysis")

            # Process every token in the morph_analysis layer
            for token in text.morph_analysis:
                span = str((token.start, token.end))
                word_text = token.text
                pos = token.partofspeech[0] if token.partofspeech else ""
                label = "-"
                if span == target_span_str:
                    core = _norm_label(homonym_label)
                    if core:
                        label = f"{core}_{pos}" if pos else core

                processed_sentences.append(
                    {
                        "sentence_id": sentence_id,
                        "words": word_text,
                        "labels": label,
                    }
                )

        df_out = pd.DataFrame(
            processed_sentences, columns=["sentence_id", "words", "labels"]
        )
        if save_parquet:
            Path(save_parquet).parent.mkdir(parents=True, exist_ok=True)
            df_out.to_parquet(save_parquet, index=False)
        return df_out

    @staticmethod
    def shuffle_blocks_by_source(
        data: pd.DataFrame, source_col: str = "source", seed: int | None = None
    ) -> pd.DataFrame:
        """
        Shuffle a DataFrame by blocks defined by `source_col`, preserving the original
        order of rows within each source block.

        Parameters
        - data: DataFrame to shuffle (not modified in-place).
        - source_col: column name used to form blocks (default "source").
        - seed: optional integer seed for reproducible shuffling.

        Returns
        - Shuffled DataFrame with blocks rearranged and internal block ordering preserved.
        """
        if data.empty:
            return data.copy()

        groups = data.groupby(
            source_col, sort=False
        ).groups  # dict: source -> Index of labels
        keys = list(groups.keys())

        rnd = np.random.default_rng(seed)
        rnd.shuffle(keys)

        if isinstance(data.index, pd.RangeIndex):
            ordered_pos = np.concatenate([groups[k].values for k in keys])
        else:
            idx = data.index
            ordered_pos = np.concatenate([idx.get_indexer_for(groups[k]) for k in keys])

        return data.take(ordered_pos).reset_index(drop=True)

    @staticmethod
    def train_test_split(
        df: pd.DataFrame,
        test_size: float = 0.2,
        source_col: str = "source",
        seed: int | None = None,
        homonym_col: str = "words",
        form_col: str = "form",
    ) -> tuple[pd.DataFrame, pd.DataFrame, list, list]:
        """
        Split a DataFrame into train and test sets by sentence blocks while
        guaranteeing minimum homonym coverage in training.

        Sentence blocks are identified by (`source_col`, `sentence_id`).
        The minimum training set contains at least one sentence for every
        observed (homonym word, form) pair on labelled rows (`labels != "-"`).

        After the minimum set is created, additional random sentence blocks are
        added to training until the test share approaches `test_size`, stopping
        before adding the next block would push test share below `test_size`.
        If the minimum set already implies a smaller test share than requested,
        coverage is prioritised and no blocks are removed.

        Args:
            df (pd.DataFrame): Input DataFrame to split.
            test_size (float, optional): Proportion of test data. Defaults to 0.2.
            source_col (str, optional): Column name for source identifiers. Defaults to "source".
            seed (int | None, optional): Random seed for reproducibility. Defaults to None.
            homonym_col (str, optional): Column containing homonym token text. Defaults to "words".
            form_col (str, optional): Column containing homonym form. Defaults to "form".

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, list, list]: (train_df, test_df, train_sources, test_sources)
             - train_df: DataFrame containing training data.
             - test_df: DataFrame containing test data.
             - train_sources: Ordered unique sources present in training set.
             - test_sources: Ordered unique sources present in test set.
        """
        if df.empty:
            return df.copy(), df.copy(), [], []

        if not (0.0 <= test_size < 1.0):
            raise ValueError("`test_size` must be in range [0.0, 1.0).")
        # Validate required columns
        sentence_col = "sentence_id"
        label_col = "labels"
        required_cols = {source_col, sentence_col, homonym_col, form_col, label_col}
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns for train_test_split: {missing_cols}"
            )
        # Create a working copy of the DataFrame to avoid modifying the original
        working_df = df.reset_index(drop=True)
        # Initialize random number generator for reproducibility
        rng = np.random.default_rng(seed)
        # Group by sentence blocks defined by (source, sentence_id)
        sentence_groups = working_df.groupby(
            [source_col, sentence_col], sort=False
        ).indices
        # Extract unique sentence keys and count total sentences
        sentence_keys = [
            typing.cast(tuple[typing.Any, typing.Any], key)
            for key in sentence_groups.keys()
        ]
        total_sentences = len(sentence_keys)
        if total_sentences == 0:
            return working_df.copy(), working_df.copy(), [], []
        # Map (homonym, form) pairs to the first sentence key where they appear on labelled rows
        pair_to_sentence: dict[tuple[str, str], tuple[typing.Any, typing.Any]] = {}
        for row in working_df.itertuples(index=False):
            label_value = str(getattr(row, label_col))
            homonym_value = getattr(row, homonym_col)
            form_value = getattr(row, form_col)
            # Only consider rows with valid labels and non-missing homonym and form values
            if (
                label_value == "-"
                or pd.isna(homonym_value)
                or pd.isna(form_value)
                or str(homonym_value) == "-"
                or str(form_value) == "-"
            ):
                continue
            # Use string representations of homonym and form to ensure hashability and consistency
            pair_key = (str(homonym_value), str(form_value))
            sentence_key = (getattr(row, source_col), getattr(row, sentence_col))
            # Map the pair to the first sentence key where it appears, ensuring coverage of all pairs in training
            if pair_key not in pair_to_sentence:
                pair_to_sentence[pair_key] = sentence_key
        # Start with the minimum training set that covers all observed (homonym, form) pairs
        selected_sentences: set[tuple[typing.Any, typing.Any]] = set(
            pair_to_sentence.values()
        )
        train_sentence_count = len(selected_sentences)
        # Randomly add more sentence blocks to training until test share approaches `test_size`, stopping before adding the next block would push test share below `test_size`
        remaining_sentence_keys = [
            key for key in sentence_keys if key not in selected_sentences
        ]
        # Shuffle the remaining sentence keys to ensure random selection of additional blocks
        rng.shuffle(remaining_sentence_keys)
        # Iterate over the remaining sentence keys and add them to training if it does not reduce test share below `test_size`
        for sentence_key in remaining_sentence_keys:
            current_test_share = 1.0 - (train_sentence_count / total_sentences)
            if current_test_share <= test_size:
                break

            next_train_count = train_sentence_count + 1
            next_test_share = 1.0 - (next_train_count / total_sentences)
            if next_test_share < test_size:
                break

            selected_sentences.add(sentence_key)
            train_sentence_count = next_train_count
        # Create train and test DataFrames based on the selected sentence keys
        train_sentence_keys = [
            key for key in sentence_keys if key in selected_sentences
        ]
        test_sentence_keys = [
            key for key in sentence_keys if key not in selected_sentences
        ]
        # Concatenate the indices of the selected sentence blocks to create train and test sets, preserving the original order of sentences within each block
        train_positions = (
            np.concatenate([sentence_groups[key] for key in train_sentence_keys])
            if train_sentence_keys
            else np.array([], dtype=int)
        )
        test_positions = (
            np.concatenate([sentence_groups[key] for key in test_sentence_keys])
            if test_sentence_keys
            else np.array([], dtype=int)
        )
        # Use .take() to select rows by position and reset index for the resulting DataFrames
        train_df = working_df.take(train_positions).reset_index(drop=True)
        test_df = working_df.take(test_positions).reset_index(drop=True)
        # Extract the unique sources present in train and test sets, preserving the order of first occurrence in the original DataFrame
        train_sources = list(dict.fromkeys([key[0] for key in train_sentence_keys]))
        test_sources = list(dict.fromkeys([key[0] for key in test_sentence_keys]))

        return train_df, test_df, train_sources, test_sources
