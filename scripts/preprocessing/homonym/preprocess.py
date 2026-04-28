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
                        label = sentence.meta["class_label"]
                        data.append(
                            {
                                "num": num,
                                "inflection_type": infl_type,
                                "sentence": sentence.text,
                                "word": annotation.text,
                                "word_span": tuple([annotation.start, annotation.end]),
                                "label": label,
                                "source": input_file.name,
                            }
                        )
            if do_individual_dfs:
                df = pd.DataFrame(data)
                output_parquet = output_dir / Path(
                    f"homonyms_sentences_infltype_{num}_{infl_type}.parquet"
                )
                df.to_parquet(output_parquet, index=False)
                print(f"Saved processed data to {output_parquet}")
            overall_data.extend(data)

        if do_overall_df:
            # Create overall dataframe
            overall_df = pd.DataFrame(overall_data)
            overall_output_parquet = output_dir / Path(
                "homonyms_overall_sentences.parquet"
            )
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
        global_sentence_id = 0
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
            for sentence in inner:
                sentence_id = global_sentence_id
                global_sentence_id += 1
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
                                "label": label,
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
                                "label": "-",
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

        global_sentence_id = 0
        for _, row in homonym_dataset.iterrows():
            sentence_id = global_sentence_id
            global_sentence_id += 1
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
                        "word": word_text,
                        "label": label,
                    }
                )

        df_out = pd.DataFrame(
            processed_sentences, columns=["sentence_id", "word", "label"]
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
        seed: int | None = None,
        sentence_id_col: str = "sentence_id",
        infl_type_col: str = "infl_type",
        label_col: str = "label",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split token-level data into train and test sets using sentence-level
        stratification.

        Each sentence is assigned to a stratum built from (`infl_type`,
        `label_col`) using the single labelled token in the sentence
        (`label_col != "-"). Sentence IDs are treated as globally unique and
        are split directly without coupling to source.

        Args:
            df (pd.DataFrame): Input DataFrame to split.
            test_size (float, optional): Proportion of test data. Defaults to 0.2.
            seed (int | None, optional): Random seed for reproducibility. Defaults to None.
            sentence_id_col (str): Column containing sentence IDs. Defaults to "sentence_id".
            infl_type_col (str): Column containing inflection type. Defaults to "infl_type".
            label_col (str): Column containing the single stratification
                label. Defaults to "label".

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
        """
        if df.empty:
            return df.copy(), df.copy()

        if not (0.0 <= test_size < 1.0):
            raise ValueError("`test_size` must be in range [0.0, 1.0).")

        # Validate required columns.
        required_cols = {sentence_id_col, infl_type_col, label_col}
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns for train_test_split: {missing_cols}"
            )
        # If test_size is 0.0, return all data as train and an empty test set
        if test_size == 0.0:
            train_df = df.reset_index(drop=True)
            return train_df, train_df.iloc[0:0].copy()

        # Create a working copy of the DataFrame to avoid modifying the original
        working_df = df.reset_index(drop=True)
        rng = np.random.default_rng(seed)
        # Group by sentence_id to get sentence-level indices and metadata
        sentence_groups = working_df.groupby(sentence_id_col, sort=False).indices
        sentence_ids = list(sentence_groups.keys())
        total_sentences = len(sentence_ids)
        if total_sentences <= 1:
            return working_df.copy(), working_df.iloc[0:0].copy()

        # Assign each sentence to a stratum based on inflection type and homonym form.
        sentence_meta: dict[typing.Any, tuple[str, str]] = {}
        for sentence_id, positions in sentence_groups.items():
            sentence_rows = working_df.take(np.asarray(positions, dtype=int))
            infl_type_value = str(sentence_rows[infl_type_col].iloc[0])

            labelled_rows = sentence_rows[sentence_rows[label_col].astype(str) != "-"]
            if not labelled_rows.empty:
                form_value = str(labelled_rows[label_col].iloc[0])
            else:
                form_value = "unlabelled"

            sentence_meta[sentence_id] = (infl_type_value, form_value)

        # Stratify at sentence level: one sentence belongs to one stratum.
        strata: dict[tuple[str, str], list[typing.Any]] = {}
        for sentence_id in sentence_ids:
            key = sentence_meta[sentence_id]
            if key not in strata:
                strata[key] = []
            strata[key].append(sentence_id)
        # Perform stratified sampling of sentences for the test set.
        test_sentence_ids: set[typing.Any] = set()
        for stratum_sentence_ids in strata.values():
            stratum_size = len(stratum_sentence_ids)
            if stratum_size == 1:
                continue
            # Calculate the number of sentences to sample for the test set from this stratum, ensuring at least one sentence is included if test_size > 0 and the stratum has enough sentences.
            target_test = int(round(stratum_size * test_size))
            if test_size > 0.0 and target_test == 0:
                target_test = 1
            if target_test >= stratum_size:
                target_test = stratum_size - 1
            if target_test <= 0:
                continue
            # Randomly sample sentence IDs for the test set from this stratum without replacement.
            chosen_idx = rng.permutation(stratum_size)[:target_test]
            chosen_ids = [stratum_sentence_ids[int(i)] for i in chosen_idx]
            test_sentence_ids.update(chosen_ids)

        # Ensure test set is non-empty when test_size > 0 and enough sentences exist.
        if test_size > 0.0 and not test_sentence_ids and total_sentences > 1:
            random_idx = int(rng.integers(0, total_sentences))
            random_sentence = sentence_ids[random_idx]
            test_sentence_ids.add(random_sentence)
        # Derive train sentence IDs as those not in the test set, ensuring at least one sentence is included in the train set if possible.
        train_sentence_ids = [
            sid for sid in sentence_ids if sid not in test_sentence_ids
        ]
        if not train_sentence_ids:
            moved_sentence = next(iter(test_sentence_ids))
            test_sentence_ids.remove(moved_sentence)
            train_sentence_ids = [
                sid for sid in sentence_ids if sid not in test_sentence_ids
            ]
        # Construct train and test DataFrames by concatenating sentence groups in the order of sentence_ids to preserve original sentence order within splits.
        train_positions = (
            np.concatenate([sentence_groups[sid] for sid in train_sentence_ids])
            if train_sentence_ids
            else np.array([], dtype=int)
        )
        test_sentence_ordered = [
            sid for sid in sentence_ids if sid in test_sentence_ids
        ]
        test_positions = (
            np.concatenate([sentence_groups[sid] for sid in test_sentence_ordered])
            if test_sentence_ordered
            else np.array([], dtype=int)
        )

        train_df = working_df.take(train_positions).reset_index(drop=True)
        test_df = working_df.take(test_positions).reset_index(drop=True)

        return train_df, test_df
