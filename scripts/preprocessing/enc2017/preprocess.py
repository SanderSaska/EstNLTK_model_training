import os
import typing
import tqdm
import csv
import estnltk
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pathlib

from tqdm import tqdm
import numpy as np
import random
from bisect import bisect_left, bisect_right

from scripts.model.bert_morph_tagger import BertMorphTagger
from sklearn.model_selection import train_test_split


class Preprocessor:
    """
    Preprocessor for the Estonian National Corpus 2017 (ENC2017).
    TODO: Add description of the class and its methods.
    """

    def __init__(self):
        pass

    @staticmethod
    def find_text_type(text: estnltk.Text, file_name: str):
        """
        Determines and assigns the text type metadata based on the provided file name.

        This function examines the text's metadata to check if a 'texttype' is already assigned.
        If the 'texttype' is missing, it will infer and assign a text type based on the file name's prefix.
        If inference fails, it assigns 'unknown' as the text type.

        Args:
            text (estnltk.Text): The EstNLTK Text object whose metadata is being processed.
            file_name (str): The name of the file being processed, used to determine the text type
                if it is not already present in the metadata.

        """
        text_type = text.meta.get("texttype")  # Text type
        if not text_type:
            if file_name.startswith("wiki17"):
                text.meta.update({"texttype": "wikipedia"})
            elif file_name.startswith("web13"):
                text.meta.update({"texttype": "blogs_and_forums"})
            else:
                print("Unknown text type. Assigning text type 'unknown'.")
                text.meta.update({"texttype": "unknown"})
        return

    @staticmethod
    def create_csv_file_by_file_enc2017(
        jsons: typing.List[str | pathlib.Path], csv_dir: str | pathlib.Path
    ):
        """
        Creates a CSV file for each text file. \n
        Skips CSV files that have already been created. \n
        For each <code>.json</code> file, the following info is gathered:
        <ul>
            <li><code>sentence_id</code> -- given for each sentence</li>
            <li><code>words</code> -- words gathered from text</li>
            <li><code>form</code> -- word form notation</li>
            <li><code>pos</code> -- part of speech</li>
            <li><code>type</code> -- text type (i.e. genre)</li>
            <li><code>source</code> -- file name where the text is taken from</li>
        </ul>
        <a href="https://github.com/Filosoft/vabamorf/blob/e6d42371006710175f7ec328c98f90b122930555/doc/tagset.md">Tables of morphological categories</a> for more information about <code>form</code> and <code>pos</code>.

        Args:
            jsons (list): List of json filepaths from which to read in the text
            csv_dir (str): Directory where to save the new csv files
        """
        print("Beginning to morphologically tag file by file")
        for file_path in tqdm(jsons):
            tokens = list()
            sentence_id = 0

            # Skipping previous CSV files
            file_name = pathlib.Path(file_path).stem + pathlib.Path(file_path).suffix
            csv_file_name = file_name[:-4] + "csv"
            if os.path.exists(os.path.join(csv_dir, csv_file_name)):
                # print(f"Skipping {file_name} as {csv_file_name} already exists.")
                continue

            # print(f"Beginning to tag {file_name}")

            # Morph. tagging using estnltk
            text = estnltk.converters.json_to_text(file=file_path)
            Preprocessor.find_text_type(
                text, file_name
            )  # Assign text type metadata if not already assigned
            morph_analysis = text.tag_layer("morph_analysis")
            for sentence in morph_analysis.sentences:
                sentence_analysis = sentence.morph_analysis
                for s_text, s_form, s_pos in zip(
                    sentence_analysis.text,
                    sentence_analysis.form,
                    sentence_analysis.partofspeech,
                ):
                    if s_text:
                        tokens.append(
                            (
                                sentence_id,
                                s_text,
                                s_form[0],
                                s_pos[0],
                                text.meta.get("texttype"),
                                file_name,
                            )
                        )  # In case of ambiguity, select the first or index 0
                sentence_id += 1

            # print(f"{file_name} tagged, now saving")

            # Salvestamine
            with open(os.path.join(csv_dir, csv_file_name), "w") as f:
                fieldnames = ["sentence_id", "word", "form", "pos", "type", "source"]
                writer = csv.writer(
                    f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                writer.writerow(fieldnames)
                for row in tokens:
                    writer.writerow(row)

            # print(f"{file_name} saved to {csv_file_name}\n")

        print("Morphological tagging completed successfully")

    @staticmethod
    def create_json_file_by_file_enc2017(
        jsons: typing.List[str | pathlib.Path],
        save_dir: str | pathlib.Path,
        do_morph_layer: bool = True,
        bert_morph_tagger: typing.Optional[BertMorphTagger] = None,
        necessary_layers: typing.List[str] = [
            "words",
            "sentences",
            "morph_analysis",
            "bert_morph_tagging",
        ],
        ignore_errors: bool = False,
        replace_files: bool = False,
    ):
        """
        Creates a new JSON file containing the morphological analysis from each JSON file.
        <ul>
            <li>Skips JSON files that have already been created.</li>
            <li>Converts JSON file into EstNLTK Text object.</li>
            <li>Adds text type metadata and morph analysis.</li>
            <li>Adds <code>BertMorphTagger</code> layer</li>
            <li>Removes unnecessary layers.</li>
            <li>Converts EstNLTK Text object into JSON using <code>estnltk.converters.text_to_json.</code></li>
        </ul>
        Args:
            jsons (list[str | pathlib.Path]): List of json filepaths from which to read in the text
            save_dir (str | pathlib.Path): Directory where to save the new json files
            bert_morph_tagger (optional, BertMorphTagger): Configured <code>BertMorphTagger</code> class instance, if None, will not use this tagger
            necessary_layers (optional, list[str]): Text object layers that will not be deleted
            ignore_errors (optional, bool): Ignores texts that give errors when tagging
            replace_files (optional, bool): Replaces files with new ones in the given directory
        """

        count_errors = 0

        print("Beginning to morphologically tag file by file")
        for file_path in tqdm(jsons):
            file_name = pathlib.Path(file_path).stem + pathlib.Path(file_path).suffix
            # Skipping previous JSON files
            if not replace_files and os.path.exists(os.path.join(save_dir, file_name)):
                continue

            # Convert json to EstNLTK Text object
            text = estnltk.converters.json_to_text(file=file_path)

            Preprocessor.find_text_type(
                text, file_name
            )  # Assign text type metadata if not already assigned

            # Add morph layer
            if do_morph_layer:
                text.tag_layer("morph_analysis")

            # Add BERT morph layer
            if isinstance(bert_morph_tagger, BertMorphTagger):
                if not do_morph_layer:
                    text.tag_layer("sentences")
                if not ignore_errors:
                    text.add_layer(bert_morph_tagger.make_layer(text))
                else:
                    try:
                        text.add_layer(bert_morph_tagger.make_layer(text))
                    except Exception as e:
                        count_errors += 1

            # Remove unnecessary layers
            for layer in text.layers:
                if layer not in necessary_layers:
                    text.pop_layer(layer, cascading=False)

            if (
                "morph_analysis" in text.layers and "bert_morph_tagging" in text.layers
            ):  # Assertion that the length of both layers are the same
                assert len(text.morph_analysis) == len(
                    text.bert_morph_tagging
                ), f"""Failed to assert file '{file_path}'
                Length of layers aren't the same:
                morph_analysis = {len(text.morph_analysis)}
                bert_morph_tagging = {len(text.bert_morph_tagging)}"""
            # Save to JSON
            os.makedirs(save_dir, exist_ok=True)
            estnltk.converters.text_to_json(
                text=text, file=os.path.join(save_dir, file_name)
            )

        print(
            "Morphological tagging completed successfully"
        ) if count_errors == 0 or ignore_errors else print(
            "Morphological tagging completed"
        )
        if count_errors > 0:
            print(f"Failed to tag {count_errors} texts")

    @staticmethod
    def create_df_enc2017(
        jsons: typing.List[str | pathlib.Path], output_filename: str | pathlib.Path
    ):
        """
        Creates a new dataset from the given JSON files and saves it as a CSV file. \n
        Assumes that the JSON files have already been morphologically tagged and contain the necessary layers. \n
        For each <code>.json</code> file, the following info is gathered:
        <ul>
            <li><code>sentence_id</code> -- given for each sentence</li>
            <li><code>words</code> -- words gathered from text</li>
            <li><code>form</code> -- word form notation</li>
            <li><code>pos</code> -- part of speech</li>
            <li><code>file_prefix</code> -- metadata</li>
            <li><code>source</code> -- file name where the text is taken from</li>
        </ul>
        <a href="https://github.com/Filosoft/vabamorf/blob/e6d42371006710175f7ec328c98f90b122930555/doc/tagset.md">Tables of morphological categories</a> for more information about <code>form</code> and <code>pos</code>.

        Args:
            jsons (list[str | pathlib.Path]): List of json filepaths from which to read in the text
            output_filename (str | pathlib.Path): Filename where to save the gathered text. Supports .csv and .parquet extensions. If the file already exists, it will be overwritten.
        """
        # Check that the output filename has a supported extension
        file_extension = pathlib.Path(output_filename).suffix.lower()
        if file_extension not in [".csv", ".parquet"]:
            raise ValueError(
                f"Unsupported file extension: {file_extension}. Supported extensions are .csv and .parquet."
            )
        # Column names for the output dataset
        fieldnames = ["sentence_id", "words", "form", "pos", "labels", "type", "source"]

        # If file exists, remove it (we overwrite by default)
        if os.path.exists(output_filename):
            os.remove(output_filename)

        print("Beginning to create dataset from JSON files.")

        def _rows_for_file(file_path: typing.Union[str, pathlib.Path]) -> list:
            """Collect rows for a single JSON file.

            Returns a list of tuples matching `fieldnames` or an empty list if the
            file should be skipped.
            """
            sentence_id_local = 0
            text = estnltk.converters.json_to_text(
                file=file_path
            )  # Convert json to EstNLTK Text object
            # Assign text type metadata if not already assigned
            file_name_local = (
                pathlib.Path(file_path).stem + pathlib.Path(file_path).suffix
            )
            Preprocessor.find_text_type(text, file_name_local)
            if "morph_analysis" not in text.layers:
                print(
                    f"Text from file '{file_path}' doesn't have morph_analysis layer, skipping this text."
                )
                return []
            # Iterate through sentences and gather the necessary info for each word
            rows_local: list[tuple] = []
            for sentence in text.sentences:
                sentence_analysis = sentence.morph_analysis
                for s_text, s_form, s_pos in zip(
                    sentence_analysis.text,
                    sentence_analysis.form,
                    sentence_analysis.partofspeech,
                ):
                    if s_text:
                        # In case of ambiguity, select the first or index 0 for form and pos tag, and create a label by combining them. If either form or pos tag is missing, use the available one as the label.
                        label = s_form[0] + "_" + s_pos[0]
                        if s_form[0] == "" and s_pos[0] != "":
                            label = s_pos[0]
                        elif s_form[0] != "" and s_pos[0] == "":
                            label = s_form[0]
                        rows_local.append(
                            (
                                sentence_id_local,
                                s_text,
                                s_form[0],
                                s_pos[0],
                                label,
                                text.meta.get("texttype"),
                                file_name_local,
                            )
                        )
                sentence_id_local += 1

            return rows_local

        # Save the gathered info into a new file with the given name and extension
        if file_extension == ".parquet":  # Parquet
            writer: typing.Optional[pq.ParquetWriter] = None
            for file_path in tqdm(jsons):
                rows = _rows_for_file(file_path)
                if not rows:
                    continue
                # Save in chunks to avoid memory issues with large datasets
                df_chunk = pd.DataFrame(rows, columns=fieldnames)
                table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_filename, table.schema)
                writer.write_table(table)

            if writer is not None:
                writer.close()

        else:  # CSV
            header_written = False
            for file_path in tqdm(jsons):
                rows = _rows_for_file(file_path)
                if not rows:
                    continue
                # Save in chunks to avoid memory issues with large datasets
                df_chunk = pd.DataFrame(rows, columns=fieldnames)
                df_chunk.to_csv(
                    output_filename,
                    mode="a",
                    header=not header_written,
                    index=False,
                    encoding="utf-8",
                )
                header_written = True

        print("Morphological tagging completed successfully")
        print(f"Tagged texts saved to {output_filename}\n")

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
    def stratified_sample_by_type(
        shuffled_df: pd.DataFrame,
        N: int,
        source_col: str = "source",
        type_col: str = "type",
        seed: int | None = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Stratified sample N rows from `shuffled_df` by selecting whole source blocks.
        - Preserves internal order of each source block.
        - Tries to sample roughly uniformly across text `type`s.
        - Overshoot is allowed (sources cannot be split).
        """
        rnd = random.Random(seed)

        sizes = shuffled_df.groupby(source_col, sort=False).size().rename("size")
        types = shuffled_df.groupby(source_col, sort=False)[type_col].first()
        sources_df = pd.concat(
            [sizes, types], axis=1
        ).reset_index()  # columns: source, size, type

        mapping: dict[str, list[tuple[str, int]]] = {}
        for _, row in sources_df.iterrows():
            t = row[type_col]
            s = row[source_col]
            sz = int(row["size"])
            mapping.setdefault(t, []).append((s, sz))

        for t in mapping:
            rnd.shuffle(mapping[t])

        types_list = list(mapping.keys())
        num_types = max(1, len(types_list))
        base_quota = N // num_types
        remainder = N % num_types

        selected_sources: list[str] = []
        accumulated_total = 0

        for i, t in enumerate(types_list):
            quota = base_quota + (1 if i < remainder else 0)
            lst = mapping[t]

            lst.sort(key=lambda x: x[1])
            srcs = [x[0] for x in lst]
            sizes_list = [x[1] for x in lst]

            acc = 0
            while sizes_list and acc < quota:
                rem = quota - acc
                idx_eq = bisect_left(sizes_list, rem)
                if idx_eq < len(sizes_list) and sizes_list[idx_eq] == rem:
                    pick_idx = idx_eq
                else:
                    idx_le = bisect_right(sizes_list, rem) - 1
                    if idx_le >= 0:
                        pick_idx = idx_le
                    else:
                        pick_idx = 0

                src = srcs.pop(pick_idx)
                sz = sizes_list.pop(pick_idx)
                selected_sources.append(src)
                acc += sz
                accumulated_total += sz

            print(
                f"Type '{t}': quota={quota}, selected={acc}, overshoot={acc - quota}"
            ) if verbose else None
            mapping[t] = list(zip(srcs, sizes_list))

        if accumulated_total < N:
            print(
                f"Accumulated total {accumulated_total} is less than N={N}, selecting more sources from remaining pool."
            ) if verbose else None
            remaining = []
            for lst in mapping.values():
                remaining.extend(lst)
            remaining.sort(key=lambda x: x[1], reverse=True)
            idx = 0
            while accumulated_total < N and idx < len(remaining):
                src, sz = remaining[idx]
                selected_sources.append(src)
                accumulated_total += sz
                idx += 1

        groups = shuffled_df.groupby(source_col, sort=False).groups
        positions = [groups[s] for s in selected_sources if s in groups]
        if not positions:
            return pd.DataFrame(columns=shuffled_df.columns)

        ordered_pos = np.concatenate(positions)
        return shuffled_df.take(ordered_pos).reset_index(drop=True)

    @staticmethod
    def split_by_source(
        df: pd.DataFrame,
        test_size: float = 0.2,
        source_col: str = "source",
        stratify_col: str | None = "type",
        seed: int | None = None,
    ):
        """
        Split a DataFrame into train and test sets by source, ensuring that all rows from the same source are in the same set.
        Optionally stratifies the split by a specified column (e.g. text type) to maintain a similar distribution of that column in both sets.

        Args:
            df (pd.DataFrame): Input DataFrame to split.
            test_size (float, optional): Proportion of test data. Defaults to 0.2.
            source_col (str, optional): Column name for source identifiers. Defaults to "source".
            stratify_col (str | None, optional): Column name for stratification values (e.g. text type). Defaults to "type".
            seed (int | None, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, list, list]: (train_df, test_df, train_sources, test_sources)
             - train_df: DataFrame containing training data.
             - test_df: DataFrame containing test data.
             - train_sources: List of unique source identifiers in the training set.
             - test_sources: List of unique source identifiers in the test set.
        """
        groups = df.groupby(source_col, sort=False).groups  # source -> Int64Index
        sources = np.array(list(groups.keys()))

        # optional per-source stratify values (one value per source)
        stratify_vals = None
        if stratify_col is not None:
            # take the type of the first row of each source (all rows in a source share the same type)
            stratify_vals = np.array(
                [df.iloc[groups[s][0]][stratify_col] for s in sources]
            )

        train_src, test_src = train_test_split(
            sources, test_size=test_size, random_state=seed, stratify=stratify_vals
        )

        # Preserve the original source-block ordering by sorting by first-row position
        def preserve_order(src_list):
            order = np.argsort([groups[s][0] for s in src_list])
            return src_list[order]

        train_src = preserve_order(train_src)
        test_src = preserve_order(test_src)

        # Efficiently build row positions and slice (avoids per-row operations)
        train_pos = (
            np.concatenate([groups[s] for s in train_src])
            if len(train_src) > 0
            else np.array([], dtype=int)
        )
        test_pos = (
            np.concatenate([groups[s] for s in test_src])
            if len(test_src) > 0
            else np.array([], dtype=int)
        )

        train_df = df.take(train_pos).reset_index(drop=True)
        test_df = df.take(test_pos).reset_index(drop=True)

        return train_df, test_df, list(train_src), list(test_src)
