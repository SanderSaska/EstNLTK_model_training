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

from scripts.model.bert_morph_tagger import BertMorphTagger


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
                for text, form, pos in zip(
                    sentence_analysis.text,
                    sentence_analysis.form,
                    sentence_analysis.partofspeech,
                ):
                    if text:
                        tokens.append(
                            (
                                sentence_id,
                                text,
                                form[0],
                                pos[0],
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

        sentence_id = 0
        fieldnames = ["sentence_id", "words", "form", "pos", "type", "source"]

        # If file exists, remove it (we overwrite by default)
        if os.path.exists(output_filename):
            os.remove(output_filename)

        print("Beginning to create dataset from JSON files.")

        if file_extension == ".parquet":  # Parquet
            writer: typing.Optional[pq.ParquetWriter] = None
            for file_path in tqdm(jsons):
                sentence_id = 0
                text = estnltk.converters.json_to_text(file=file_path)
                file_name = (
                    pathlib.Path(file_path).stem + pathlib.Path(file_path).suffix
                )
                Preprocessor.find_text_type(text, file_name)
                if "morph_analysis" not in text.layers:
                    print(
                        f"Text from file '{file_path}' doesn't have morph_analysis layer, skipping this text."
                    )
                    continue

                rows: list[tuple] = []
                for sentence in text.sentences:
                    sentence_analysis = sentence.morph_analysis
                    for s_text, s_form, s_pos in zip(
                        sentence_analysis.text,
                        sentence_analysis.form,
                        sentence_analysis.partofspeech,
                    ):
                        if s_text:
                            rows.append(
                                (
                                    sentence_id,
                                    s_text,
                                    s_form[0],
                                    s_pos[0],
                                    text.meta.get("texttype"),
                                    file_name,
                                )
                            )
                    sentence_id += 1

                if not rows:
                    continue

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
                sentence_id = 0
                text = estnltk.converters.json_to_text(file=file_path)
                file_name = (
                    pathlib.Path(file_path).stem + pathlib.Path(file_path).suffix
                )
                Preprocessor.find_text_type(text, file_name)
                if "morph_analysis" not in text.layers:
                    print(
                        f"Text from file '{file_path}' doesn't have morph_analysis layer, skipping this text."
                    )
                    continue

                rows: list[tuple] = []
                for sentence in text.sentences:
                    sentence_analysis = sentence.morph_analysis
                    for s_text, s_form, s_pos in zip(
                        sentence_analysis.text,
                        sentence_analysis.form,
                        sentence_analysis.partofspeech,
                    ):
                        if s_text:
                            rows.append(
                                (
                                    sentence_id,
                                    s_text,
                                    s_form[0],
                                    s_pos[0],
                                    text.meta.get("texttype"),
                                    file_name,
                                )
                            )
                    sentence_id += 1

                if not rows:
                    continue

                df_chunk = pd.DataFrame(rows, columns=fieldnames)
                # append to csv; write header only once
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
