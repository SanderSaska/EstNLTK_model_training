import os
import typing
import tqdm
import estnltk
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pathlib

from scripts.model.bert_morph_tagger import BertMorphTagger


class Preprocessor:
    """
    Preprocessor for the Universal Dependencies Estonian Dependency Treebank (UD_EST-EDT).
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
            if file_name.startswith("aja"):
                text.meta.update({"texttype": "periodicals"})
            elif file_name.startswith("ilu") or file_name.startswith("arborest"):
                text.meta.update({"texttype": "fiction"})
            elif file_name.startswith("tea"):
                text.meta.update({"texttype": "science"})
            else:
                print("Unknown text type. Assigning text type 'unknown'.")
                text.meta.update({"texttype": "unknown"})
        return

    @staticmethod
    def create_json_file_by_file_est_ud(
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
        for file_path in tqdm.tqdm(jsons):
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
    def create_df_ud_corpus(
        jsons: typing.List[str | pathlib.Path],
        tokenizer: str,
        output_filename: typing.Union[str, pathlib.Path],
    ):
        """
        Creates a new dataset from converted the Estonian UD EDT <a href="https://github.com/UniversalDependencies/UD_Estonian-EDT">corpus</a> JSON files. \n
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
            jsons (list[str | pathlib.Path]): List of json files from which to read in the text
            tokenizer (str): Use goldstandard (<code>ud_morph_reduced</code>) or Vabamorf tokenization ((<code>morph_analysis</code>))
            output_filename (str | pathlib.Path): Filename where to save the gathered text. Supports .csv and .parquet extensions. If the file already exists, it will be overwritten.
        """
        # Check that the tokenizer argument is valid
        if tokenizer not in {"ud_morph_reduced", "morph_analysis"}:
            raise ValueError(
                "create_df_ud_corpus: tokenizer must be one of %r."
                % {"ud_morph_reduced", "morph_analysis"}
            )

        # Check that the output filename has a supported extension
        file_extension = pathlib.Path(output_filename).suffix.lower()
        if file_extension not in [".csv", ".parquet"]:
            raise ValueError(
                f"Unsupported file extension: {file_extension}. Supported extensions are .csv and .parquet."
            )

        # If file exists, remove it (we overwrite by default)
        if os.path.exists(output_filename):
            os.remove(output_filename)

        sentence_id = 0
        fieldnames = ["sentence_id", "words", "form", "pos", "file_prefix", "source"]

        print("Beginning to morphologically tag file by file. This can take a while.")

        def _extract_rows(
            text_obj: estnltk.Text,
            tokenizer_value: str,
            file_prefix_value,
            file_name_value,
        ) -> list:
            """Extract rows (sentence_id, word, form, pos, file_prefix, source) from an EstNLTK Text.

            Returns a list of tuples. Sentence ids start at 0 for each text and increment per sentence.
            """
            rows_local: list[tuple] = []
            sentence_id_local = 0
            for sentence in text_obj.sentences:
                if tokenizer_value == "ud_morph_reduced":
                    sentence_analysis = sentence.ud_morph_reduced
                    iter_triplet = zip(
                        sentence_analysis.text,
                        sentence_analysis.form,
                        sentence_analysis.pos,
                    )
                else:
                    sentence_analysis = sentence.morph_analysis
                    iter_triplet = zip(
                        sentence_analysis.text,
                        sentence_analysis.form,
                        sentence_analysis.partofspeech,
                    )

                for s_text, s_form, s_pos in iter_triplet:
                    if s_text:
                        rows_local.append(
                            (
                                sentence_id_local,
                                s_text,
                                s_form[0],
                                s_pos[0],
                                file_prefix_value,
                                file_name_value,
                            )
                        )
                sentence_id_local += 1

            return rows_local

        task_progress = tqdm.tqdm(
            jsons, desc="Processing files", unit="file", total=len(jsons)
        )

        if file_extension == ".parquet":
            writer: typing.Optional[pq.ParquetWriter] = None
            for file_path in task_progress:
                text = estnltk.converters.json_to_text(file=file_path)
                if tokenizer == "morph_analysis":
                    text.tag_layer("morph_analysis")
                file_prefix = text.meta.get("file_prefix")
                file_name = pathlib.Path(file_path).name

                rows: list[tuple] = _extract_rows(
                    text, tokenizer, file_prefix, file_name
                )

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
            for file_path in task_progress:
                text = estnltk.converters.json_to_text(file=file_path)
                if tokenizer == "morph_analysis":
                    text.tag_layer("morph_analysis")
                file_prefix = text.meta.get("file_prefix")
                file_name = pathlib.Path(file_path).name
                rows: list[tuple] = _extract_rows(
                    text, tokenizer, file_prefix, file_name
                )

                if not rows:
                    continue

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
