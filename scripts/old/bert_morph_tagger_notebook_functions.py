# Module imports

import os
import re
import csv
import typing
import itertools
import json
import logging
import warnings
import IPython.display
import types
import pandas as pd
import sklearn as sk
import numpy as np
import estnltk, estnltk.converters, estnltk.taggers
import torch
import math
import IPython
import gc

from tqdm import tqdm
from simpletransformers.ner import NERModel, NERArgs
from scripts.models.bert_morph_tagger import BertMorphTagger
from scripts.models.morph_eval_utils import (
    MorphDiffSummarizer,
    MorphDiffFinder,
    write_formatted_diff_str_to_file,
)
from scripts.preprocessing.ud_et_edt.est_ud_utils import (
    load_ud_file_texts_with_corrections,
    load_ud_file_with_corrections,
)
from scripts.preprocessing.ud_et_edt.est_ud_morph_conv import (
    convert_ud_layer_to_reduced_morph_layer,
)


class NotebookFunctions:
    ################################################################
    # Notebook 1
    ################################################################

    @staticmethod
    def create_csv_file_by_file_enc2017(
        jsons: typing.List[str], in_dir: str, csv_dir: str
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
            jsons (list): List of json files from which to read in the text
            in_dir (str): Directory where to read the json files from
            csv_dir (str): Directory where to save the new csv files
        """
        print("Beginning to morphologically tag file by file")
        for file_name in tqdm(jsons):
            tokens = list()
            sentence_id = 0

            # Skipping previous CSV files
            csv_file_name = file_name[:-4] + "csv"
            if os.path.exists(os.path.join(csv_dir, csv_file_name)):
                # print(f"Skipping {file_name} as {csv_file_name} already exists.")
                continue

            # print(f"Beginning to tag {file_name}")

            # Morph. tagging using estnltk
            text = estnltk.converters.json_to_text(file=os.path.join(in_dir, file_name))
            text_type = text.meta.get("texttype")  # Text type
            if file_name.startswith("wiki17"):
                text_type = "wikipedia"
            elif file_name.startswith("web13"):
                text_type = "blogs_and_forums"
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
                            (sentence_id, text, form[0], pos[0], text_type, file_name)
                        )  # In case of multiplicity, select the first or index 0
                sentence_id += 1

            # print(f"{file_name} tagged, now saving")

            # Salvestamine
            with open(os.path.join(csv_dir, csv_file_name), "w") as f:
                fieldnames = ["sentence_id", "word", "form", "pos"]
                writer = csv.writer(
                    f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                writer.writerow(fieldnames)
                for row in tokens:
                    writer.writerow(row)

            # print(f"{file_name} saved to {csv_file_name}\n")

        print("Morphological tagging completed successfully")

    @staticmethod
    def create_df_enc2017_csv(jsons: typing.List[str], in_dir: str, csv_filename: str):
        """
        Creates a new dataset from converted the Estonian UD EDT <a href="https://github.com/UniversalDependencies/UD_Estonian-EDT">corpus</a>. \n
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
            jsons (list): List of json files from which to read in the text
            in_dir (str): Directory containing list of files (<code>jsons</code>)
            csv_filename (str): CSV filename where to save the gathered text
        """
        tokens = list()
        sentence_id = 0
        fieldnames = ["sentence_id", "words", "form", "pos", "type", "source"]

        print("Beginning to morphologically tag file by file. This can take a while.")
        for file_name in tqdm(jsons):
            # print(f"Beginning to tag {file_name}")
            sentence_id = 0

            # Morph. tagging
            text = estnltk.converters.json_to_text(file=os.path.join(in_dir, file_name))
            text_type = text.meta.get("texttype")  # Text type
            if file_name.startswith("wiki17"):
                text_type = "wikipedia"
            elif file_name.startswith("web13"):
                text_type = "blogs_and_forums"
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
                            (sentence_id, text, form[0], pos[0], text_type, file_name)
                        )  # In case of multiplicity, select the first or index 0
                sentence_id += 1
            # print(f"{file_name} tagged")

        print("Morphological tagging completed successfully")
        print("Creating Pandas dataframe")
        df = pd.DataFrame(data=tokens, columns=fieldnames)
        df.to_csv(path_or_buf=csv_filename, index=False)
        print(f"Tagged texts saved to {csv_filename}\n")

    @staticmethod
    def clean_df(df: pd.DataFrame, df_file_name: typing.Optional[str] = None):
        """Finishes dataframe by:
        <ul>
            <li>filling NaN values in columns <code>form</code> and <code>pos</code> with empty strings;</li>
            <li>removing NaN words.</li>
        </ul>

        Args:
            df (pandas.DataFrame): Pandas dataframe to clean
            df_file_name (str): CSV file name from which dataframe was created
        """
        print("Assigning NaN values in columns form and pos with an empty string")
        # NaN values are assigned with an empty string
        df["form"] = df["form"].fillna("")
        df["pos"] = df["pos"].fillna("")
        print("Removing NaN words")
        # Removing NaN words
        df.dropna(subset=["words"], inplace=True)
        if df_file_name:
            df.to_csv(path_or_buf=df_file_name, index=False)
            print(f"Modified dataframe saved to {df_file_name}")
        else:
            print("Dataframe cleaned")

    @staticmethod
    def create_labels_column(
        df: pd.DataFrame, df_file_name: typing.Optional[str] = None
    ):
        """
        Creates a new column <code>labels</code> concatenating the values of
        columns <code>pos</code> (part of speech) and <code>form</code> (word form notation)

        Args:
            df (pd.DataFrame): Pandas dataframe to create a new column
            df_file_name (str): CSV file name where to save the dataframe
        """
        print("Creating column 'labels'")
        df["labels"] = df.apply(
            lambda row: (
                str(row["form"]) + "_" + str(row["pos"])
                if row["form"] and row["pos"]
                else str(row["form"]) or str(row["pos"])
            ),
            axis=1,
        )
        print("Column 'labels' created")
        if df_file_name:
            df.to_csv(path_or_buf=df_file_name, index=False)
            print(f"Modified dataframe saved to {df_file_name}")

    @staticmethod
    def get_unique_labels(json_file: typing.Optional[str] = None):
        """Reads from JSON file or
        creates list of unique labels that the model must predict
        by creating all possible combinations of POS (Part Of Speech) and form.

        <i>Gathering unique labels from the enc2017 database proved to be insufficient for future model evaluation,
        because the database does not contain all possible combinations of POS and form.
        Evaluating model with UD Est-EDT test corpus proved that this problem existed.</i>

        Args:
            json_file (optional, str): Path to JSON file containing all unique labels

        Returns:
            list: List of unique labels
        """

        if json_file:
            try:
                with open(file=json_file, mode="r") as f:
                    unique_labels = json.load(f)
                return unique_labels
            except:
                print("Could not read JSON file. Creating unique labels list.")

        # Separately, if one of two doesn't exist
        pos_labels = [
            "A",
            "C",
            "D",
            "G",
            "H",
            "I",
            "J",
            "K",
            "N",
            "O",
            "P",
            "S",
            "U",
            "V",
            "X",
            "Y",
            "Z",
        ]
        form_labels = [
            "ab",
            "abl",
            "ad",
            "adt",
            "all",
            "el",
            "es",
            "g",
            "ill",
            "in",
            "kom",
            "n",
            "p",
            "pl",
            "sg",
            "ter",
            "tr",
            "b",
            "d",
            "da",
            "des",
            "ge",
            "gem",
            "gu",
            "ks",
            "ksid",
            "ksime",
            "ksin",
            "ksite",
            "ma",
            "maks",
            "mas",
            "mast",
            "mata",
            "me",
            "n",
            "neg",
            "nud",
            "nuks",
            "nuksid",
            "nuksime",
            "nuksin",
            "nuksite",
            "nuvat",
            "o",
            "s",
            "sid",
            "sime",
            "sin",
            "site",
            "ta",
            "tagu",
            "taks",
            "takse",
            "tama",
            "tav",
            "tavat",
            "te",
            "ti",
            "tud",
            "tuks",
            "tuvat",
            "v",
            "vad",
            "vat",
        ]

        pos_labels_mutable = ["A", "C", "N", "H", "O", "P", "S", "U", "Y", "X"]
        # pos_labels_immutable = ['D', 'G', 'I', 'J', 'K', 'X', 'Y', 'Z']
        pos_label_verb = ["V"]
        form_labels_noun = [
            "ab",
            "abl",
            "ad",
            "adt",
            "all",
            "el",
            "es",
            "g",
            "ill",
            "in",
            "kom",
            "n",
            "p",
            "ter",
            "tr",
        ]
        form_labels_noun_count = ["pl", "sg"]
        form_labels_verb = [
            "b",
            "d",
            "da",
            "des",
            "ge",
            "gem",
            "gu",
            "ks",
            "ksid",
            "ksime",
            "ksin",
            "ksite",
            "ma",
            "maks",
            "mas",
            "mast",
            "mata",
            "me",
            "n",
            "neg",
            "neg ge",
            "neg gem",
            "neg gu",
            "neg ks",
            "neg me",
            "neg nud",
            "neg nuks",
            "neg o",
            "neg vat",
            "neg tud",
            "nud",
            "nuks",
            "nuksid",
            "nuksime",
            "nuksin",
            "nuksite",
            "nuvat",
            "o",
            "s",
            "sid",
            "sime",
            "sin",
            "site",
            "ta",
            "tagu",
            "taks",
            "takse",
            "tama",
            "tav",
            "tavat",
            "te",
            "ti",
            "tud",
            "tuks",
            "tuvat",
            "v",
            "vad",
            "vat",
        ]

        noun_labels_without_pos = list(
            itertools.product(form_labels_noun_count, form_labels_noun)
        )
        noun_labels_nested = list(
            itertools.product(noun_labels_without_pos, pos_labels_mutable)
        )
        form_pos_labels = list(itertools.product(form_labels, pos_labels))
        noun_labels = list()
        verb_labels = list(itertools.product(form_labels_verb, pos_label_verb))

        # Connect count and form in mutables
        for form, pos in noun_labels_nested:
            noun_labels.append((form[0] + " " + form[1], pos))

        pos_label_only = [("", pos) for pos in pos_labels]
        form_label_only = [(form, "") for form in form_labels]
        unknown_form_labels = [
            ("?", pos) for pos in pos_labels
        ]  # form '?' comes from enc2017 corpus after tagging

        unique_labels = (
            pos_label_only
            + form_label_only
            + noun_labels
            + verb_labels
            + unknown_form_labels
            + form_pos_labels
            + ["?"]
        )  # '?' for labels unknown to Vabamorf

        unique_labels_df = pd.DataFrame(unique_labels, columns=["form", "pos"])
        NotebookFunctions.create_labels_column(unique_labels_df)
        unique_labels_df.drop(labels=["form", "pos"], axis=1)
        print("List of unique labels created")
        return unique_labels_df["labels"].tolist()

    @staticmethod
    def gather_rows_for_text_type(
        df: pd.DataFrame, n: int, random_state: typing.Optional[int] = None
    ):
        """Gathers about `n` (>= n) rows for each text type\n
        Ensures that all text types have about the same number of words.

        Args:
            df (pd.DataFrame): The DataFrame containing the text data.
            n (int): Number of words to gather.
            random_state (optional, int): Seed for the shuffle function (acts like <code>random_state</code>).

        Returns:
            pd.DataFrame: Gathered rows for each text type.
        """

        def gather_rows_for_type(
            group: pd.DataFrame, n: int, random_state: typing.Optional[int] = None
        ):
            """Gathers about `n` (>= n) rows for the text type\n

            Args:
                group (): Pandas group's dataframe
                n (int): Number of words to gather
                random_state (int): Seed for the shuffle function (acts like <code>random_state</code>)

            Returns:
                pd.DataFrame: Gathered rows for text type
            """
            gathered_rows = pd.DataFrame()
            sources = group["source"].unique()

            if random_state:
                np.random.seed(random_state)
                np.random.shuffle(sources)

            for source in sources:
                source_rows = group[group["source"] == source]
                gathered_rows = pd.concat([gathered_rows, source_rows])
                if len(gathered_rows) >= n:
                    break

            return gathered_rows

        grouped = df.groupby("type")
        data = pd.concat(
            [gather_rows_for_type(group, n, random_state) for _, group in grouped]
        )
        return data

    @staticmethod
    def predict_tags(model, sentences):
        """Predicts tags to a sentence or number of sentences

        Args:
            model: Model to use to predict tags.
            sentences (List[str] or str): List of sentences strings or a sentence string

        Raises:
            TypeError: Variable <code>sentences</code> is not List[str] or str

        Returns:
            (predictions, raw_outputs): Predictions and model's raw output
        """

        if isinstance(sentences, str):
            text = estnltk.Text(sentences)
            text.tag_layer("morph_analysis")
            text.morph_analysis
            sentences = [s.text for s in text.sentences]
            predictions, raw_outputs = model.predict(sentences, split_on_space=False)

        elif isinstance(sentences, list) and isinstance(sentences[0], list):
            predictions, raw_outputs = model.predict(sentences, split_on_space=False)

        else:
            raise TypeError(
                f"Input is in wrong format. Possible formats are str or list of lists. Your input is {type(sentences)}"
            )

        return predictions, raw_outputs

    @staticmethod
    def get_top_n_tags(raw_outputs, tag_list, n=3, with_confidence=True):
        """Extract the top <code>n</code> tags and their probabilities for each word based on the raw output logits.

        Args:
            raw_outputs (list): Raw prediction logits from the model.
            tag_list (list): List of all possible tags.
            n (int): Number of top tags to extract.

        Returns:
            list: A list of lists, where each sublist contains tuples of the top `n` tags and their probabilities for a word.
        """
        top_n_tags = []

        for sentence_logits in raw_outputs:
            for word_logits in sentence_logits:
                # Get the logits for the word
                logits = list(word_logits.values())[0][0]

                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(
                    torch.tensor(logits), dim=-1
                ).numpy()

                # Get the indices of the top `n` probabilities
                top_n_indices = np.argsort(probabilities)[-n:][::-1]

                # Map these indices to the actual tags and their probabilities
                if with_confidence:
                    top_n_tags.append(
                        [(tag_list[i], probabilities[i]) for i in top_n_indices]
                    )
                else:
                    top_n_tags.append([(tag_list[i]) for i in top_n_indices])

        return top_n_tags

    @staticmethod
    def predict_top_n_tags(model, to_predict, with_confidence=True):
        """Predicts top <code>n</code> tags and their probabilities for each word.

        Args:
            to_predict (list): A python list of text (str) to be sent to the model for prediction.
            Must be a a list of lists, with the inner list being a list of strings consisting of the split sequences.
            The outer list is the list of sequences to predict on.
            with_confidence (bool): Whether to output confidence values for each tag.
        """
        _, raw_outputs = model.predict(to_predict, split_on_space=False)

        tag_list = (
            model.config.id2label
        )  # Assuming the model has a mapping from ids to labels
        top_n_tags_per_word = NotebookFunctions.get_top_n_tags(
            raw_outputs, tag_list, n=3, with_confidence=with_confidence
        )

        for sentence in to_predict:
            for word, tags in zip(sentence, top_n_tags_per_word):
                print(f"Word: {word}")
                if with_confidence:
                    for tag, confidence in tags:
                        print(f"\tTag: {tag} \t Confidence: {confidence:.4f}")
                else:
                    for tag in tags:
                        print(f"\tTag: {tag}")

    @staticmethod
    def check_token_count(model, sentence):
        """Checks token count in the sentence

        Args:
            model (): NER model that is predicted with
            sentence (list): list of words in a sentence

        Returns:
            bool: Whether the token count exceeds model's maximum sequence length
        """

        inputs = model.tokenizer("  ".join(sentence), return_tensors="pt")
        return bool(len(inputs["input_ids"][0]) >= model.args.max_seq_length)

    @staticmethod
    def get_clause_parts(model, clause):
        """Splits clause into equal length clause segments. <i>Clauses can get long when there is a list in a sentence</i>

        Args:
            model (): NER model that is predicted with
            clause (list): list of words in a clause

        Returns:
            list: List of clause segments. Each segment is a list of words.
        """

        inputs = model.tokenizer("  ".join(clause), return_tensors="pt")
        clause_parts = np.array_split(
            clause, math.ceil(len(inputs["input_ids"][0]) / model.args.max_seq_length)
        )
        return clause_parts

    @staticmethod
    def get_clauses(model, sentence):
        """Splits sentence into clauses using EstNLTK clauses layer.

        Args:
            model (): NER model that is predicted with
            sentence (list): list of words in a sentence

        Returns:
            list: List of clauses. Each clause is a list of words.
        """

        sentence_text = "  ".join(sentence)
        text = estnltk.Text(sentence_text)
        text.tag_layer("clauses")
        clauses = list()
        for clause in text.clauses:
            if NotebookFunctions.check_token_count(model, clause.text):
                clause_parts = NotebookFunctions.get_clause_parts(model, clause.text)
                clauses.extend(clause_parts)
            else:
                clauses.append(clause.text)
        return clauses

    @staticmethod
    def get_clauses_labels(model, sentence):
        """Predicts labels to clauses

        Args:
            model (): NER model that is predicted with
            sentence (list): list of words in a sentence

        Returns:
            list: List of tags predicted to words in clauses
        """

        clauses = NotebookFunctions.get_clauses(model, sentence)

        ner_labels_parts = list()

        # Predict tags
        predictions, _ = model.predict(clauses, split_on_space=False)
        for prediction_part in predictions:
            # ner_words_part = [list(p.keys())[0] for p in prediction_part]
            ner_labels_part = [list(p.values())[0] for p in prediction_part]

            # ner_words_parts.append(ner_words_part)
            ner_labels_parts.append(ner_labels_part)

        # ner_words = list(itertools.chain.from_iterable(ner_words_parts))
        ner_labels = list(itertools.chain.from_iterable(ner_labels_parts))

        return ner_labels

    @staticmethod
    def get_sentence_labels(model, sentence):
        """Predicts labels to a sentence

        Args:
            model (): NER model that is predicted with
            sentence (list): list of words in a sentence

        Returns:
            list: List of tags predicted to words in a sentence
        """

        # Predict tags
        predictions, _ = model.predict([sentence], split_on_space=False)

        # ner_words = [list(p.keys())[0] for p in predictions[0]]
        ner_labels = [list(p.values())[0] for p in predictions[0]]
        return ner_labels

    @staticmethod
    def process_groups(model, groups):
        """Predicts labels to a list of groups. This group contains sentences for each source text file.

        Args:
            model (): NER model that is predicted with
            groups (): Group containing sentences for each source text file

        Raises:
            AssertionError: When the length of predicted labels mismatch the length of the sentence length.
            <i>This might mean that generated clauses are wrong (meaning some words are missing because of <code>'  '.join(sentence)</code>)</i>

        Returns:
            list: List of predicted labels for each sentence in each source text file
        """
        chunk_results = []

        for _, group in groups:
            sentence = group.words.tolist()

            if NotebookFunctions.check_token_count(
                model, sentence
            ):  # Sentence splitting if token count is above model's max sequence length
                ner_labels = NotebookFunctions.get_clauses_labels(model, sentence)
            else:
                ner_labels = NotebookFunctions.get_sentence_labels(model, sentence)

            if len(ner_labels) != len(sentence):
                IPython.display.display(group)
                print(len(sentence), len(ner_labels))
                raise AssertionError(
                    "Predicted labels length mismatch sentence length."
                )

            group["ner_labels"] = ner_labels
            chunk_results.append(group)

        return chunk_results

    ################################################################
    # Notebook 2
    ################################################################

    @staticmethod
    def find_no_xpostag_rows():
        """
        Finds rows in the Estonian UD EDT treebank that contain rows where
        <code>xpostag == '_'</code>

        <i>In file <code>est_ud_utils.py</code> class <code>EstUDCorrectionsRewriter</code> has function <code>rewrite</code>, which has comment: \n
        #72: If <code>xpostag == '_'</code>, then add it based on upostag \n
        But not all xpostag conditions exist in the code as convertion throws an <code>AssertionError</code>.</i>
        """
        no_xpostag_regex = r"^\d+\t\S+\t\S+\t\S+\t_"
        conllu_dir = "UD_Estonian-EDT-r2.14"
        conllu_files = [
            "et_edt-ud-dev.conllu",
            "et_edt-ud-test.conllu",
            "et_edt-ud-train.conllu",
        ]
        for c_file in conllu_files:
            print("\n", c_file, "\n")
            with open(file=os.path.join(conllu_dir, c_file), mode="r") as f:
                text = f.read()
                # Find all matches
                matches = re.findall(no_xpostag_regex, text, re.MULTILINE)

                # Print the matching rows
                for match in matches:
                    print(match)

    @staticmethod
    def convert_ud_to_vabamorf(ud_corpus_dir, output_dir):
        """Converts Universal Dependencies' (UD) corpus to Vabamorf format

        Args:
            ud_corpus_dir (str): path to directory containing UD corpus .conllu files
            output_dir (str): path to directory, where Vabamorf jsons files will be written
        """
        # Create directory if it doesn't exist
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        assert os.path.isdir(output_dir)

        # Load UD corpus' files as EstNLTK Text objects
        loaded_texts = []
        ud_layer_name = "ud_syntax"
        for fname in os.listdir(ud_corpus_dir):
            # if 'train' in fname:
            #    continue
            # if 'dev' in fname:
            #    continue
            # if 'test' in fname:
            #    continue
            if fname.endswith(".conllu"):
                fpath = os.path.join(ud_corpus_dir, fname)
                texts = load_ud_file_texts_with_corrections(fpath, ud_layer_name)
                for text in texts:
                    text.meta["file"] = fname
                    loaded_texts.append(text)

        # Convert UD's morphosyntactic annotations to Vabamorf-like annotations
        for tid, text in enumerate(loaded_texts):
            convert_ud_layer_to_reduced_morph_layer(
                text, "ud_syntax", "ud_morph_reduced", add_layer=True
            )
            fname = text.meta["file"].replace(
                ".conllu", "_" + ("{:03d}".format(tid)) + ".json"
            )
            fpath = os.path.join(output_dir, fname)
            estnltk.converters.text_to_json(text, file=fpath)

    @staticmethod
    def create_df_ud_corpus(jsons, in_dir, tokenizer, csv_filename):
        """
        Creates a new dataset from converted the Estonian UD EDT <a href="https://github.com/UniversalDependencies/UD_Estonian-EDT">corpus</a>. \n
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
            jsons (list[str]): List of json files from which to read in the text
            in_dir (str): Directory containing list of files (<code>jsons</code>)
            tokenizer (str): Use goldstandard (<code>ud_morph_reduced</code>) or Vabamorf tokenization ((<code>morph_analysis</code>))
            csv_filename (str): CSV filename where to save the gathered text
        """
        if tokenizer not in {"ud_morph_reduced", "morph_analysis"}:
            raise ValueError(
                "create_df_ud_corpus: tokenizer must be one of %r."
                % {"ud_morph_reduced", "morph_analysis"}
            )

        tokens = list()
        sentence_id = 0
        fieldnames = ["sentence_id", "words", "form", "pos", "file_prefix", "source"]

        print("Beginning to morphologically tag file by file. This can take a while.")
        for file_name in jsons:
            # print(f"Beginning to tokenize {file_name}")
            sentence_id = 0

            # Morph. tagging
            text = estnltk.converters.json_to_text(file=os.path.join(in_dir, file_name))
            if tokenizer == "morph_analysis":
                text.tag_layer("morph_analysis")
            file_prefix = text.meta.get("file_prefix")
            for sentence in text.sentences:
                if tokenizer == "ud_morph_reduced":
                    sentence_analysis = sentence.ud_morph_reduced
                    for text, form, pos in zip(
                        sentence_analysis.text,
                        sentence_analysis.form,
                        sentence_analysis.pos,
                    ):
                        if text:
                            tokens.append(
                                (
                                    sentence_id,
                                    text,
                                    form[0],
                                    pos[0],
                                    file_prefix,
                                    file_name,
                                )
                            )  # In case of multiplicity, select the first or index 0
                else:
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
                                    file_prefix,
                                    file_name,
                                )
                            )  # In case of multiplicity, select the first or index 0
                sentence_id += 1
            # print(f"{file_name} tokenized")

        print("Morphological tagging completed successfully")
        print("Creating Pandas dataframe")
        df = pd.DataFrame(data=tokens, columns=fieldnames)
        df.to_csv(path_or_buf=csv_filename, index=False)
        print(f"Tagged texts saved to {csv_filename}\n")

    @staticmethod
    def unknown_labels(unique_labels_list, data):
        """Finds labels that are not present in the unique labels list.

        Args:
            unique_labels_list (list): list of unique labels (obtained from reading <code>unique_labels.json</code> file)
            data (pandas.core.frame.DataFrame): data to check for labels
        """
        df_unique_labels = pd.DataFrame(unique_labels_list, columns=["labels"])
        unique_labels_series = df_unique_labels["labels"]
        df_labels = data["labels"]  # .drop_duplicates()
        labels_not_in_unique = df_labels[~df_labels.isin(unique_labels_series)]

        print("Labels in data that are not in unique labels list:")
        print(labels_not_in_unique)
        print("Unique:")
        print(labels_not_in_unique.unique())
        return labels_not_in_unique

    @staticmethod
    def initialize_model(
        model_name, unique_labels, no_progress_bars=False, cleanup=True
    ):
        """Intializes the CamemBERT model with specified name and all possible labels list.

        Args:
            model_name (str): Default Transformer model name or path to a directory containing Transformer model file
            unique_labels (List[str]): List of all possible unique labels model can predict
            no_progress_bars (bool, optional): Whether to output progress bars. Defaults to False.
            cleanup (bool, optional): Will clean up any existing model before reinitialization.

        Returns:
            simpletransformers.ner.NERModel: Initialized model
        """
        if cleanup and "model" in globals():
            del globals()["model"]
        if cleanup and "model" in locals():
            del locals()["model"]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Set up logging
        logger = logging.getLogger("simpletransformers.ner.ner_model")
        logger.setLevel(logging.ERROR)

        # Suppress specific warnings
        # warnings.filterwarnings("ignore", category=FutureWarning) # For warning message "FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated."
        warnings.filterwarnings(
            "ignore", category=UserWarning
        )  # For warnings like "UserWarning: <tag> seems not to be NE tag."

        # Configurations
        model_args = NERArgs()
        model_args.train_batch_size = 8
        model_args.evaluate_during_training = False
        model_args.learning_rate = 5e-5
        model_args.num_train_epochs = 10
        model_args.use_early_stopping = True
        model_args.use_cuda = torch.cuda.is_available()  # Use GPU if available
        model_args.save_eval_checkpoints = False
        model_args.save_model_every_epoch = False  # Takes a lot of storage space
        model_args.save_steps = -1
        model_args.overwrite_output_dir = True
        model_args.cache_dir = model_name + "/cache"
        model_args.best_model_dir = model_name + "/best_model"
        model_args.output_dir = model_name
        model_args.use_multiprocessing = False
        model_args.silent = no_progress_bars

        # Initialization
        model = NERModel("camembert", model_name, args=model_args, labels=unique_labels)
        return model

    ################################################################
    # Notebook 3
    ################################################################

    @staticmethod
    def find_text_type(text: estnltk.Text, corpus_name: str, file_name: str):
        """
        Determines and assigns the text type metadata based on the provided corpus name and file name.

        This function examines the text's metadata to check if a 'texttype' is already assigned.
        If the 'texttype' is missing, it will infer and assign a text type based on the corpus
        name and the file name's prefix.

        Args:
            text (estnltk.Text): The EstNLTK Text object whose metadata is being processed.
            corpus_name (str): The name of the corpus to identify text type rules.
                Supported corpus names are: <code>'enc2017'</code> and <code>'UD_EST-EDT'</code>.
            file_name (str): The name of the file being processed, used to determine the text type
                if it is not already present in the metadata.

        """

        match corpus_name:
            case "enc2017":
                text_type = text.meta.get("texttype")  # Text type
                if not text_type:
                    if file_name.startswith("wiki17"):
                        text.meta.update({"texttype": "wikipedia"})
                    elif file_name.startswith("web13"):
                        text.meta.update({"texttype": "blogs_and_forums"})
                    else:
                        print("Unknown text type. Assigning text type 'unknown'.")
                        text.meta.update({"texttype": "unknown"})
            case "UD_EST-EDT":
                text_type = text.meta.get("texttype")  # Text type
                if not text_type:
                    if file_name.startswith("aja"):
                        text.meta.update({"texttype": "periodicals"})
                    elif file_name.startswith("ilu") or file_name.startswith(
                        "arborest"
                    ):
                        text.meta.update({"texttype": "fiction"})
                    elif file_name.startswith("tea"):
                        text.meta.update({"texttype": "science"})
                    else:
                        print("Unknown text type. Assigning text type 'unknown'.")
                        text.meta.update({"texttype": "unknown"})
            case _:
                return

    @staticmethod
    def create_json_file_by_file_enc2017(
        jsons: typing.List[str],
        in_dir: str,
        save_dir: str,
        corpus_name: typing.Optional[str],
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
        Creates a JSON file for each text file.
        <ul>
            <li>Skips JSON files that have already been created.</li>
            <li>Converts JSON file into EstNLTK Text object.</li>
            <li>Adds text type metadata and morph analysis.</li>
            <li>Adds <code>BertMorphTagger</code> layer</li>
            <li>Removes unnecessary layers.</li>
            <li>Converts EstNLTK Text object into JSON using <code>estnltk.converters.text_to_json.</code></li>
        </ul>
        Args:
            jsons (list): List of json files from which to read in the text
            in_dir (str): Directory where to read the json files from
            save_dir (str): Directory where to save the new json files
            corpus_name (optional, str): Name of the corpus from which the json files were generated.
                Currently supported are: <code>'enc2017'</code> and <code>'UD_EST-EDT'</code>.
            bert_morph_tagger (optional, BertMorphTagger): Configured <code>BertMorphTagger</code> class instance, if None, will not use this tagger
            necessary_layers (optional, list[str]): Text object layers that will not be deleted
            ignore_errors (optional, bool): Ignores texts that give errors when tagging
            replace_files (optional, bool): Replaces files with new ones in the given directory
        """

        count_errors = 0

        print("Beginning to morphologically tag file by file")
        for file_name in tqdm(jsons):
            # Skipping previous JSON files
            if not replace_files and os.path.exists(os.path.join(save_dir, file_name)):
                continue

            # Convert json to EstNLTK Text object
            text = estnltk.converters.json_to_text(file=os.path.join(in_dir, file_name))

            # Add text type metadata
            if corpus_name:
                NotebookFunctions.find_text_type(text, corpus_name, file_name)

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
                ), f"""Failed to assert file '{file_name}'
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
    def find_and_summarize_differences(
        jsons: typing.List[str],
        in_dir: str,
        output_dir: str,
        morph_diff_finder: MorphDiffFinder,
        morph_diff_summarizer: MorphDiffSummarizer,
    ):
        """
        Finds morphological differences in multiple JSON files and saves a summary.

        This function processes a list of JSON files, identifying morphological
        differences in the text of each file. The differences are then summarized
        and stored in an output file.

        For each JSON file:
        <ul>
            <li>Text objects are created from the JSON data.</li>
            <li>Morphological differences between the text layers are identified.</li>
            <li>These differences are recorded using the provided summarizer.</li>
            <li>If any differences are found, they are saved to a separate file.</li>
            <li>Finally, a summary of all differences is written to a statistics file.</li>
        </ul>

        Args:
            jsons (List[str]): A list of JSON filenames to be processed.
            in_dir (str): Directory path where the input JSON files are located.
            output_dir (str): Directory path where output summary files will be stored.
            morph_diff_finder (MorphDiffFinder): An instance of MorphDiffFinder used
                to identify morphological differences.
            morph_diff_summarizer (MorphDiffSummarizer): An instance of MorphDiffSummarizer
                used to record and summarize the differences.

        Outputs:
        <ul>
            <li>For each JSON file with differences, a text file will be created in the
            output directory containing the formatted differences.</li>
            <li>A final summary file is created in the output directory with statistics
            on the total differences across all processed files.</li>
        </ul>
        """

        # Finds and saves differences for each json file
        os.makedirs(output_dir, exist_ok=True)
        for json in tqdm(jsons):
            text_obj = estnltk.converters.json_to_text(file=os.path.join(in_dir, json))
            morph_diff_layer, formatted_diffs_str, _ = (
                morph_diff_finder.find_difference(
                    text_obj, fname=json, text_cat=text_obj.meta["texttype"]
                )
            )
            morph_diff_summarizer.record_from_diff_layer(
                "morph_analysis",
                morph_diff_layer,
                text_obj.meta["texttype"],
                start_new_doc=True,
            )
            if formatted_diffs_str is not None and len(formatted_diffs_str) > 0:
                fpath = os.path.join(output_dir, f"_{json}__ann_diffs.txt")
                write_formatted_diff_str_to_file(fpath, formatted_diffs_str)
            text_obj = None
            morph_diff_layer = None
            formatted_diffs_str = None

        # Summarizes the collected differences into results
        summarizer_result_str = morph_diff_summarizer.get_diffs_summary_output(
            show_doc_count=True
        )
        fpath = os.path.join(output_dir, f"_{'enc_2017'}__stats.txt")
        with open(fpath, "w", encoding="utf-8") as out_f:
            out_f.write("TOTAL DIFF STATISTICS:" + os.linesep + summarizer_result_str)
