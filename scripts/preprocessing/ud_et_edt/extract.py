import os
import re
import estnltk

from scripts.model.bert_morph_tagger import BertMorphTagger
from est_ud_utils import load_ud_file_texts_with_corrections
from est_ud_morph_conv import convert_ud_layer_to_reduced_morph_layer


class Extractor:
    """
    Extractor for the Universal Dependencies Estonian Dependency Treebank (UD_EST-EDT).
    TODO: Add description of the class and its methods.
    """

    def __init__(self):
        pass

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
