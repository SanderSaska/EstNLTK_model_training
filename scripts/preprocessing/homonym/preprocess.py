import typing
import pandas as pd
import tqdm
import estnltk
from pathlib import Path

from estnltk.converters.label_studio.labelling_configurations import (
    PhraseClassificationConfiguration,
)
from estnltk.converters.label_studio.labelling_tasks import PhraseClassificationTask


class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def create_df(
        input_files: typing.List[typing.Tuple],
        output_dir: typing.Union[str, Path],
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
                                "sentence": sentence_id,
                                "word": ma.text,
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
                                "sentence": sentence_id,
                                "word": ma.text,
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
            df = pd.DataFrame(data)
            output_csv = output_dir / Path(f"homonyms_infltype_{num}_{infl_type}.csv")
            df.to_csv(output_csv, index=False)
            print(f"Saved processed data to {output_csv}")
            overall_data.extend(data)

        if do_overall_df:
            # Create overall dataframe
            overall_df = pd.DataFrame(overall_data)
            overall_output_parquet = output_dir / Path("homonyms_overall.parquet")
            overall_df.to_parquet(overall_output_parquet, index=False)
            print(f"Saved overall processed data to {overall_output_parquet}")
