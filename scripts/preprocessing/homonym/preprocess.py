import typing
import pandas as pd
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
            df = pd.DataFrame(data)
            output_csv = output_dir / Path(f"homonyms_infltype_{num}_{infl_type}.csv")
            df.to_csv(output_csv, index=False)
            print(f"Saved processed data to {output_csv}")
            overall_data.extend(data)

        if do_overall_df:
            # Create overall dataframe
            overall_df = pd.DataFrame(overall_data)
            overall_output_csv = output_dir / Path("homonyms_overall.csv")
            overall_df.to_csv(overall_output_csv, index=False)
            print(f"Saved overall processed data to {overall_output_csv}")
