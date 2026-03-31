import os
from pathlib import Path
import typing

from estnltk.converters.label_studio.labelling_configurations import (
    PhraseClassificationConfiguration,
)
from estnltk.converters.label_studio.labelling_tasks import PhraseClassificationTask


class Extractor:
    def __init__(self):
        pass

    @staticmethod
    def extract(input_dir: typing.Union[str, Path]):

        # Collect input files
        input_files = []
        input_dir = Path(input_dir)
        for fname in os.listdir(input_dir):
            if os.path.isdir(os.path.join(input_dir, fname)):
                for subfname in os.listdir(os.path.join(input_dir, fname)):
                    if subfname.endswith(".json"):
                        inflection_type = int(
                            subfname.split("_")[2]
                        )  # infl_type_xx_1000_vx...
                        input_files.append(
                            (
                                inflection_type,
                                input_dir / fname / subfname,
                            )
                        )
            else:
                if fname.endswith(".json"):
                    inflection_type = int(
                        fname.split("_")[2]
                    )  # infl_type_xx_randomly_picked_1000_sentences...
                    input_files.append((inflection_type, input_dir / fname))

        if not input_files:
            raise RuntimeError("No input files found!")

        print(f"Found {len(input_files)} input files.")

        return input_files
