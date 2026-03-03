import pandas as pd
import typing
import itertools
import json


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
        except:
            print("Could not read JSON file. Creating unique labels list.")

    # Separately, if one of two doesn't exist
    # pos_labels = [
    #     "A",
    #     "C",
    #     "D",
    #     "G",
    #     "H",
    #     "I",
    #     "J",
    #     "K",
    #     "N",
    #     "O",
    #     "P",
    #     "S",
    #     "U",
    #     "V",
    #     "X",
    #     "Y",
    #     "Z",
    # ]
    # form_labels = [
    #     "ab",
    #     "abl",
    #     "ad",
    #     "adt",
    #     "all",
    #     "el",
    #     "es",
    #     "g",
    #     "ill",
    #     "in",
    #     "kom",
    #     "n",
    #     "p",
    #     "pl",
    #     "sg",
    #     "ter",
    #     "tr",
    #     "b",
    #     "d",
    #     "da",
    #     "des",
    #     "ge",
    #     "gem",
    #     "gu",
    #     "ks",
    #     "ksid",
    #     "ksime",
    #     "ksin",
    #     "ksite",
    #     "ma",
    #     "maks",
    #     "mas",
    #     "mast",
    #     "mata",
    #     "me",
    #     "n",
    #     "neg",
    #     "neg ge",
    #     "neg gem",
    #     "neg gu",
    #     "neg ks",
    #     "neg me",
    #     "neg nud",
    #     "neg nuks",
    #     "neg o",
    #     "neg vat",
    #     "neg tud",
    #     "nud",
    #     "nuks",
    #     "nuksid",
    #     "nuksime",
    #     "nuksin",
    #     "nuksite",
    #     "nuvat",
    #     "o",
    #     "s",
    #     "sid",
    #     "sime",
    #     "sin",
    #     "site",
    #     "ta",
    #     "tagu",
    #     "taks",
    #     "takse",
    #     "tama",
    #     "tav",
    #     "tavat",
    #     "te",
    #     "ti",
    #     "tud",
    #     "tuks",
    #     "tuvat",
    #     "v",
    #     "vad",
    #     "vat",
    # ]

    pos_mutable = [
        "A",
        "C",
        "H",
        "N",
        "O",
        "P",
        "S",
        "T",
        "U",
        "X",
        "Y",
    ]  # T is foreign word or "tsitaatsõna" in Estonian
    pos_immutable = ["D", "G", "I", "J", "K", "Z"]
    pos_verb = ["V"]
    form_mutable = [
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
    form_mutable_count = ["pl", "sg"]
    form_verb = [
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
        # "neg da",
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

    # Combinations of only form
    only_form = (
        form_mutable
        + form_verb
        + list(
            itertools.product(form_mutable_count, form_mutable)
        )  # Non-verb mutable form with count
    )

    # Combinations of only pos
    only_pos = pos_mutable + pos_immutable + pos_verb

    # Combinations of mutable pos and their forms
    mutable_combinations = list(
        itertools.product(
            form_mutable, pos_mutable
        )  # Non-verb combinations without count
    ) + list(
        itertools.product(
            itertools.product(form_mutable_count, form_mutable), pos_mutable
        )  # Non-verb combinations with count
    )

    # Combinations of verb pos and their forms
    verb_combinations = list(
        itertools.product(
            form_verb,
            pos_verb,
        )
    )

    # Labels
    mutable_labels = list()

    # Connect count and form in mutables
    for comb, pos in mutable_combinations:
        form = comb
        if isinstance(form, tuple):  # form is a combination of count and form
            form = comb[0] + " " + comb[1]
        mutable_labels.append((form, pos))

    # immutable_labels = [("", pos) for pos in pos_immutable] # Is replaced by only_pos_labels
    verb_labels = [(form, pos) for form, pos in verb_combinations]
    only_pos_labels = [("", pos) for pos in only_pos]
    only_form_labels = list()
    for comb in only_form:
        form = comb
        if isinstance(form, tuple):  # form is a combination of count and form
            form = form[0] + " " + form[1]
        only_form_labels.append((form, ""))

    # Unknown form labels for all POS tags
    unknown_form_labels = [
        ("?", pos) for pos in only_pos
    ]  # form '?' comes from enc2017 corpus after tagging
    for comb in only_form:
        form = comb
        if form in form_verb:
            continue  # verb forms are only combined with verb POS, so we skip them here
        if isinstance(form, tuple):  # form is a combination of count and form
            form = form[0] + " " + form[1]
        unknown_form_labels.append((form, "?"))

    unique_labels = (
        mutable_labels
        # + immutable_labels
        + verb_labels
        + only_pos_labels
        + only_form_labels
        + unknown_form_labels
        + [("?", "")]
    )  # '?' for labels unknown to Vabamorf

    # Create a DataFrame for unique labels
    unique_labels_df = pd.DataFrame(unique_labels, columns=["form", "pos"])

    # Create the 'labels' column by combining 'form' and 'pos'
    unique_labels_df["labels"] = unique_labels_df.apply(
        lambda row: (
            row["form"] + "_" + row["pos"]
            if row["form"] and row["pos"]
            else row["form"] or row["pos"]
        ),
        axis=1,
    )

    # Drop form and pos columns and keep only labels column
    unique_labels_df = unique_labels_df.drop(labels=["form", "pos"], axis=1)
    print("List of unique labels created")
    return unique_labels_df["labels"].tolist()
