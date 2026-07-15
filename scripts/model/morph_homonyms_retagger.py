"""EstNLTK retagger for correcting morphological homonyms.

The retagger uses a homonym lexicon as a gate and a BERT-based expert model
to re-evaluate only the words that are known homonymous forms. It leaves the
rest of the morphological layer untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Collection, MutableMapping, Optional, Tuple, Union

from estnltk import Layer, Text
from estnltk.taggers import Retagger

from scripts.model.bert_morph_tagger import BertMorphTagger

# TODO: Needs to be refactored when putting it into estnltk package.
DEFAULT_HOMONYM_LIST_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "homonymous_word_forms"
    / "processed"
    / "homonymous_words.txt"
)


def _load_homonym_words(path: Path, ignore_case: bool = True) -> set[str]:
    """Load homonymous words from a plain text file.

    Parameters
    ----------
    path:
        Path to a file containing one homonymous word per line.
    ignore_case:
        If ``True``, the words are normalised to lowercase for matching.

    Returns
    -------
    set[str]
        The loaded homonymous words.
    """

    if not path.exists():
        raise FileNotFoundError(f"Could not find homonym list file at {path!s}.")

    with path.open("r", encoding="utf-8") as file_handle:
        if ignore_case:
            return {line.strip().lower() for line in file_handle if line.strip()}
        return {line.strip() for line in file_handle if line.strip()}


class MorphHomonymsRetagger(Retagger):
    """Retagger that corrects morphological homonyms with a BERT expert.

    The retagger assumes that the input text has already been morphologically
    analysed. It then applies a homonym lexicon gate and only re-evaluates
    words that belong to that lexicon. If the expert model predicts a label
    that already exists among the word's analyses, the retagger keeps only the
    matching analyses.
    """

    conf_param = (
        "model_location",
        "homonym_list_path",
        "homonym_words",
        "ignore_case",
        "get_top_n_predictions",
        "layer_to_change",
        "prediction_layer_name",
        "token_level",
        "split_pos_form",
        "sentences_layer",
        "words_layer",
        "input_layers",
        "_bert_morph_tagger",
        "output_attributes",
    )

    def __init__(
        self,
        model_location: Optional[str] = None,
        homonym_list_path: Optional[Union[str, Path]] = None,
        homonym_words: Optional[Collection[str]] = None,
        ignore_case: bool = True,
        get_top_n_predictions: int = 1,
        layer_to_change: str = "bert_morph_tagging",
        token_level: bool = False,
        split_pos_form: bool = True,
        sentences_layer: str = "sentences",
        words_layer: str = "words",
        **kwargs,
    ):
        """Initialise the retagger.

        Parameters
        ----------
        model_location:
            Path to the BERT expert model directory. If omitted, the
            underlying BERT tagger uses its default resource lookup.
        homonym_list_path:
            Path to a newline-separated list of homonymous word forms.
            Used when ``homonym_words`` is not provided.
        homonym_words:
            Optional in-memory collection of homonymous words. If supplied,
            this takes precedence over ``homonym_list_path``.
        ignore_case:
            If ``True``, homonym matching is done in lowercase.
        get_top_n_predictions:
            Number of expert labels to request from the BERT model.
            The first matching analysis is used for retagging.
        layer_to_change:
            Name of the existing morphological layer that will be modified in
            place.
        sentences_layer:
            Name of the sentence layer.
        words_layer:
            Name of the word layer.
        **kwargs:
            Additional tokenizer kwargs forwarded to the internal BERT tagger.
        """

        self.model_location = model_location
        self.homonym_list_path = (
            str(homonym_list_path) if homonym_list_path is not None else None
        )
        self.ignore_case = ignore_case
        self.get_top_n_predictions = get_top_n_predictions
        self.layer_to_change = layer_to_change
        self.output_layer = layer_to_change
        self.sentences_layer = sentences_layer
        self.words_layer = words_layer
        self.input_layers = [sentences_layer, words_layer, layer_to_change]
        self.token_level = token_level
        self.split_pos_form = split_pos_form

        if homonym_words is not None:
            self.homonym_words = {
                word.lower() if ignore_case else str(word)
                for word in homonym_words
                if str(word).strip()
            }
        else:
            resolved_path = (
                Path(homonym_list_path)
                if homonym_list_path is not None
                else DEFAULT_HOMONYM_LIST_PATH
            )
            self.homonym_list_path = str(resolved_path)
            self.homonym_words = _load_homonym_words(
                resolved_path, ignore_case=ignore_case
            )
        self.output_attributes = (
            ["bert_tokens", "form", "partofspeech", "probability"]
            if self.split_pos_form
            else ["bert_tokens", "morph_label", "probability"]
        )
        prediction_output_layer = f"{self.output_layer}__homonym_predictions"
        self.prediction_layer_name = prediction_output_layer
        self._bert_morph_tagger = BertMorphTagger(
            model_location=model_location,
            get_top_n_predictions=get_top_n_predictions,
            output_layer=prediction_output_layer,
            sentences_layer=sentences_layer,
            words_layer=words_layer,
            token_level=token_level,
            split_pos_form=split_pos_form,
            **kwargs,
        )

    def _is_homonym_word(self, word_text: str) -> bool:
        """Check whether a surface form belongs to the homonym lexicon."""

        candidate = word_text.lower() if self.ignore_case else word_text
        return candidate in self.homonym_words

    def _extract_predicted_label(self, predicted_span) -> Optional[Tuple[str, str]]:
        """Return the expert's best form/POS pair for a predicted word."""

        if not predicted_span.annotations:
            return None

        predicted_annotation = predicted_span.annotations[0]
        form = predicted_annotation.get("form")
        partofspeech = predicted_annotation.get("partofspeech")
        if form is None or partofspeech is None:
            return None
        return str(form), str(partofspeech)

    def _build_predicted_span_map(
        self,
        text: Text,
        layers: MutableMapping[str, Layer],
        status: dict,
    ) -> dict[tuple[int, int], Layer]:
        """Build a lookup table from span boundaries to expert predictions.

        The BERT helper is only invoked for sentences that contain at least one
        homonymous word. Sentences without homonyms are skipped entirely.
        """

        predicted_by_span: dict[tuple[int, int], Layer] = {}
        sentences_layer = layers[self.sentences_layer]
        source_layer = layers[self.layer_to_change]

        for sentence in sentences_layer:
            sentence_has_homonym = any(
                self._is_homonym_word(span.text)
                for span in source_layer
                if span.start >= sentence.start and span.end <= sentence.end
            )
            if not sentence_has_homonym:
                continue

            sentence_text = Text(sentence.enclosing_text)
            sentence_text.tag_layer([self.sentences_layer, self.words_layer])
            sentence_layers = sentence_text.layers.union(sentence_text.relation_layers)
            expert_layer = self._bert_morph_tagger.make_layer(
                text=sentence_text,
                layers=sentence_layers,
                status=status,
            )

            for span in expert_layer:
                predicted_by_span[
                    (span.start + sentence.start, span.end + sentence.start)
                ] = span

        return predicted_by_span

    def _copy_span_annotations(
        self,
        source_span,
        target_layer: Layer,
        annotations,
    ) -> None:
        """Copy annotations from a source span into the target layer."""

        for annotation in annotations:
            target_layer.add_annotation(
                (source_span.start, source_span.end),
                **dict(annotation),
            )

    def _correct_layer(
        self,
        source_layer: Layer,
        predicted_by_span: dict[tuple[int, int], Layer],
    ) -> Layer:
        """Apply homonym corrections in place on the source layer."""

        corrected_words = 0
        inspected_words = 0
        # Iterate over the source layer and check each span against the predicted spans
        for source_span in source_layer:
            # Check if the source span has a corresponding predicted span
            predicted_span = predicted_by_span.get((source_span.start, source_span.end))
            # If there is no predicted span or the source span is not a homonym, skip it
            if predicted_span is None or not self._is_homonym_word(source_span.text):
                continue
            # If the predicted span exists and the source span is a homonym, we need to check if the predicted label matches any of the existing annotations in the source span
            inspected_words += 1
            predicted_label = self._extract_predicted_label(predicted_span)
            # If the predicted label is None, we skip this span
            if predicted_label is None:
                continue
            # Check if the predicted label matches any of the existing annotations in the source span
            predicted_form, predicted_pos = predicted_label
            matching_annotations = [
                annotation
                for annotation in source_span.annotations
                if annotation.get("form") == predicted_form
                and annotation.get("partofspeech") == predicted_pos
            ]
            # If there are matching annotations, we keep only those;
            # otherwise, we keep the original annotations.
            # We also count how many words were corrected.
            if matching_annotations:
                annotations_to_copy = list(matching_annotations)
                corrected_words += 1
            else:
                annotations_to_copy = list(source_span.annotations)
            # Clear the existing annotations in the source span and copy the selected annotations back into it
            source_span.clear_annotations()
            for annotation in annotations_to_copy:
                source_span.add_annotation(dict(annotation))
        # Store some statistics about the retagging process in the source layer's metadata
        source_layer.meta["morph_homonyms_retagger"] = {
            "inspected_words": inspected_words,
            "corrected_words": corrected_words,
            "homonym_list_size": len(self.homonym_words),
            "layer_to_change": self.layer_to_change,
            "output_layer": self.output_layer,
        }
        return source_layer

    def _change_layer(
        self,
        text: Text,
        layers: MutableMapping[str, Layer],
        status: dict,
    ) -> None:
        """Retag the existing morphological layer in place."""

        assert self.output_layer in layers
        assert self.sentences_layer in layers
        assert self.words_layer in layers
        assert self.layer_to_change in layers

        morph_layer = layers[self.layer_to_change]
        predicted_by_span = self._build_predicted_span_map(text, layers, status)
        self._correct_layer(
            source_layer=morph_layer,
            predicted_by_span=predicted_by_span,
        )
        layers[self.layer_to_change] = morph_layer
