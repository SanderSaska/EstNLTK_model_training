#
#   Converts files that have 1000 sentences with homonymous words to labelstudio annotation tasks. 
#   The goal of tasks is to disambiguate homonymous forms. 
#
#   Writes labelstudio import files into subdirectories of the directory "to_labelstudio", each 
#   subdirectory will be named after {pick_id}, e.g. files of 
#     "infl_type_01_randomly_picked_1000_sentences_pick_1.jl" will go into "to_labelstudio/1"
#     "infl_type_01_randomly_picked_1000_sentences_pick_2.jl" will go into "to_labelstudio/2"
#   etc.
#
#   Requires estnltk version that has label_studio PhraseClassificationTask available
#   ( version 1.7.4+ )
# 

import os, os.path
import json, re
import argparse

from typing import Sequence

from estnltk import Text, Layer
from estnltk.converters.label_studio.labelling_configurations import PhraseClassificationConfiguration
from estnltk.converters.label_studio.labelling_tasks import PhraseClassificationTask

annotation_confs = {1:  PhraseClassificationConfiguration(phrase_labels=['analüüsitav sõna'], 
                                                          class_labels={'sg n': 'sg n', 'sg g': 'sg g'}, 
                                                          header="Vali sõna morfoloogiline vorm (sg n - ainsuse nimetav, sg g -- ainsuse omastav):",
                                                          header_placement='middle'),
                    16: PhraseClassificationConfiguration(phrase_labels=['analüüsitav sõna'], 
                                                          class_labels={'sg n': 'sg n', 'sg g': 'sg g'}, 
                                                          header="Vali sõna morfoloogiline vorm (sg n - ainsuse nimetav, sg g -- ainsuse omastav):",
                                                          header_placement='middle'),
                    17: PhraseClassificationConfiguration(phrase_labels=['analüüsitav sõna'], 
                                                          class_labels={'sg n': 'sg n', 'sg g': 'sg g', 'sg p': 'sg p'}, 
                                                          header="Vali sõna morfoloogiline vorm (sg n - ainsuse nimetav, sg g - ainsuse omastav, sg p - ainsuse osastav):",
                                                          header_placement='middle'),
                    19: PhraseClassificationConfiguration(phrase_labels=['analüüsitav sõna'], 
                                                          class_labels={'sg g': 'sg g', 'sg p': 'sg p', 'adt': 'adt'}, 
                                                          header="Vali sõna morfoloogiline vorm (sg g - ainsuse omastav, sg p - ainsuse osastav, adt - lühike sisseütlev):",
                                                          header_placement='middle')}

def extract_randomly_picked_sentences_file_info(fname):
    '''Extracts inflection type, sentences amount and pick number from given 
       randomly picked sentences file name and returns as a tuple of integers.
       Returns None if the file name does not follow the pattern of 
       randomly picked sentences files.'''
    m0 = re.match(r'^infl_type_(\d+)_randomly_picked_(\d+)_sentences\.jl$', fname)
    if m0:
        return (int(m0.group(1)), int(m0.group(2)), 1)
    m1 = re.match(r'^infl_type_(\d+)_randomly_picked_(\d+)_sentences_pick_(\d+)\.jl$', fname)
    if m1:
        return (int(m1.group(1)), int(m1.group(2)), int(m1.group(3)))
    return None

# Smoke tests
assert extract_randomly_picked_sentences_file_info('infl_type_17_randomly_picked_1000_sentences.jl') == \
            (17, 1000, 1)
assert extract_randomly_picked_sentences_file_info('infl_type_17_randomly_picked_1000_sentences_pick_1.jl') == \
            (17, 1000, 1)

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for Label Studio export.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Optional argument list. If ``None``, values are read from ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='Convert randomly picked homonymous sentence files to Label Studio tasks.'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='random_pick_jl',
        help='Directory containing picked sentence .jl files (default: random_pick_jl).',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='to_labelstudio',
        help='Base output directory for Label Studio exports (default: to_labelstudio).',
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Exports random picks into Label Studio import and interface files.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Optional argument list. If ``None``, values are read from ``sys.argv``.

    Returns
    -------
    None
        Writes exported Label Studio files to the target output directory.
    """
    args = parse_args(argv)
    input_dir = args.input_dir
    out_base_dir = args.output_dir

    assert os.path.exists(input_dir), f'(!) Missing input directory {input_dir!r}'

    # Collect input files
    input_files = []
    for fname in os.listdir(input_dir):
        file_info = extract_randomly_picked_sentences_file_info(fname)
        if file_info is not None and fname.endswith('.jl'):
            infl_type = file_info[0]
            pick_id = file_info[-1]
            assert infl_type in annotation_confs.keys()
            input_files.append((os.path.join(input_dir, fname), fname, infl_type, pick_id))
    if not input_files:
        print(f'(!) No randomly picked sentences jl files found from {input_dir!r}.')

    os.makedirs(out_base_dir, exist_ok=True)

    for (in_fpath, fname, infl_type, pick_id) in input_files:
        annotation_conf = annotation_confs[infl_type]
        text_objs = []
        out_dir = os.path.join(out_base_dir, f'{pick_id}')
        os.makedirs(out_dir, exist_ok=True)
        with open(in_fpath, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                line = line.strip()
                if len(line) > 0:
                    sentence_dict = json.loads(line)
                    text = Text(sentence_dict['text'])
                    text.meta['corpus'] = sentence_dict['corpus']
                    text.meta['doc_id'] = sentence_dict['doc_id']
                    text.meta['sent_id'] = sentence_dict['sent_id']
                    layer = Layer('morph', attributes=('label',), text_object=text)
                    word_span = (sentence_dict['start'], sentence_dict['end'])
                    assert (
                        text.text[word_span[0]:word_span[1]].lower() == sentence_dict['word'].lower()
                    ), sentence_dict['word']
                    layer.add_annotation(word_span, label='analüüsitav sõna')
                    text.add_layer(layer)
                    text_objs.append(text)
                    # if len(text_objs) > 5:
                    #     break
        task = PhraseClassificationTask(
            annotation_conf,
            input_layer='morph',
            output_layer='morph',
            label_attribute='label',
        )
        exported_data = task.export_data(text_objs, indent=2)
        # Write out task configuration
        out_label_conf_fname = fname.replace('.jl', '_labeling_interface.txt')
        out_label_conf_fname = out_label_conf_fname.replace('_randomly_picked_', '_')
        out_label_conf_fname = out_label_conf_fname.replace(f'_pick_{pick_id}', '')
        with open(os.path.join(out_dir, out_label_conf_fname), 'w', encoding='utf-8') as out_f:
            out_f.write(task.interface_file)
        # Write out data for annotation task
        out_json_fname = fname.replace('.jl', '.json')
        out_json_fname = out_json_fname.replace('_randomly_picked_', '_')
        out_json_fname = out_json_fname.replace(f'_pick_{pick_id}', '')
        with open(os.path.join(out_dir, out_json_fname), 'w', encoding='utf-8') as out_f:
            out_f.write(exported_data)


if __name__ == '__main__':
    main()
