#
#   Picks randomly 1000 sentences for each target inflection types (1, 16, 17, 19) 
#   from the set of all extracted sentences containing words with homonymous forms 
#   ( `all_enc_koond_homonymous_forms_sentences.jl` ). 
#
#   Saves resulting sentences into directory 'random_pick_jl', into files which 
#   names follow the pattern: 
#       f'infl_type_{infl_type}_randomly_picked_1000_sentences_pick_{run_id}.jl'
#
#   By default, attempts to pick unique sentences across all inflection types. 
#
#   By default, attempts to discard all sentences picked in previous runs of 
#   this script. For this, collects all files from the outputdir that follow 
#   the pattern 
#       f'infl_type_{infl_type}_randomly_picked_1000_sentences_pick_{run_id}.jl'
#   and discards all sentences within these files.
#
#   Prints statistics about the whole dataset and about randomly picked subsets. 
#  

import json
import argparse
import os, os.path
import re

from typing import Sequence

import pandas as pd

from random import randint, seed

from tqdm import tqdm


def count_and_percent(items, items_total):
    '''Calculates items percentage from items_total.'''
    assert items_total > 0
    return f'{items} / {items_total} ({(items/items_total)*100.0:.2f}%)'


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


def load_previously_picked_sentences(input_dir, previous_picked_sentences, verbose=True):
    '''Loads sentences already picked in the previous runs of this script from the input directory. 
       Saves results into given dictionary previous_picked_sentences.'''
    assert isinstance( previous_picked_sentences, dict )
    start_count = len(previous_picked_sentences.keys())
    for fname in os.listdir( input_dir ):
        file_info = extract_randomly_picked_sentences_file_info(fname)
        if file_info is not None and fname.endswith('.jl'):
            with open( os.path.join(input_dir, fname), 'r', encoding='utf-8' ) as in_f:
                for line in in_f:
                    line = line.strip()
                    if len( line ) > 0:
                        sentence_dict = json.loads(line)
                        text_str = sentence_dict['text']
                        sent_id = sentence_dict["sent_id"]
                        doc_id = sentence_dict["doc_id"]
                        previous_picked_sentences[text_str] = [doc_id, sent_id]
    if verbose:
        end_count = len( previous_picked_sentences.keys() )
        print(f' Loaded {end_count-start_count} previously picked sentences.')
    return previous_picked_sentences


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for sentence sampling configuration.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Optional argument list for parsing. If ``None``, arguments are read
        from ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Pick random homonymous-form sentences for each inflection type '
            'and save picks into jsonlines files.'
        )
    )
    parser.add_argument(
        '--pick-target',
        type=int,
        default=1000,
        help='Number of sentences to pick per inflection type (default: 1000).',
    )
    parser.add_argument(
        '--all-sentences-uniq',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'If true, enforce sentence uniqueness across all inflection types '
            '(default: true).'
        ),
    )
    parser.add_argument(
        '--discard-previously-picked',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='If true, discard sentences picked in previous runs (default: true).',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='random_pick_jl',
        help='Output directory for picked sentence files (default: random_pick_jl).',
    )
    parser.add_argument(
        '--input-forms-file',
        type=str,
        default='homonymous_forms_1_16_17_19.csv',
        help='Input CSV file of homonymous forms.',
    )
    parser.add_argument(
        '--input-sent-file',
        type=str,
        default='all_enc_koond_homonymous_forms_sentences.jl',
        help='Input jsonlines file containing extracted sentences.',
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Runs random sentence sampling for target inflection types.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Optional argument list for parsing. If ``None``, arguments are read
        from ``sys.argv``.

    Returns
    -------
    None
        This function writes output files and prints summary statistics.
    """
    args = parse_args(argv)

    pick_target = args.pick_target
    if pick_target <= 0:
        raise ValueError('(!) --pick-target must be a positive integer.')

    all_sentences_uniq = args.all_sentences_uniq
    discard_previously_picked = args.discard_previously_picked
    output_dir = args.output_dir
    input_forms_file = args.input_forms_file
    input_sent_file = args.input_sent_file

    assert os.path.exists(input_forms_file), (
        f'(!) Missing homonymous forms file {input_forms_file!r}'
    )
    assert os.path.exists(input_sent_file), (
        f'(!) Missing homonymous form sentences file {input_sent_file!r}'
    )

    df_homonymous_forms = pd.read_csv(
        input_forms_file,
        header=0,
        keep_default_na=False,
    ).to_dict(orient='split')
    possible_homonymous_forms_by_infl_type = {}
    fields = ['lemma', 'inflect_type', 'pos', 'case', 'form']
    assert fields == df_homonymous_forms['columns']
    lemma_is_form = 0
    lemma_not_form = 0
    for row in df_homonymous_forms['data']:
        row_items = {f: v for f, v in zip(fields, row)}
        pos = row_items['pos']
        case = row_items['case']
        form = row_items['form']
        lemma = row_items['lemma']
        if form == lemma:
            lemma_is_form += 1
        else:
            lemma_not_form += 1
        origin_inf_type = row_items['inflect_type']
        inf_type = origin_inf_type
        if inf_type == '19|2':
            inf_type = '19'
        if inf_type not in possible_homonymous_forms_by_infl_type.keys():
            possible_homonymous_forms_by_infl_type[inf_type] = {}
        if form not in possible_homonymous_forms_by_infl_type[inf_type]:
            possible_homonymous_forms_by_infl_type[inf_type][form] = []
        possible_homonymous_forms_by_infl_type[inf_type][form].append(
            {'pos': pos, 'case': case, 'inf_type': origin_inf_type}
        )

    print('Total homonymous forms: ', lemma_is_form + lemma_not_form)
    print()
    print('     homonymous forms   ')
    print('    with lemma == form: ', lemma_is_form)
    print()
    for infl_type in sorted(
        possible_homonymous_forms_by_infl_type.keys(),
        key=lambda x: len(possible_homonymous_forms_by_infl_type[x]),
        reverse=True,
    ):
        print(
            f'   INFLECTION TYPE {infl_type}:  homonymous word forms : ',
            len(possible_homonymous_forms_by_infl_type[infl_type]),
        )
    print()

    homonymous_form_sents_by_infl_type = {}
    homonymous_form_uniq_sents_by_infl_type = {}
    homonymous_form_uniq_words_by_infl_type = {}
    sent_id = 0
    with open(input_sent_file, 'r', encoding='utf-8') as in_f:
        for line in tqdm(in_f, ascii=True):
            line = line.strip()
            if len(line) > 0:
                sent_id += 1
                sent_json = json.loads(line)
                assert 'text' in sent_json.keys()
                assert 'forms' in sent_json.keys()
                sent_str = sent_json['text']
                for form_entry in sent_json['forms']:
                    word_text = form_entry['word']
                    word_form = form_entry['form']
                    word_pos = word_form.split('_')[0]
                    start, end = form_entry['start'], form_entry['end']
                    word_in_sent = sent_str[start:end]
                    assert word_in_sent.lower() == word_text.lower()
                    for infl_type in sorted(
                        possible_homonymous_forms_by_infl_type.keys()
                    ):
                        if word_text in possible_homonymous_forms_by_infl_type[infl_type].keys():
                            for entry in possible_homonymous_forms_by_infl_type[infl_type][word_text]:
                                if entry['pos'] == word_pos:
                                    if infl_type not in homonymous_form_sents_by_infl_type:
                                        homonymous_form_sents_by_infl_type[infl_type] = []
                                        homonymous_form_uniq_sents_by_infl_type[infl_type] = set()
                                        homonymous_form_uniq_words_by_infl_type[infl_type] = set()
                                    new_entry = {}
                                    new_entry['corpus'] = sent_json['corpus']
                                    new_entry['doc_id'] = sent_json['doc_id']
                                    new_entry['sent_id'] = sent_id
                                    new_entry['text'] = sent_json['text']
                                    new_entry['word'] = word_text
                                    new_entry['partofspeech'] = word_pos
                                    new_entry['start'] = form_entry['start']
                                    new_entry['end'] = form_entry['end']
                                    homonymous_form_sents_by_infl_type[infl_type].append(new_entry)
                                    homonymous_form_uniq_sents_by_infl_type[infl_type].add(sent_id)
                                    homonymous_form_uniq_words_by_infl_type[infl_type].add(word_text)
                                    break

    # ===============   Collect randomly picked sentences from previous runs of this script (if any)  =====================
    # ===============               Remove previously picked sentences from available picks           =====================
    run_id = 0
    previous_picked_sentences = {}
    if discard_previously_picked and os.path.isdir(output_dir):
        # Discover previous runs
        for fname in os.listdir(output_dir):
            file_info = extract_randomly_picked_sentences_file_info(fname)
            if file_info is not None and fname.endswith('.jl'):
                prev_run_id = file_info[-1]
                if prev_run_id > run_id:
                    run_id = prev_run_id
        # Load previously picked sentences
        load_previously_picked_sentences(output_dir, previous_picked_sentences)
        # Remove previous picks from all possible picks
        if len(previous_picked_sentences.keys()) > 0:
            print(' Removing previously picked sentences from all available sentences.')
            for infl_type in homonymous_form_sents_by_infl_type.keys():
                filtered_sentences = []
                homonymous_form_uniq_sents_by_infl_type[infl_type] = set()
                homonymous_form_uniq_words_by_infl_type[infl_type] = set()
                for sent in homonymous_form_sents_by_infl_type[infl_type]:
                    sent_id = sent['sent_id']
                    word_text = sent['word']
                    if sent['text'] not in previous_picked_sentences.keys():
                        filtered_sentences.append(sent)
                        homonymous_form_uniq_sents_by_infl_type[infl_type].add(sent_id)
                        homonymous_form_uniq_words_by_infl_type[infl_type].add(word_text)
                    else:
                        # Remove already picked sentence
                        homonymous_form_uniq_sents_by_infl_type[infl_type].discard(sent_id)
                homonymous_form_sents_by_infl_type[infl_type] = filtered_sentences
    if run_id > 0:
        run_id += 1
    else:
        run_id = 1

    # =====================    Display statistics about each inflection type   ===============================

    print()
    for infl_type in sorted(
        homonymous_form_sents_by_infl_type.keys(),
        key=lambda x: len(homonymous_form_sents_by_infl_type[x]),
        reverse=True,
    ):
        uniq_sentences = len(homonymous_form_uniq_sents_by_infl_type[infl_type])
        propernoun_sentences = len(
            [
                s
                for s in homonymous_form_sents_by_infl_type[infl_type]
                if s['partofspeech'] == 'H'
            ]
        )
        total_sentences = len(homonymous_form_sents_by_infl_type[infl_type])
        print(f'    INFLECTION TYPE {infl_type}: number of homonymous form sentences: ', total_sentences)
        print(f'                           uniq sentences: ', count_and_percent(uniq_sentences, total_sentences))
        print(f'                propernoun word sentences: ', count_and_percent(propernoun_sentences, total_sentences))
        uniq_words = len(homonymous_form_uniq_words_by_infl_type[infl_type])
        all_words = len(possible_homonymous_forms_by_infl_type[infl_type])
        print(f'all uniq words coverage (from VM lexicon): ', count_and_percent(uniq_words, all_words))

    print()

    # =====================   Pick N sentences randomly from each inflection types  ===============================
    seed(1)
    picked_sentences_by_type = {}
    all_picked_sent_ids = set()
    print(f'Picking randomly sentences from each inflection type (pick #{run_id})...')
    print()
    for infl_type in sorted(
        homonymous_form_sents_by_infl_type.keys(),
        key=lambda x: len(homonymous_form_sents_by_infl_type[x]),
        reverse=True,
    ):
        # =====================   Make a random pick  ===============================
        local_picked_sent_ids = set()
        picked_sentences_by_type[infl_type] = []
        _total = len(homonymous_form_sents_by_infl_type[infl_type])
        failed_attempts = 0
        total_sentences = len(homonymous_form_sents_by_infl_type[infl_type])
        print(
            f'   INFLECTION TYPE {infl_type}: picking {pick_target} sentences from total of {total_sentences} sentences ... '
        )
        while len(picked_sentences_by_type[infl_type]) < pick_target:
            if _total > 1:
                i = randint(0, _total - 1)
            else:
                i = 0
            pick_sent = homonymous_form_sents_by_infl_type[infl_type][i]
            pick_sent_id = pick_sent['sent_id']
            sent_words = pick_sent['text'].split()
            all_sentences_uniq_satisfied = (not all_sentences_uniq) or (
                all_sentences_uniq and pick_sent_id not in all_picked_sent_ids
            )
            local_picked_uniq_sentences_satisfied = pick_sent_id not in local_picked_sent_ids
            # Sentence should be at least 4 words long and should
            # contain at leaste one lowercase word and should be
            # shorter than 1500 characters
            sent_constraints_satisfied = (
                len(sent_words) >= 4
                and any([w[0].islower() for w in sent_words])
                and len(pick_sent['text']) < 1500
            )
            # Discard sentence already picked in previous runs
            previously_not_picked_satisfied = (
                not discard_previously_picked
                or (pick_sent['text'] not in previous_picked_sentences.keys())
            )
            if (
                all_sentences_uniq_satisfied
                and local_picked_uniq_sentences_satisfied
                and sent_constraints_satisfied
                and previously_not_picked_satisfied
            ):
                picked_sentences_by_type[infl_type].append(pick_sent)
                failed_attempts = 0
                local_picked_sent_ids.add(pick_sent_id)
                all_picked_sent_ids.add(pick_sent_id)
            else:
                failed_attempts += 1
                if failed_attempts >= 20:
                    print('(!) 20 unsuccessful random picks in a row: terminating ...')
                    break
        print()
        # =====================   Get summary statistics of picked sentences  ===============================
        p_uniq_sentences_by_id = len(set([e['sent_id'] for e in picked_sentences_by_type[infl_type]]))
        p_uniq_sentences_by_text = len(set([e['text'] for e in picked_sentences_by_type[infl_type]]))
        p_uniq_words = len(set([e['word'] for e in picked_sentences_by_type[infl_type]]))
        p_propernoun_sentences = len(
            [s for s in picked_sentences_by_type[infl_type] if s['partofspeech'] == 'H']
        )
        total_sentences2 = len(picked_sentences_by_type[infl_type])
        print(f'         picked uniq sentences (by sent_id): ', count_and_percent(p_uniq_sentences_by_id, total_sentences2))
        print(f'       picked uniq sentences (by sent_text): ', count_and_percent(p_uniq_sentences_by_text, total_sentences2))
        print(f'                  propernoun word sentences: ', count_and_percent(p_propernoun_sentences, total_sentences2))
        all_words = len(possible_homonymous_forms_by_infl_type[infl_type])
        print(f'   all uniq words coverage (from VM lexicon): ', count_and_percent(p_uniq_words, all_words))
        print()
        os.makedirs(output_dir, exist_ok=True)
        out_file = (
            f'infl_type_{int(infl_type):02d}_randomly_picked_{pick_target}_sentences_pick_{run_id}.jl'
        )
        with open(os.path.join(output_dir, out_file), 'w', encoding='utf-8') as out_f:
            for entry_dict in picked_sentences_by_type[infl_type]:
                out_f.write(json.dumps(entry_dict, ensure_ascii=False))
                out_f.write('\n')
        print(f'   Writing randomly picked sentences into file {out_file} ... ')
        print()
        # Debug: display average, shortest, and longest picked sentence lengths.
        sorted_txts = sorted([s['text'] for s in picked_sentences_by_type[infl_type]], key=lambda x: len(x))
        from statistics import mean

        print('mean sentence length: ', mean([len(s) for s in sorted_txts]))
        print('3 longest lengths:    ', [len(s) for s in sorted_txts][-5:])
        for s in sorted_txts:
            if len(s) > 1400:
                print(f'sentence with length > 1400 ({len(s)}):    {s!r}')
                break
        print('3 shortest sentences: ', [f'{s!r}({len(s)})' for s in sorted_txts[:3]])
        print()
    print()


if __name__ == '__main__':
    main()

