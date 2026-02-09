#
#   Script for making a random selection from *__ann_diffs_* file.
#
#   Partly based on:
#      https://github.com/estnltk/eval_experiments_lrec_2020
#

import re
import argparse
import os, os.path

from math import floor

from collections import defaultdict

from random import randint, seed

from estnltk import logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
       "Selects a random subset of differences from *__ann_diffs_* file. "+
       "By default, the chosen subset of differences will be written into a "+
       "file named as: the original file name + '_x' + amount of randomly "+
       "chosen differences. ")
    parser.add_argument('ann_diffs_file', type=str, \
                        help="file containing morphological annotation differences. "+\
                             "must be *__ann_diffs_* file created by one of the scripts "+\
                             "that finds morph analysis differences (e.g. diff_vm_bin.py).")
    parser.add_argument('rand_pick', type=int, \
                        help="integer value specifying the amount of differences "+\
                             "to be randomly chosen from the ann_diffs_file." )
    parser.add_argument('--seed', dest='random_seed_value', action='store', type=int, default=1,\
                        help="seed value used in making the random selection (default: 1).")
    parser.add_argument('-e', '--even', dest='pick_evenly', default=False, action='store_true', \
                        help="If set, attempts to pick even amount of random differences "+
                             "from each text subcategory / subcorpus. Note: this may fail due "+
                             "to some text subcategories having too little differences, or "+
                             "due to the amount of random picks could not be evenly distributed "+
                             "among all text subcategories. In case of a failure, "+
                             "falls back to making the selection over all files without "+
                             "considering the subcategory / subcorpus information. "+
                             "The same strategy is also used if the flag is not set. "+
                             "(default: False)", \
                        )
    parser.add_argument('--logging', dest='logging', action='store', default='info',\
                        choices=['debug', 'info', 'warning', 'error', 'critical'],\
                        help='logging level (default: info)')
    args = parser.parse_args()
    
    logger.setLevel( (args.logging).upper() )
    log = logger
    
    rand_pick = args.rand_pick
    diff_file = args.ann_diffs_file
    pick_evenly = args.pick_evenly
    random_seed_value = args.random_seed_value
    assert os.path.exists( diff_file ), '(!) Input file {} not found!'.format( diff_file )
    if '__ann_diffs_' not in diff_file:
        log.warn( f'The input file  {diff_file}  does not contain substring "__ann_diffs_". Is it correct file?' )
    assert 0 < rand_pick, '(!) rand_pick must be a positive integer'
    # Collect all diff gaps
    #   nc_periodicals::nc_255_27990::2
    pattern_diff_index = re.compile(r'^\s*([^:]*)::([^:]+)::(\(\d+,\s+\d+\))::(\d+)\s*$')
    diff_gaps = defaultdict(list)
    log.info('Collecting difference indexes ...')
    total_differences = 0
    with open(diff_file, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            line = line.strip()
            diff_ind_match = pattern_diff_index.match( line )
            if diff_ind_match:
                subcorpus_ind = diff_ind_match.group(1)
                diff_gaps[subcorpus_ind].append( line )
                total_differences += 1
    if total_differences == 0:
        log.error('(!) No differences found from the given ann_diffs_file. Invalid file, perhaps?')
        exit(1)
    if total_differences <= rand_pick:
        log.error('(!) Unreasonable rand_pick value {}: the ann_diffs_file contains only {} differences.'.format(rand_pick, total_differences))
        exit(1)
    # Output summary statistics over categories
    categories = len(diff_gaps.keys())
    even_pick_size = floor( rand_pick / categories )
    log.info('')
    log.info('Differences by text category / subcorpus:')
    categories_meeting_even_pick_size = 0
    total = sum([ len(diff_gaps[k]) for k in diff_gaps.keys() ])
    for subcorpus_ind in sorted(diff_gaps.keys(), key=lambda x: len(diff_gaps[x]), reverse=True ):
        per = (len(diff_gaps[subcorpus_ind]) / total)*100.0
        log.info( f' {len(diff_gaps[subcorpus_ind])}  ({per:.2f}%)  {subcorpus_ind}')
        if even_pick_size <= len( diff_gaps[subcorpus_ind] ):
            categories_meeting_even_pick_size += 1
    log.info('')
    log.info( f' {total}  (100.0%)  TOTAL')
    log.info('')
    if pick_evenly:
        if even_pick_size == 0:
            log.warn(f'Unable to make an even pick: there are {categories} categories, but only {rand_pick} can be chosen. Discarding pick_evenly setting.')
            pick_evenly = False
        elif rand_pick % categories != 0:
            log.warn(f'Unable to make an even pick from all {categories} categories: {rand_pick % categories} pick(s) will remain. Discarding pick_evenly setting.')
            pick_evenly = False
        elif categories_meeting_even_pick_size < categories:
            log.warn(f'Unable to make an even pick from {categories_meeting_even_pick_size} / {categories} categories. Discarding pick_evenly setting.')
            pick_evenly = False
    
    seed( random_seed_value )
    diff_gap_picks_flat = None
    if pick_evenly and even_pick_size > 0:
        # Make an even pick over all categories
        log.info(f'Picking randomly {even_pick_size} differences from each text category ...')
        diff_gap_picks = defaultdict(set)
        for subcorpus_ind in sorted(diff_gaps.keys(), key=lambda x: len(diff_gaps[x]), reverse=True ):
            subcorpus_total = len(diff_gaps[subcorpus_ind])
            failed_attempts = 0
            while len( diff_gap_picks[subcorpus_ind] ) < even_pick_size:
                if subcorpus_total > 1:
                    i = randint(0, subcorpus_total - 1)
                else:
                    i = 0
                gap_ind = diff_gaps[subcorpus_ind][i]
                if gap_ind not in diff_gap_picks[subcorpus_ind]:
                    diff_gap_picks[subcorpus_ind].add(gap_ind)
                    failed_attempts = 0
                else:
                    failed_attempts += 1
                    if failed_attempts >= 20:
                        log.error('(!) 20 unsuccessful random picks in a row: terminating ...')
                        break
        total_picks_made = sum([len(diff_gap_picks[s]) for s in diff_gap_picks.keys()])
        assert total_picks_made == rand_pick
        diff_gap_picks_flat = sorted( [p for s in diff_gap_picks.keys() for p in diff_gap_picks[s]] )
    else:
        # Make a pick over the whole corpus
        all_diff_gaps = [df_gap for k in sorted(diff_gaps.keys()) for df_gap in diff_gaps[k]]
        assert len(all_diff_gaps) == total_differences
        log.info(f'Picking randomly {rand_pick} differences from the whole corpus ...')
        diff_gap_picks = set()
        failed_attempts = 0
        while len( diff_gap_picks ) < rand_pick:
            if len(all_diff_gaps) > 1:
                i = randint(0, len(all_diff_gaps) - 1)
            else:
                i = 0
            gap_ind = all_diff_gaps[i]
            if gap_ind not in diff_gap_picks:
                diff_gap_picks.add(gap_ind)
                failed_attempts = 0
            else:
                failed_attempts += 1
                if failed_attempts >= 20:
                    log.error('(!) 20 unsuccessful random picks in a row: terminating ...')
                    break
        diff_gap_picks_flat = sorted( list(diff_gap_picks) )
    
    log.info('Collecting randomly picked differences ...')

    lines_of_rand_picked_examples = defaultdict(list)
    pattern_separator = ('='*85)
    with open(diff_file, 'r', encoding='utf-8') as in_f:
        collected_lines = []
        pickable_line   = False
        pick_index      = -1
        for line in in_f:
            line = line.strip()
            diff_ind_match = pattern_diff_index.match( line )
            if diff_ind_match:
                for pick_id, gap_id in enumerate(diff_gap_picks_flat):
                    if gap_id in line:
                        pickable_line = True
                        pick_index = pick_id
                        break
            collected_lines.append(line)
            if pattern_separator in line:
                if pickable_line:
                    assert pick_index not in lines_of_rand_picked_examples
                    lines_of_rand_picked_examples[pick_index] = collected_lines
                collected_lines = []
                pickable_line = False

    in_f_head, in_f_tail = os.path.split(diff_file)
    in_f_root, in_f_ext = os.path.splitext(in_f_tail)
    assert in_f_ext == '' or in_f_ext.startswith('.')
    is_even = '_even' if pick_evenly else ''
    out_fname = os.path.join(in_f_head, f'{in_f_root}_x{rand_pick}{is_even}{in_f_ext}')
    log.info('Saving into  {} ...'.format(out_fname) )
    with open(out_fname, 'w', encoding='utf-8') as out_f:
        for k in sorted( lines_of_rand_picked_examples.keys() ):
            for line in lines_of_rand_picked_examples[k]:
                out_f.write(line.rstrip())
                out_f.write('\n')
    log.info( 'Done.')
