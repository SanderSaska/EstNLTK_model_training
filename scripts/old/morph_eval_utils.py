# =========================================================
# =========================================================
#  Utilities for finding & recording morph_analysis
#  differences
# =========================================================
# =========================================================
#
#  Partly based on: 
#   https://github.com/estnltk/eval_experiments_lrec_2020
#

import os, os.path, re
from collections import defaultdict

from estnltk.text import Text
from estnltk import Layer
from estnltk_core.layer_operations import flatten
from estnltk import Annotation
from estnltk.taggers import DiffTagger
from estnltk.taggers.system.diff_tagger import iterate_diff_conflicts
from estnltk.taggers.system.diff_tagger import iterate_modified

# =================================================
# =================================================
#    Creating flat layers
# =================================================
# =================================================

def create_flat_v1_6_morph_analysis_layer( text_obj, morph_layer, output_layer, add_layer=True ):
    '''Creates copy of estnltk v1.6 morph_analysis layer that is a flat layer containing only segmentation. '''
    assert isinstance(text_obj, Text)
    assert morph_layer in text_obj.layers, '(!) Layer {!r} missing from: {!r}'.format(morph_layer, text_obj.layers)
    flat_morph = flatten(text_obj[ morph_layer ], output_layer )
    if add_layer:
        text_obj.add_layer( flat_morph )
    return flat_morph

def create_redux_morph_layer( text_obj, morph_layer, output_layer, add_layer=True ):
    '''Creates a reduced version of morph_analysis layer which only contains attributes 'partofspeech' and 'form'.
       The reduced morph layer is required for comparing Vabamorf's annotations against Bert annotations 
       via DiffTagger (because DiffTagger assumes that comparable layers have common attributes).
    '''
    assert isinstance(text_obj, Text)
    assert morph_layer in text_obj.layers, \
           '(!) Layer {!r} missing from: {!r}'.format(morph_layer, text_obj.layers)
    redux_layer = Layer(name=output_layer, \
                        attributes=('partofspeech', 'form'), \
                        text_object=text_obj,\
                        ambiguous=True)
    for morph_word in text_obj[ morph_layer ]:
        for ann in morph_word.annotations:
            assert 'partofspeech' in ann.keys(), '(!) Attribute {!r} missing from: {!r}'.format('pos', ann)
            assert 'form' in ann.keys(), '(!) Attribute {!r} missing from: {!r}'.format('form', ann)
            postag = ann['partofspeech']
            form = ann['form']
            redux_layer.add_annotation( morph_word.base_span, {'partofspeech':  postag,\
                                                               'form': form } )
    if add_layer:
        text_obj.add_layer( redux_layer )
    return redux_layer


# =================================================
# =================================================
#    Creating BERT layer with 
#    Vabamorf's annotations
# =================================================
# =================================================

def create_bert_with_vabamorf_annotations_layer( text_obj:Text, bert_morph_layer:str, new_morph_layer:str, add_layer: bool):
    '''Creates a BERT layer with Vabamorf's annotations (<code>partofspeech</code> and <code>form</code>)

    Args:
        text_obj (Text): EstNLTK Text object
        bert_morph_layer (str): BERT morph layer's name
        new_morph_layer (str): New BERT morph layer's name
        add_layer (bool): Whether to add the new morph layer to the Text object
    '''
    new_layer = Layer(name=new_morph_layer, attributes=['bert_tokens', 'form', 'partofspeech'], text_object=text_obj, parent='words', ambiguous=True)
    for sp in text_obj[bert_morph_layer].spans:
        span = sp.base_span
        tokens = sp.bert_tokens[0]
        label = sp.morph_labels[0].split('_')
        pos = ''
        form = ''
        if len(label) == 1: # Only one feature exists (form or partofspeech)
            if label[0].isupper(): # partofspeech is uppercased
                pos = sp.morph_labels[0].split('_')[0]
            else:
                form = sp.morph_labels[0].split('_')[0]
        else: # Both features exist
            form = sp.morph_labels[0].split('_')[0]
            pos = sp.morph_labels[0].split('_')[1]

        annotation = {
            'bert_tokens': tokens,
            'form': form,
            'partofspeech': pos
        }
        new_layer.add_annotation(span, annotation)
    if add_layer:
        text_obj.add_layer(new_layer)


# =================================================
# =================================================
#    Get morph analysis diff
#    ( Vabamorf's annotations )
# =================================================
# =================================================

def get_estnltk_morph_analysis_diff_annotations( text_obj, layer_a, layer_b, diff_layer ):
    ''' Collects differing sets of annotations from EstNLTK's morph_analysis diff_layer. '''
    STATUS_ATTR = '__status'
    assert isinstance(text_obj, Text)
    assert isinstance(layer_a, Layer)
    assert isinstance(layer_b, Layer)
    assert isinstance(diff_layer, Layer)
    layer_a_name = layer_a.name
    layer_b_name = layer_b.name
    layer_a_spans = layer_a
    layer_b_spans = layer_b
    common_attribs = set(layer_a.attributes).intersection( set(layer_b.attributes) )
    assert len(common_attribs) > 0, '(!) Layers {!r} and {!r} have no common attributes!'.format(layer_a_name, layer_b_name)
    assert STATUS_ATTR not in common_attribs, "(!) Unexpected attribute {!r} in {!r}.".format(STATUS_ATTR, common_attribs)
    assert layer_a_name not in ['start', 'end']
    assert layer_b_name not in ['start', 'end']
    collected_diffs = []
    missing_annotations = 0
    extra_annotations   = 0
    a_id = 0
    b_id = 0
    for diff_span in iterate_modified( diff_layer, 'span_status' ):
        ds_start = diff_span.start
        ds_end =   diff_span.end
        # Find corresponding span in both layer
        a_span = None
        b_span = None
        while a_id < len(layer_a_spans):
            cur_a_span = layer_a_spans[a_id]
            if cur_a_span.start == ds_start and cur_a_span.end == ds_end:
                a_span = cur_a_span
                break
            a_id += 1
        while b_id < len(layer_b_spans):
            cur_b_span = layer_b_spans[b_id]
            if cur_b_span.start == ds_start and cur_b_span.end == ds_end:
                b_span = cur_b_span
                break
            b_id += 1
        if a_span == None:
            raise Exception('(!) {!r} not found from layer {!r}'.format(diff_span, layer_a_name))
        if b_span == None:
            raise Exception('(!) {!r} not found from layer {!r}'.format(diff_span, layer_b_name))
        a_annotations = []
        for a_anno in a_span.annotations:
            a_dict = a_anno.__dict__.copy()
            a_dict = {a:a_dict[a] for a in a_dict.keys() if a in common_attribs}
            a_dict[STATUS_ATTR] = None
            a_annotations.append( a_dict )
        b_annotations = []
        for b_anno in b_span.annotations:
            b_dict = b_anno.__dict__.copy()
            b_dict = {b:b_dict[b] for b in b_dict.keys() if b in common_attribs}
            b_dict[STATUS_ATTR] = None
            b_annotations.append( b_dict )
        for a_anno in a_annotations:
            match_found = False
            for b_anno in b_annotations:
                if a_anno == b_anno:
                    a_anno[STATUS_ATTR] = 'COMMON'
                    b_anno[STATUS_ATTR] = 'COMMON'
                    match_found = True
                    break
            if not match_found:
                missing_annotations += 1
                a_anno[STATUS_ATTR] = 'MISSING'
        for b_anno in b_annotations:
            if b_anno not in a_annotations:
                extra_annotations += 1
                b_anno[STATUS_ATTR] = 'EXTRA'
        collected_diffs.append( {'text':diff_span.text, 
                                 layer_a_name: a_annotations, 
                                 layer_b_name: b_annotations, 
                                 'start':diff_span.start, 
                                 'end':diff_span.end} )
    # Sanity check: missing vs extra annotations:
    # Note: text_obj[diff_layer].meta contains more *_annotations items, because it also 
    #       counts annotations in missing spans and extra spans; Unfortunately, merely
    #       subtracting:
    #                       missing_annotations - missing_spans
    #                       extra_annotations - extra_spans
    #       does not work either, because one missing or extra span may contain more 
    #       than one annotation. So, we have to re-count extra and missing annotations ...
    normalized_extra_annotations   = 0
    normalized_missing_annotations = 0
    for diff_span in diff_layer:
        for status in diff_span.span_status:
            if status == 'missing':
                normalized_missing_annotations += 1
            elif status == 'extra':
                normalized_extra_annotations += 1
    assert missing_annotations == diff_layer.meta['missing_annotations'] - normalized_missing_annotations
    assert extra_annotations == diff_layer.meta['extra_annotations'] - normalized_extra_annotations
    return collected_diffs


def get_estnltk_morph_analysis_annotation_alignments( collected_diffs, layer_names, morph_diff_layer, \
                                                                focus_attributes=['root','partofspeech', 'form'], \
                                                                remove_status=True ):
    ''' Calculates annotation alignments between annotations in collected_diffs. '''
    assert isinstance(layer_names, list) and len(layer_names) == 2
    STATUS_ATTR = '__status'
    MATCHING_ATTR    = '__matching'
    MISMATCHING_ATTR = '__mismatching'
    alignments  = []
    annotations_by_layer = defaultdict(int)
    if len(collected_diffs) > 0:
        first_diff = collected_diffs[0]
        all_attributes = []
        for key in first_diff.keys():
            if key not in ['text', 'start', 'end']:
                all_attributes = [k for k in first_diff[key][0].keys() if k != STATUS_ATTR]
                assert key in layer_names
        assert len( all_attributes ) > 0
        assert len([a for a in focus_attributes if a in all_attributes]) == len(focus_attributes)
        for word_diff in collected_diffs:
            alignment = word_diff.copy()
            a_anns = word_diff[layer_names[0]]
            b_anns = word_diff[layer_names[1]]
            annotations_by_layer[layer_names[0]] += len(a_anns)
            annotations_by_layer[layer_names[1]] += len(b_anns)
            alignment['alignments'] = []
            del alignment[layer_names[0]]
            del alignment[layer_names[1]]
            a_used = set()
            b_used = set()
            for a_id, a in enumerate(a_anns):
                # Find fully matching annotation
                for b_id, b in enumerate(b_anns):
                    if a == b:
                        al = {STATUS_ATTR:'COMMON', layer_names[0]:a, layer_names[1]:b }
                        al[MISMATCHING_ATTR] = []
                        al[MATCHING_ATTR] = all_attributes.copy()
                        alignment['alignments'].append( al )
                        a_used.add(a_id)
                        b_used.add(b_id)
                        break
                if a_id in a_used:
                    continue
                # Find partially matching annotation
                closest_b = None
                closest_b_id = None
                closest_common   = []
                closest_uncommon = []
                for b_id, b in enumerate(b_anns):
                    if a_id in a_used or b_id in b_used:
                        continue
                    if b[STATUS_ATTR] == 'COMMON':
                        # Skip b that has been previously found as being common
                        continue
                    if a != b:
                        #count common attribs
                        matching_attribs = []
                        mismatching = []
                        for attr in all_attributes:
                            if a[attr] == b[attr]:
                                matching_attribs.append(attr)
                            else:
                                mismatching.append(attr)
                        if len(matching_attribs) > len(closest_common):
                            focus_1 = []
                            focus_2 = []
                            if closest_b != None:
                                focus_1 = [a for a in focus_attributes if a in matching_attribs]
                                focus_2 = [a for a in focus_attributes if a in closest_common]
                            # in case of a tie, prefer matches with more focus attributes
                            if len(focus_1) == len(focus_2) or len(focus_1) > len(focus_2):
                                closest_common   = matching_attribs
                                closest_uncommon = mismatching
                                closest_b_id = b_id
                                closest_b = b
                if closest_b != None:
                    al = {STATUS_ATTR:'MODIFIED', layer_names[0]:a, layer_names[1]:closest_b }
                    al[MISMATCHING_ATTR] = closest_uncommon
                    al[MATCHING_ATTR] = closest_common
                    alignment['alignments'].append( al )
                    a_used.add(a_id)
                    b_used.add(closest_b_id)
                else:
                    al = {STATUS_ATTR:'MISSING', layer_names[0]:a, layer_names[1]:{} }
                    al[MISMATCHING_ATTR] = all_attributes.copy()
                    al[MATCHING_ATTR] = []
                    alignment['alignments'].append( al )
                    a_used.add(a_id)
            for b_id, b in enumerate(b_anns):
                if b_id not in b_used:
                    al = {STATUS_ATTR:'EXTRA', layer_names[0]:{}, layer_names[1]:b }
                    al[MISMATCHING_ATTR] = all_attributes.copy()
                    al[MATCHING_ATTR] = []
                    alignment['alignments'].append( al )
            alignments.append( alignment )
    # Sanity check #1: check that we haven't lost any annotations during the careful alignment
    annotations_by_layer_2 = defaultdict(int)
    for word_diff in alignments:
        for al in word_diff['alignments']:
            for layer in layer_names:
                if len(al[layer].keys()) > 0:
                    annotations_by_layer_2[layer] += 1
    for layer in layer_names:
        if annotations_by_layer[layer] != annotations_by_layer_2[layer]:
           # Output information about the context of the failure
            from pprint import pprint
            print('='*50)
            print(layer,'  ',annotations_by_layer[layer], '  ', annotations_by_layer_2[layer])
            print('='*50)
            pprint(collected_diffs)
            print('='*50)
            pprint(alignments)
            print('='*50)
        assert annotations_by_layer[layer] == annotations_by_layer_2[layer], '(!) Failure in annotation conversion.'
    # Remove STATUS_ATTR's from annotations dict's (if required)
    if remove_status:
        for word_diff in alignments:
            for al in word_diff['alignments']:
                for layer in layer_names:
                    if STATUS_ATTR in al[layer].keys():
                        del al[layer][STATUS_ATTR]
    # Sanity check #2: check that we are consistent with counts in morph_diff_layer:
    #   unchanged_annotations + missing_annotations = number_of_annotations_in_old_layer
    #   unchanged_annotations + extra_annotations   = number_of_annotations_in_new_layer
    normalized_extra_annotations   = 0
    normalized_missing_annotations = 0
    for diff_span in morph_diff_layer:
        for status in diff_span.span_status:
            if status == 'missing':
                normalized_missing_annotations += 1
            elif status == 'extra':
                normalized_extra_annotations += 1
    unchanged_annotations = morph_diff_layer.meta['unchanged_annotations']
    missing_annotations   = morph_diff_layer.meta['missing_annotations'] - normalized_missing_annotations 
    extra_annotations     = morph_diff_layer.meta['extra_annotations'] - normalized_extra_annotations
    missing_annotations_2 = 0
    extra_annotations_2   = 0
    for word_alignment in alignments:
        for annotation_alignment in word_alignment['alignments']:
            if annotation_alignment['__status'] == 'MODIFIED':
                missing_annotations_2 += 1
                extra_annotations_2   += 1
            elif annotation_alignment['__status'] == 'MISSING':
                missing_annotations_2 += 1
            elif annotation_alignment['__status'] == 'EXTRA':
                extra_annotations_2 += 1
    assert missing_annotations == missing_annotations_2
    assert extra_annotations == extra_annotations_2
    return alignments


def get_concise_morph_diff_alignment_str( alignments, layer_a, layer_b, focus_attributes=['root','partofspeech','form'], return_list=False ):
    ''' Formats differences of morph analysis annotations as a string (or a list of strings).'''
    STATUS_ATTR = '__status'
    MATCHING_ATTR    = '__matching'
    MISMATCHING_ATTR = '__mismatching'
    out_str = []
    max_len = max(len(layer_a), len(layer_b))
    max_label_len = max( [len(a) for a in ['MODIFIED', 'MISSING', 'EXTRA', 'COMMON']])
    for alignment in alignments:
        assert STATUS_ATTR      in alignment.keys()
        assert MATCHING_ATTR    in alignment.keys()
        assert MISMATCHING_ATTR in alignment.keys()
        assert layer_a in alignment.keys()
        assert layer_b in alignment.keys()
        if alignment[STATUS_ATTR] == 'MODIFIED':
            focus_is_matching = len([fa for fa in focus_attributes if fa in alignment[MATCHING_ATTR]]) == len(focus_attributes)
            if not focus_is_matching:
                a = [alignment[layer_a][fa] for fa in focus_attributes]
                b = [alignment[layer_b][fa] for fa in focus_attributes]
                out_str.append( (' --- {:'+str(max_label_len)+'} {} ').format(alignment[STATUS_ATTR], '-'*50) )
                out_str.append((' {:'+str(max_len)+'}   ').format(layer_a) + ' '+str(a))
                out_str.append((' {:'+str(max_len)+'}   ').format(layer_b) + ' '+str(b))
            else:
                a = [alignment[layer_a][fa] for fa in focus_attributes+alignment[MISMATCHING_ATTR]]
                b = [alignment[layer_b][fa] for fa in focus_attributes+alignment[MISMATCHING_ATTR]]
                out_str.append( (' --- {:'+str(max_label_len)+'} {} ').format(alignment[STATUS_ATTR], '-'*50) )
                out_str.append((' {:'+str(max_len)+'}   ').format(layer_a) + ' '+str(a))
                out_str.append((' {:'+str(max_len)+'}   ').format(layer_b) + ' '+str(b))
        elif alignment[STATUS_ATTR] == 'COMMON':
            a = [alignment[layer_a][fa] for fa in focus_attributes]
            b = [alignment[layer_b][fa] for fa in focus_attributes]
            out_str.append( (' --- {:'+str(max_label_len)+'} {} ').format(alignment[STATUS_ATTR], '-'*50) )
            out_str.append((' {:'+str(max_len)+'}   ').format(layer_a) + ' '+str(a))
            out_str.append((' {:'+str(max_len)+'}   ').format(layer_b) + ' '+str(b))
        elif alignment[STATUS_ATTR] in ['EXTRA', 'MISSING']:
            a = [alignment[layer_a][fa] for fa in focus_attributes] if len(alignment[layer_a].keys()) > 0 else []
            b = [alignment[layer_b][fa] for fa in focus_attributes] if len(alignment[layer_b].keys()) > 0 else []
            out_str.append( (' --- {:'+str(max_label_len)+'} {} ').format(alignment[STATUS_ATTR], '-'*50) )
            if a:
                out_str.append((' {:'+str(max_len)+'}   ').format(layer_a) + ' '+str(a))
            if b:
                out_str.append((' {:'+str(max_len)+'}   ').format(layer_b) + ' '+str(b))
        else:
            raise Exception( '(!) unexpected __status: {!r}'.format(alignment[STATUS_ATTR]) )
    return '\n'.join( out_str ) if not return_list else out_str


def _text_snippet( text_obj, start, end ):
    '''Takes a snippet out of the text, assuring that text boundaries are not exceeded.'''
    start = 0 if start < 0 else start
    start = len(text_obj.text) if start > len(text_obj.text) else start
    end   = len(text_obj.text) if end > len(text_obj.text)   else end
    end   = 0 if end < 0 else end
    snippet = text_obj.text[start:end]
    snippet = snippet.replace('\n', '\\n')
    return snippet


def format_morph_diffs_string( fname_stub, text_obj, diff_word_alignments, layer_a, layer_b, gap_counter=0, text_cat='',
                                                                           focus_attributes=['root','partofspeech','form'] ):
    '''Formats aligned differences as human-readable text snippets.'''
    #assert layer_a in text_obj.layers, '(!) Layer {!r} missing from: {!r}'.format(layer_a, text_obj.layers)
    #assert layer_b in text_obj.layers, '(!) Layer {!r} missing from: {!r}'.format(layer_b, text_obj.layers)
    N = 60
    output_lines = []
    for word_alignments in diff_word_alignments:
        w_start = word_alignments['start']
        w_end   = word_alignments['end']
        before = '...'+_text_snippet( text_obj, w_start - N, w_start )
        after  = _text_snippet( text_obj, w_end, w_end + N )+'...'
        output_lines.append('='*85)
        output_lines.append('')
        output_lines.append('  '+text_cat+'::'+fname_stub+'::'+f'({str(w_start)}, {str(w_end)})'+'::'+str(gap_counter))
        output_lines.append('')
        output_lines.append( before+' {'+word_alignments['text']+'} '+after  )
        sub_strs = get_concise_morph_diff_alignment_str(word_alignments['alignments'], layer_a, layer_b, \
                                                        return_list=True, focus_attributes=focus_attributes )
        output_lines.extend( sub_strs )
        output_lines.append('')
        gap_counter += 1
    if len(output_lines)>0:
        output_lines.append('')
    return '\n'.join(output_lines) if len(output_lines)>0 else None, gap_counter


def write_formatted_diff_str_to_file( out_fname, output_lines ):
    '''Writes/appends formatted differences to the given file.'''
    if not os.path.exists(out_fname):
        with open(out_fname, 'w', encoding='utf-8') as f:
            pass
    with open(out_fname, 'a', encoding='utf-8') as f:
        ## write content
        f.write(output_lines)
        if not output_lines.endswith('\n'):
            f.write('\n')


class MorphDiffFinder:
    '''Finds all differences between two (Vabamorf's) morphological analysis 
       layers, and groups differences in modified spans in a way that both 
       matching and mismatching annotations are shown.
       Note: output grouped differences only cover modified spans; annotations 
       on non-overlapping spans (missing and extra spans) will be left out.
    '''
    
    def __init__( self, old_layer: str, 
                        new_layer: str,
                        diff_attribs  = ['root', 'lemma', 'root_tokens', 'ending', 'clitic', 'partofspeech', 'form'],
                        focus_attribs = ['root', 'ending', 'clitic', 'partofspeech', 'form'],
                        output_format:  str = 'vertical',
                        flat_layer_suffix: str = '_flat'):
        """Initializes MorphDiffFinder. A specification of comparable layers
           must be provided.
        
           :param old_layer: str
               Name of the old morph_analysis layer.
           :param new_layer: str
               Name of the new morph_analysis layer.
           :param diff_attribs:   list
               List containing morph_analysis attributes which will be used
               for finding difference with DiffTagger. 
               Defaults to ['root', 'lemma', 'root_tokens', 'ending', 'clitic', 
               'partofspeech', 'form'];
           :param focus_attribs:  list
               List containing morph_analysis attributes which values will be
               displayed in the output. 
               Defaults to ['root', 'ending', 'clitic', 'partofspeech', 'form'];
           :param output_format: str
               Whether the differences are aligned in the output string vertically 
               or horizontally.
               Possible values: 'vertical' (default), 'horizontal'.
           :param flat_layer_suffix: str
               Flat layers will be created from comparable morph_analysis 
               layers before the comparison, and this is the suffix that will 
               be added to both flat layers. Defaults to '_flat';
        """
        self._flat_layer_suffix = '_flat'
        self._redux_layer_suffix = '_redux'
        self.old_layer   = old_layer
        self.new_layer   = new_layer
        self.diff_attribs  = diff_attribs
        self.focus_attribs = focus_attribs
        self.morph_diff_tagger = DiffTagger( layer_a = old_layer+self._flat_layer_suffix,
                                             layer_b = self.new_layer+self._flat_layer_suffix,
                                             output_layer='morph_diff_layer',
                                             output_attributes=('span_status', ) + tuple(diff_attribs),
                                             span_status_attribute='span_status' )
        self.output_format = output_format
        self.gap_counter = 0
        self.doc_counter = 0


    def find_difference( self, text, fname, text_cat='', start_new_doc=True ):
        """Finds differences between old layer and new layer in given text, 
           and returns as a tuple (diff_layer, formatted_diffs_str, total_diff_gaps).
        
           :param text: `Text` object
               `Text` object in which differences will be found. Must contain
               `old_layer` and `new_layer`.
           :param fname: str
               Name of the file or document corresponding to the `Text` object.
               The name appears in formatted output (formatted_diffs_str) as 
               a part of the identifier of each difference.
           :param text_cat: str
               Name of the genre or subcorpus where the `Text` object belongs to.
               The name appears in formatted output (formatted_diffs_str) as 
               a part of the identifier of each difference. Defaults to '';
           :param start_new_doc: 
               Whether this `Text` object starts a new document or continues
               an existing document.
               If `True` (default), then it starts a new document and document 
               count will be increased.
           
           :return tuple
               A tuple `(diff_layer, formatted_diffs_str, total_diff_gaps)`:
               * `diff_layer` -- `Layer` of differences created by DiffTagger.
                                 contains differences between old layer and new 
                                 layer.
               * `formatted_diffs_str` -- output string showing grouped differences 
                                          along with their contexts.
                                          Note: these differences only cover modified
                                          spans, annotations on non-overlapping spans
                                          (missing and extra spans) will be left out.
               * `total_diff_gaps` -- integer: total number of grouped differences.
        """
        
        # Check input layers
        assert self.old_layer in text.layers, f'(!) Input text is missing "{self.old_layer}" layer.'
        assert self.new_layer in text.layers, f'(!) Input text is missing "{self.new_layer}" layer.'
        # 1) Create reduced morph analysis layers
        create_redux_morph_layer( text, self.old_layer, self.old_layer + self._redux_layer_suffix, add_layer=True )
        create_redux_morph_layer( text, self.new_layer, self.new_layer + self._redux_layer_suffix, add_layer=True )

        # 2) Create flat v1_6 morph analysis layers
        flat_morph_1 = create_flat_v1_6_morph_analysis_layer( text, self.old_layer + self._redux_layer_suffix, 
                                                                    self.old_layer + self._flat_layer_suffix, 
                                                                    add_layer=False )
        flat_morph_2 = create_flat_v1_6_morph_analysis_layer( text, self.new_layer + self._redux_layer_suffix,
                                                                    self.new_layer + self._flat_layer_suffix, 
                                                                    add_layer=False )
        # 3) Find differences & alignments
        diff_layer = self.morph_diff_tagger.make_layer( text, { self.old_layer + self._flat_layer_suffix : flat_morph_1,
                                                                self.new_layer + self._flat_layer_suffix : flat_morph_2 } )
        ann_diffs = get_estnltk_morph_analysis_diff_annotations( text, flat_morph_1, flat_morph_2, diff_layer )
        flat_morph_layers = [self.old_layer + self._flat_layer_suffix, self.new_layer + self._flat_layer_suffix]
        focus_attributes  = ['root', 'ending', 'clitic', 'partofspeech', 'form']
        alignments = get_estnltk_morph_analysis_annotation_alignments( ann_diffs, flat_morph_layers ,\
                                                                       diff_layer,
                                                                       focus_attributes=self.focus_attribs )
        # 4) Group differences and add context (for better readability)
        formatted_diffs_str, new_morph_diff_gap_counter = \
             format_morph_diffs_string( fname, text, alignments, self.old_layer + self._flat_layer_suffix, \
                                                                 self.new_layer + self._flat_layer_suffix, \
                                                                 gap_counter=self.gap_counter,
                                                                 text_cat=text_cat, \
                                                                 focus_attributes=self.focus_attribs )
        total_diff_gaps = new_morph_diff_gap_counter - self.gap_counter
        self.gap_counter = new_morph_diff_gap_counter
        if start_new_doc:
            self.doc_counter += 1

        # Remove unnecessary layers, back to the original state
        text.pop_layer(name=self.old_layer + self._redux_layer_suffix)
        text.pop_layer(name=self.new_layer + self._redux_layer_suffix)
        text.pop_layer(name=self.new_layer)

        return diff_layer, formatted_diffs_str, total_diff_gaps



class MorphDiffSummarizer:
    '''Aggregates and summarizes morph_analysis span & annotation difference statistics based on information from 
       diff layers. The class is based on  SegmentDiffSummarizer & MorphDiffSummarizer  from: 
          https://github.com/estnltk/eval_experiments_lrec_2020/blob/master/scripts_and_data/segm_eval_utils.py
          https://github.com/estnltk/eval_experiments_lrec_2020/blob/master/scripts_and_data/morph_eval_utils.py
    '''

    def __init__(self, first_model, second_model):
        """Initializes MorphDiffSummarizer.
        
           :param first_model: str
               Name of the first layer that is compared (the old layer).
           :param second_model: str
               Name of the second layer that is compared (the new layer).
        """
        self.diffs_counter = {}
        self.first_model    = first_model
        self.second_model   = second_model
        self.has_span_differences = False
    
    def record_from_diff_layer( self, layer_name, layer, text_category, start_new_doc=True ):
        """Records differences in given document, based on the statistics in metadata of diff_layer.
           
           :param layer_name: str
               Name of the layer which two versions were compared in diff_layer.
           :param diff_layer: `Layer` object
               `Layer` object containing differences between the old layer and 
               the new layer. Must be a layer created by DiffTagger.
           :param text_category: str
               Name of the genre or subcorpus where the given document belongs to.
           :param start_new_doc: bool
               Whether the given document is a new document or a continuation of 
               the previous document.
               If `True` (default), then it starts a new document and document 
               count will be increased. Otherwise, document count is not updated.
        """
        assert isinstance(text_category, str)
        assert len(text_category) > 0
        if layer_name not in self.diffs_counter:
            self.diffs_counter[layer_name] = {}
        if 'total' not in self.diffs_counter[layer_name]:
            self.diffs_counter[layer_name]['total'] = defaultdict(int)
        for key in layer.meta:
            self.diffs_counter[layer_name]['total'][key] += layer.meta[key]
            if key in ['missing_spans', 'extra_spans'] and layer.meta[key] > 0:
                # Span differences encountered ...
                self.has_span_differences = True
        if start_new_doc:
            self.diffs_counter[layer_name]['total']['_docs'] += 1
        if text_category not in self.diffs_counter[layer_name]:
            self.diffs_counter[layer_name][text_category] = defaultdict(int)
        for key in layer.meta:
            self.diffs_counter[layer_name][text_category][key] += layer.meta[key]
        if start_new_doc:
            self.diffs_counter[layer_name][text_category]['_docs'] += 1


    def get_diffs_summary_output( self, show_doc_count=True ):
        """Summarizes aquired difference statistics over subcorpora and over the 
           whole corpus. Returns statistics formatted as a table (string).
           
           :param show_doc_count: bool
               Whether statistics should include the number of documents in each 
               corpus. Defaults to True.
               
           :return str
               statistics formatted as a table (string).
        """
        output = []
        for layer in sorted( self.diffs_counter.keys() ):
            output.append( layer )
            output.append( '\n' )
            diff_categories = [k for k in sorted(self.diffs_counter[layer].keys()) if k != 'total']
            single_category = len(diff_categories) == 1
            assert 'total' in self.diffs_counter[layer]
            diff_categories.append('total')
            longest_cat_name = max( [len(k) for k in diff_categories] )
            #
            # 1) Similarity in spans (if there were any span differences)
            #
            if self.has_span_differences:
                output.append( '\n' )
                output.append( ' span similarity:' )
                output.append( '\n' )
                for category in diff_categories:
                    src = self.diffs_counter[layer][category]
                    if category == 'total' and single_category:
                        # No need to display TOTAL, if there was only one category
                        continue
                    if category == 'total':
                        category = 'TOTAL'
                    output.append( (' {:'+str(longest_cat_name+1)+'}').format(category) )
                    if show_doc_count:
                        output.append('|')
                        output.append(' #docs: {} '.format(src['_docs']) )
                    # unchanged_spans + modified_spans + missing_spans = length_of_old_layer
                    # unchanged_spans + modified_spans + extra_spans = length_of_new_layer
                    first_layer_len  = src['unchanged_spans'] + src['modified_spans'] + src['missing_spans']
                    second_layer_len = src['unchanged_spans'] + src['modified_spans'] + src['extra_spans']
                    total_spans = first_layer_len + second_layer_len
                    # Ratio: https://docs.python.org/3.6/library/difflib.html#difflib.SequenceMatcher.ratio
                    ratio = (src['unchanged_spans']*2.0) / total_spans
                    output.append('|')
                    output.append(' sim ratio: {} / {}  ({:.4f}) '.format(src['unchanged_spans']*2, total_spans, ratio ))
                    missing_percent = (src['missing_spans']/total_spans)*100.0
                    output.append('|')
                    output.append(' only in {}: {} ({:.4f}%) '.format(self.first_model, src['missing_spans'], missing_percent ))
                    extra_percent = (src['extra_spans']/total_spans)*100.0
                    output.append('|')
                    output.append(' only in {}: {} ({:.4f}%) '.format(self.second_model, src['extra_spans'], extra_percent ))
                    output.append('\n')
                output.append('\n')
                output.append( ' annotation similarity:' )
                output.append( '\n' )
            #
            # 2) Similarity in annotations (modified overlapping spans + annotation sim ratio)
            #
            for category in diff_categories:
                src = self.diffs_counter[layer][category]
                if category == 'total' and single_category:
                    # No need to display TOTAL, if there was only one category
                    continue
                if category == 'total':
                    category = 'TOTAL'
                output.append( (' {:'+str(longest_cat_name+1)+'}').format(category) )
                if show_doc_count:
                    output.append('|')
                    output.append(' #docs: {} '.format(src['_docs']) )
                # unchanged_annotations + missing_annotations = number_of_annotations_in_old_layer
                # unchanged_annotations + extra_annotations   = number_of_annotations_in_new_layer
                
                first_layer_len  = src['unchanged_annotations'] + src['missing_annotations']
                second_layer_len = src['unchanged_annotations'] + src['extra_annotations']
                total_spans = first_layer_len + second_layer_len
                output.append('|')
                common_spans = src['unchanged_spans'] + src['modified_spans']
                ratio = src['modified_spans'] / common_spans
                output.append(' modified spans: {} / {} ({:.4f}) '.format(src['modified_spans'], common_spans, ratio ))
                output.append('|')
                # Ratio: https://docs.python.org/3.6/library/difflib.html#difflib.SequenceMatcher.ratio
                ratio = (src['unchanged_annotations']*2.0) / total_spans
                output.append(' annotations ratio: {} / {} ({:.4f}) '.format(src['unchanged_annotations']*2, total_spans, ratio ))
                missing_percent = (src['missing_annotations']/total_spans)*100.0
                output.append('|')
                output.append(' only in {}: {} ({:.4f}%) '.format(self.first_model, src['missing_annotations'], missing_percent ))
                extra_percent = (src['extra_annotations']/total_spans)*100.0
                output.append('|')
                output.append(' only in {}: {} ({:.4f}%) '.format(self.second_model, src['extra_annotations'], extra_percent ))
                output.append('\n')
            output.append('\n')
        return ''.join(output)

