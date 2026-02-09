#
#   Utilities for processing Estonian UD corpus
#   ( https://github.com/UniversalDependencies/UD_Estonian-EDT )
#
#   * load_ud_file_with_corrections(fnm):
#          Loads Estonian UD corpus' file with post-corrections 
#          that improve comparability to VM's annotations. 
#          Returns the Text object.
#
#   * load_ud_file_texts_with_corrections(fnm):
#          Loads Estonian UD corpus' file with post-corrections that 
#          improve comparability to VM's annotations. 
#          Returns a list of Text objects.
# 
#   Repository originated from: https://github.com/estnltk/estnltk-model-training/tree/main/ud_morph_tools/amb_morph_reordering
# 

from enum import Enum
import os, os.path
import re

from collections import OrderedDict
from collections import defaultdict

from estnltk import Layer, Text
from estnltk.taggers import Tagger
from estnltk.taggers import CompoundTokenTagger
from estnltk_core.layer_operations import rebase, flatten
from estnltk.converters.conll.conll_importer import conll_to_text
from estnltk.converters.conll.conll_importer import conll_to_texts_list

# ===========================================================================
#   Importing Text objects from EWTB corpus
#   (Note: this operates on the UD version of the corpus)
# ===========================================================================

class VerbLemmaEndingType(Enum):
    '''How verb lemma endings are represented in different annotation formats.'''
    UD_ENDING = 1         # verb lemmas end with: -ma/-nud/-tud/-dud
    VABAMORF_ENDING = 2   # verb lemmas do not have -ma/-nud/-tud/-dud endings
    HFST_ENDING = 3       # verb lemmas have only -ma endings

class EstUDCorrectionsRewriter:
    '''Provides corrections to UD annotations in order to make them comparable to Vabamorf's ones.'''
    
    verb_endings = re.compile('^(.+)(ma|nud|tud|dud)$')

    def __init__(self, apply_linguistic_fixes: bool=True, \
                       verb_ending_type: VerbLemmaEndingType=VerbLemmaEndingType.VABAMORF_ENDING):
        self.apply_linguistic_fixes = apply_linguistic_fixes
        self.verb_ending_type = verb_ending_type
    
    def rewrite(self, record):
        # Attributes:
        #   id, lemma, upostag, xpostag, feats, head, deprel, deps, misc, parent_span, children, text
        #assert 'text' in record, str(record)
        # 1) 'feats' should always be available, even if empty
        if record['feats'] is None:
            record['feats'] = OrderedDict()
        # 2.1) Rewrite verbs by removing ending -ma, -nud, -tud, -dud
        #    ( so that we can match against Vabamorf's verbs )
        xpostag = record['xpostag']
        upostag = record['upostag']
        if xpostag == 'V':
            if self.verb_ending_type == VerbLemmaEndingType.VABAMORF_ENDING:
                # Vabamorf endings: delete -ma/-nud/-tud/-dud
                record['lemma'] = self.verb_endings.sub('\\1', record['lemma'])
            elif self.verb_ending_type == VerbLemmaEndingType.HFST_ENDING:
                # HFST ending: replace every ending with -ma
                record['lemma'] = self.verb_endings.sub('\\1ma', record['lemma'])
            else:
                # UD ending: keep everything as it was
                pass
        # 2.2) If xpostag == '_', then add it based on upostag 
        if xpostag is None:
            if upostag == 'CCONJ':
                record['xpostag'] = 'J'
                xpostag = record['xpostag']
            elif upostag == 'PRON':
                record['xpostag'] = 'P'
                xpostag = record['xpostag']
            elif upostag == 'AUX' and record['lemma'] == 'olema':
                record['xpostag'] = 'V'
                xpostag = record['xpostag']
            elif upostag == 'NOUN' or upostag == 'SYM' and record['lemma'] == '%': # Temp. fix for AssertionError while reading in Estonian UD EDT treebank
                record['xpostag'] = 'S'
                xpostag = record['xpostag']
            elif upostag == 'PUNCT':
                record['xpostag'] = 'Z'
                xpostag = record['xpostag']
            elif upostag == 'X': # Temp. fix for AssertionError while reading in Estonian UD EDT treebank
                record['xpostag'] = 'N'
                xpostag = record['xpostag']
            # else: # Temp. fix for AssertionError
            #     record['xpostag'] = upostag
            #     xpostag = record['xpostag']
        # print(record) # Debugging
        assert xpostag is not None, ' (!) xpostag missing in {!r} '.format(record)
        if self.apply_linguistic_fixes:
            # 3) Fix ordinal numbers that were marked as adjectives ...
            num_type = record['feats'].get('NumType', None)
            num_form = record['feats'].get('NumForm', None)
            case     = record['feats'].get('Case', None)
            number   = record['feats'].get('Number', None)
            if xpostag == 'N' and upostag == 'ADJ' and num_form == 'Digit' and num_type == 'Ord':
                record['xpostag'] = 'O'
                record['upostag'] = 'NUM'
                xpostag = record['xpostag']
                upostag = record['upostag']
            if xpostag == 'N' and upostag == 'ADJ' and num_form == None and num_type == 'Ord':
                record['xpostag'] = 'O'
                record['upostag'] = 'NUM'
                xpostag = record['xpostag']
                upostag = record['upostag']
            # We also need to fix some very specific stuff to get the conversion working
            if xpostag == 'V' and upostag == 'NOUN' and record['lemma'] == 'lihtsus':
                record['xpostag'] = 'S'
        return record


from estnltk.taggers import AnnotationRewriter

class EstUDCorrectionsRetagger(AnnotationRewriter):
    """Applies EstUDCorrectionsRewriter.
    """
    def __init__(self, layer_name, apply_linguistic_fixes=True, \
                                   verb_ending_type: VerbLemmaEndingType=VerbLemmaEndingType.VABAMORF_ENDING):
        rewrite = EstUDCorrectionsRewriter(apply_linguistic_fixes=apply_linguistic_fixes, verb_ending_type=verb_ending_type).rewrite

        def function(annotation):
            return rewrite(annotation)

        super().__init__(layer_name=layer_name, output_attributes=[], function=function)

# ===========================================================================
#   Importing Text objects from ETB corpus, ver 1:
#      Load all documents into a single Text object
# ===========================================================================

def load_ud_file_with_corrections( fnm, annotation_layer, 
                                        apply_linguistic_fixes=True, 
                                        add_compound_tokens=True, 
                                        verb_ending_type:VerbLemmaEndingType=VerbLemmaEndingType.VABAMORF_ENDING ):
    '''Loads Estonian UD corpus' file with post-corrections that improve comparability to VM's annotations. Returns the Text object. '''
    # Load a text with conll annotations layer
    text = conll_to_text(file=fnm, syntax_layer=annotation_layer)
    # Rewrite layer with postcorrections
    EstUDCorrectionsRetagger(annotation_layer,
                             apply_linguistic_fixes=apply_linguistic_fixes, 
                             verb_ending_type=verb_ending_type).retag( text )
    # Add compound tokens layer (if required)
    if add_compound_tokens and 'compound_tokens' not in text.layers.keys():
        add_empty_compound_tokens_layer( text )
    return text


def add_empty_compound_tokens_layer( text ):
    '''Adds an empty compound tokens layer to the Text object. This is required for morph analysis.'''
    compound_tokens = \
           Layer(name=CompoundTokenTagger.output_layer, \
                 attributes=CompoundTokenTagger.output_attributes, \
                 text_object=text,\
                 ambiguous=False)
    text.add_layer(compound_tokens)

# ===========================================================================
#   Importing Text objects from ETB corpus, ver 2:
#      Separate documents according to the metadata
# ===========================================================================

def load_ud_file_texts_with_corrections( fnm, annotation_layer, 
                                              apply_linguistic_fixes=True, 
                                              add_compound_tokens=True, 
                                              verb_ending_type:VerbLemmaEndingType=VerbLemmaEndingType.VABAMORF_ENDING ):
    '''Loads Estonian UD corpus' file with post-corrections that improve comparability to VM's annotations. Returns a list of Text objects. '''
    # Load texts with conllu layers
    texts = conll_to_texts_list(file=fnm, syntax_layer=annotation_layer)
    # Rewrite layer with postcorrections
    fixer = EstUDCorrectionsRetagger(annotation_layer,apply_linguistic_fixes=apply_linguistic_fixes,verb_ending_type=verb_ending_type)
    for text in texts:
        fixer.retag( text )
        # Add compound tokens layer (if required)
        if add_compound_tokens and 'compound_tokens' not in text.layers:
            add_empty_compound_tokens_layer( text )
    return texts

