## Manual TODO list for EstNLTK model training

### Confusion matrix, random baseline score (most frequent class or uniform random)

Weighted training...
Train model on the whole dataset and try to get 100 score

How to handle different weights of different data points?

Dashboard

Test sets:

- Vabamorf and BertMorphTagger comparison test set;
- UD treebank test set, enc2017 test set, homonyms test set --- score for each dataset separately
- Gather test set from morphological and syntax different cases in Estonian using GPT to filter these sentences.
- more...

Train set on all 3 datasets

Evaluation on test set score should be higher than in train set

First train on the existing BertMorphTagger model. If time permits, try to train from scratch.

Build train set:

- 3 train sets: UD treebank test set, enc2017 test set, homonyms test set

GPT experiments:

- Morphological analysis with GPT on homonyms test set. Replace the labelled word with "see" (eluta) or "tema"/name (elus) (Find a name that has different writings in different cases) OR replace with synonym. Find or search for alternative methods to augment data.
- Test GPT on 10 sample sentences from homonyms test set and see if the score is over 90%.
- Use few-shot prompting.
- Try to limit API calls to 10€ or even less for GPT.

### Background research:

Millistes keeltes käänte homonüümia üldse esineb?
How to use LLM (GPT) to predict cases for words? Morphological analysis. Search for existing research or examples using LLM based tools.,
Catastrophic forgetting literature review: does it apply to this dashboard case?

## Automatic TODO list for EstNLTK model training
