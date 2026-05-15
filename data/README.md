# Data Folder Documentation

This directory contains all datasets used for training and evaluating the morphological tagger. Three main corpora (α, β, γ) form the basis of the experiments, along with supporting data for conflict analysis.

```
data/                           # Folder containing all datasets
├── enc2017/                    # Estonian National Corpus 2017 (α dataset)
├── ud_et_edt/                  # UD Estonian treebank (β dataset)
├── vormihomonüümia/            # Word form homonymy corpus (γ dataset)
└── conflicts/                  # Morphology-syntax conflict examples
```

Data is available at University of Tartu's OwnCloud page: https://owncloud.ut.ee/owncloud/s/SQsniWSwx7E9yst

## Datasets Overview

### enc2017/ (Dataset α)

**Estonian National Corpus 2017 subset**

- **Purpose**: Training data for general morphological tagging
- **Size**: ~575,000 words
- **Format**: Parquet files with columns: `text`, `token_id`, `morphological_tags`
- **Characteristics**:
  - Balanced subcorpus from multiple genres (news, fiction, technical)
  - Automatically annotated with Vabamorf baseline
  - Used as starting point for model training (Ro-v1)
  - Contains diverse morphological phenomena but not specifically balanced for homonymy
- **File Types**:
  - `*.parquet`: Processed data in Apache Parquet format for efficient training
  - `*.csv`: (Optional) CSV format for manual inspection
- **Usage**: Load in `notebooks/model_training.ipynb` for Ro-v1 baseline training

### ud_et_edt/ (Dataset β)

**Universal Dependencies Estonian Treebank (UD_Estonian-EDT)**

- **Purpose**: Gold-standard evaluation and fine-tuning data
- **Size**: ~437,000 words (hand-annotated)
- **Format**:
  - `*.conllu`: CoNLL-U format (standard UD format)
  - `*.parquet`: Converted to parquet for training pipeline
- **Characteristics**:
  - High-quality hand-annotated morphological and syntactic annotations
  - Gold-standard accuracy: designed for evaluation
  - Covers diverse sentence structures and linguistic phenomena
  - More reliable for validating model generalization
- **Usage**:
  - Primary evaluation dataset (β)
  - Training data for Ro-v2
  - Fine-tuning after homonymy specialization to mitigate catastrophic forgetting
  - Used to validate Ro-v2-H-FT and MoE variants

### vormihomonüümia/ (Dataset γ)

**Word Form Homonymy Corpus**

- **Purpose**: Specialized training and evaluation for form homonymy challenges
- **Size**: ~7,886 sentences with ~8,000 word form homonymy examples
- **Format**: JSON or parquet with homonymy context
- **Characteristics**:
  - Carefully curated examples where same surface form has multiple morphological analyses
  - Example: Estonian "muna" (nom., gen., part. singular all have identical form)
  - Context includes surrounding words for disambiguation
  - Balanced distribution across inflection types, but not balanced by case.
- **Structure**: Typically contains:
  - `sentence`: Full sentence with homonymous form highlighted
  - `target_word`: The ambiguous word form
  - `label`: The intended morphological tag
- **Usage**:
  - Training data for Ro-v2-H (homonymy-specialized model)
  - Evaluation of model performance on ambiguous cases
  - Validation of Ro-v2-MoE expert component

### conflicts/ (Supplementary)

**Morphology-Syntax Conflict Examples**

- **Purpose**: Analysis of cases where morphological and syntactic analysis disagree
- **Content**:
  - Sentences where Vabamorf morphological analysis conflicts with syntax trees
  - Examples of morphological ambiguity that cannot be resolved by form alone
  - Cases requiring syntactic context for correct disambiguation
- **Usage**:
  - Research on limitations of morphology-only approaches
  - Motivation for incorporating syntactic information (DepChainTagger experiments)
  - Understanding where rule-based taggers fail

## Data Processing Pipeline

### Format Conversions

1. **Raw corpus** (various formats) → **Parquet**
   - Run `notebooks/data_preprocessing.ipynb`
   - Converts CONLL-U, JSON, CSV to optimized parquet format
   - Splits into train/validation/test sets

2. **Parquet Features**:
   - Efficient storage and fast loading during training
   - Preserves all morphological feature information
   - Includes train/val/test splits for reproducibility

## Dataset Statistics

| Dataset             | Tokens   | Sentences | Morphological Tags |
| ------------------- | -------- | --------- | ------------------ |
| enc2017 (α)         | ~576,000 | 39,769    | Variable           |
| ud_et_edt (β)       | ~428,000 | 30,515    | Gold-standard      |
| vormihomonüümia (γ) | ~180,000 | 7,886     | Focused            |
