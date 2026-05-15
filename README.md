# EstNLTK_model_training

This repository contains code and documentation for training a RoBERTa-based morphological tagger for Estonian using the EstNLTK library. The goal is to improve the accuracy of morphological analysis for Estonian text, particularly in cases of form homonymy and other challenging linguistic phenomena. This work is part of a master's thesis on neural morphological tagging for Estonian.

## Directory Structure

```
EstNLTK_model_training/
├── data/                         # Training and evaluation datasets
│   ├── enc2017/                    # Estonian National Corpus 2017 (α dataset)
│   ├── ud_et_edt/                  # UD Estonian treebank (β dataset)
│   ├── vormihomonüümia/            # Word form homonymy corpus (γ dataset)
│   └── conflicts/                  # Morphology-syntax conflict examples
├── models/                       # Trained model checkpoints
│   ├── NER_mudel/                  # Named entity recognition models
│   ├── NER_mudel_v2/               # NER v2 variants
│   └── NER_mudel_v2_homonym_*/     # Homonymy-specialized models
├── notebooks/                    # Jupyter notebooks for development
│   ├── data_preprocessing.ipynb    # Data cleaning and preparation
│   ├── model_training.ipynb        # RoBERTa model training
│   ├── evaluation.ipynb            # Model evaluation and metrics
│   ├── model_smoke_testing.ipynb   # Quick model validation
│   ├── MLM.ipynb                   # Masked language modeling experiments
│   └── testing.ipynb               # Additional test notebooks
├── scripts/                      # Reusable Python modules
│   ├── DepChainTagger.py           # Dependency chain tagging
│   ├── config.py                   # Configuration and constants
│   ├── preprocessing/              # Data preprocessing utilities
│   └── model/                      # Model implementations
├── outputs/                      # Evaluation results and artifacts
│   ├── evaluation_results.json     # Aggregated metrics
│   ├── unique_labels.json          # Label vocabulary
│   ├── plots/                      # Visualization plots
│   └── diffs/                      # Data difference logs
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Package configuration
└── todo.md                       # Active tasks and experiment ideas
```

## Quick Start / Setup

### Prerequisites

- Python 3.8+
- Virtual environment recommended

### Installation

1. Clone the repository and navigate to it:

```bash
cd EstNLTK_model_training
```

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

**GPU Note:** The `requirements.txt` includes PyTorch with CUDA 13.0 support (`cu130`). If you don't have an NVIDIA GPU or prefer CPU-only execution, remove the `cu130` specifier in the `torch` lines before installing:

```bash
# Replace torch[cuda13.0] with plain torch in requirements.txt, then:
pip install -r requirements.txt
```

1. Install the local package in editable mode:

```bash
pip install -e .
```

After installation, imports such as `from scripts.model.bert_morph_tagger import BertMorphTagger` will work without path hacks.

## Workflows

### Data Preparation

1. **Load raw data** (corpus files, annotations)
2. **Run `notebooks/data_preprocessing.ipynb`** to:
   - Parse corpus formats (JSON, conllu, etc.)
   - Extract morphological annotations
   - Generate parquet files for efficient training
   - Split into train/validation/test sets

### Model Training

1. **Configure training** in `notebooks/model_training.ipynb`:
   - Select base model (Est-RoBERTa)
   - Choose dataset(s) to train on
   - Set hyperparameters (learning rate, batch size, epochs)

2. **Run training**:
   - Models are saved to `models/` directory with versioned checkpoints
   - Validation metrics logged after each epoch

3. **Save trained models**

### Model Evaluation

1. **Run `notebooks/evaluation.ipynb`** to:
   - Load trained models and test datasets
   - Compute classification metrics (Accuracy, Precision, Recall, F1)
   - Generate confusion matrices by morphological features
   - Visualize results as plots (saved to `outputs/plots/`)

2. **Analyze results**:
   - Macro-averaged metrics for imbalanced classes
   - Three evaluation strategies (Range, Keskmine, Leebe) for homonymy
   - Per-class performance breakdowns

### Alternative Approaches

- **MLM Experiments** (`notebooks/MLM.ipynb`): Masked language modeling with BERT/LLM for candidate generation. Large language model as auxiliary knowledge source

## Model Variants

All models are based on Est-RoBERTa (RoBERTa pretrained on Estonian text). Training follows a progressive strategy:

| Model ID          | Training Data                                       | Purpose                                        | Key Results                                       |
| ----------------- | --------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------- |
| **Ro-v1**         | ENC2017 (α)                                         | General morphology, mimics Vabamorf            | ~88% F1 on ENC2017, ~19% F1 on homonymy           |
| **Ro-v2**         | Ro-v1 + UD treebank (β)                             | Improved general tagging with gold annotations | ~92% F1 on treebank, ~41% F1 on homonymy          |
| **Ro-v2-H**       | Ro-v2 + Homonymy corpus (γ)                         | Specialized for form homonymy                  | ~100% F1 on homonymy (strict), ~99% F1 (test set) |
| **Ro-v2-H-80**    | Ro-v2 + 80% of homonymy corpus                      | Robustness validation on reduced data          | ~93% F1 on homonymy (test)                        |
| **Ro-v2-H-FT**    | Ro-v2-H + fine-tuning on UD                         | Mitigation of catastrophic forgetting          | ~91% F1 on treebank (retained knowledge)          |
| **Ro-v2-H-80-FT** | Ro-v2-H-80 + fine-tuning on UD                      | Robustness validation with fine-tuning         | ~91% F1 on treebank, ~63% F1 on homonymy (test)   |
| **Ro-v2-MoE**     | Mixture of Experts: Ro-v2 (base) + Ro-v2-H (expert) | Routing mechanism for specialized handling     | ~100% F1 on homonymy, ~92% F1 on treebank         |

### Evaluation Strategies for Homonymy

- **Range (R)**: Strict evaluation of full morphological tag (number + form)
- **Keskmine (K)**: Normalized class space (illative/allative → short illative; rare forms → "other")
- **Leebe (L)**: Lenient evaluation, focused on target forms only (excludes "other" class)

See `notebooks/comparison` for detailed metric computation and visualization for each strategy and model variant on the homonymy corpus (γ).
