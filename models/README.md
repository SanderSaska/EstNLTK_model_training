# Models Directory

This folder contains trained RoBERTa-based morphological tagger models at different stages of specialization and fine-tuning. Models follow a progressive training strategy, building upon previous checkpoints.

Models are available at University of Tartu's OwnCloud page: https://owncloud.ut.ee/owncloud/s/SQsniWSwx7E9yst

## Model Hierarchy & Progression

```
NER_mudel (Ro-v1)
    ↓
    └─→ NER_mudel_v2 (Ro-v2)
            ↓
            ├─→ NER_mudel_v2_homonym_full (Ro-v2-H)
            │       ├─→ NER_mudel_v2_homonym_full_finetune (Ro-v2-H-FT)
            │
            │
            ├─→ NER_mudel_v2_homonym_80 (Ro-v2-H-80)
            │       └─→ NER_mudel_v2_homonym_80_finetune (Ro-v2-H-80-FT)
            └─→ NER_mudel_v2_muna_1/2 (Older "muna" variants)
```

## Model Descriptions

### NER_mudel/ — Ro-v1 (Baseline)

**General morphology on ENC2017**

- **Training Data**: Estonian National Corpus 2017 (α dataset, ~575,000 tokens)
- **Purpose**: Establish baseline general-purpose morphological tagger
- **Architecture**: Est-RoBERTa + RobertaForTokenClassification
- **Key Results**:
  - ~88% F1 on ENC2017 (general corpus evaluation)
  - ~19% F1 on homonymy test set (specialized task)
- **Characteristics**: Good general performance but struggles with form homonymy
- **Usage**: Starting point for subsequent fine-tuning; reference for measuring improvement

---

### NER_mudel_v2/ — Ro-v2 (Gold-standard Fine-tuning)

**Improved tagging with UD treebank**

- **Training Data**: Ro-v1 checkpoint + UD Estonian Treebank (β dataset, ~437,000 tokens)
- **Purpose**: Improve accuracy using hand-annotated gold-standard data
- **Architecture**: Continues from Ro-v1; training on CoNLL-U format annotations
- **Key Results**:
  - ~92% F1 on UD treebank evaluation (gold standard)
  - ~41% F1 on homonymy test set (still weak on specialization)
  - Shows improvement on general morphology
- **Characteristics**:
  - Reliable base model for general morphological tagging
  - Benefits from high-quality annotations
  - Maintains general knowledge across datasets
- **Usage**: Primary general-purpose model; foundation for Ro-v2-H and Ro-v2-H-80 specialization

---

### NER_mudel_v2_homonym_full/ — Ro-v2-H (Homonymy Specialist)

**Specialized training on form homonymy**

- **Training Data**: Ro-v2 checkpoint + Homonymy corpus (γ dataset, ~7,886 sentences)
- **Purpose**: Achieve high accuracy on morphologically ambiguous word forms
- **Key Results**:
  - ~100% F1 on homonymy corpus (strict evaluation) / ~99% F1 on test set
  - **Issue**: ~52% F1 on UD treebank (catastrophic forgetting)
- **Characteristics**:
  - Excellent homonymy disambiguation
  - Specialized knowledge of form-ambiguous cases
  - Prone to catastrophic forgetting (loses general knowledge)
- **Usage**: Do NOT use alone for general text; use MoE variant or as expert component

---

### NER_mudel_v2_homonym_80/ — Ro-v2-H-80 (Robustness Validation)

**Homonymy training on reduced dataset (80% of corpus)**

- **Training Data**: Ro-v2 checkpoint + 80% subset of homonymy corpus
- **Purpose**: Validate robustness and generalization with less data
- **Key Results**:
  - ~93% F1 on homonymy test set
  - Better stability than Ro-v2-H (less overfitting to corpus)
- **Characteristics**:
  - More conservative specialization
  - Better generalization properties
  - Useful for understanding how much homonymy data is necessary
- **Usage**: Alternative to Ro-v2-H for production; lighter-weight expert model

---

### NER_mudel_v2_homonym_full_finetune/ — Ro-v2-H-FT (Catastrophic Forgetting Mitigation)

**Fine-tuning to recover general knowledge**

- **Training Data**: Ro-v2-H checkpoint + UD treebank (β dataset, fine-tuning phase)
- **Purpose**: Recover lost general morphological knowledge after specialization
- **Key Results**:
  - ~91% F1 on UD treebank (recovered general knowledge)
  - Retains strong homonymy performance
- **Characteristics**:
  - Two-stage training: specialize then recover
  - Demonstrates feasibility of sequential training
  - Less sophisticated than MoE approach
- **Usage**: Alternative approach to MoE if ensemble complexity undesirable; full-model fine-tuning

---

### NER_mudel_v2_homonym_80_finetune/ — Ro-v2-H-80-FT (Robust + Fine-tuned)

**Combining robustness with fine-tuning recovery**

- **Training Data**: Ro-v2-H-80 checkpoint + UD treebank (fine-tuning phase)
- **Purpose**: Best balance of robustness and knowledge retention
- **Key Results**:
  - ~91% F1 on UD treebank
  - ~63% F1 on homonymy test set (reasonable compromise)
- **Characteristics**:
  - Most balanced model across datasets
  - Benefits from both robustness (80%) and fine-tuning recovery
  - Good generalization
- **Usage**: Recommended for production when MoE unavailable; balanced single model

---

### NER_mudel_v2_muna_1/, NER_mudel_v2_muna_2/ — Ro-v2-MoE (Mixture of Experts)

- **Architecture**:
  - **Base Model**: Ro-v2 (general-purpose)

- **Training Data**:
  - Components: ["muna" homonymy examples](https://github.com/estnltk/estnltk-model-training/tree/main/morph_tagging/experiment_with_homonym_muna)
  - Router: Curated list of form-ambiguous words in Estonian

- **Characteristics**:
  - Trained only on "muna" examples. First test to see if model can distinguish form homonymy cases.

## Model Performance Summary

| Model         | General Corpus (UD) | Homonymy    | Recommendation                |
| ------------- | ------------------- | ----------- | ----------------------------- |
| Ro-v1         | 88% F1              | 19% F1      | Baseline only                 |
| Ro-v2         | 92% F1              | 41% F1      | General-purpose tagging       |
| Ro-v2-H       | 52% F1              | 100% F1     | Specialist only (with caveat) |
| Ro-v2-H-80    | 68% F1              | 93% F1      | Alternative specialist        |
| Ro-v2-H-FT    | 91% F1              | Good        | Single-model alternative      |
| Ro-v2-H-80-FT | 91% F1              | 63% F1      | Balanced compromise           |
| **Ro-v2-MoE** | **92% F1**          | **100% F1** | ✅ Production recommended     |

## Loading and Using Models

### Basic Loading

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load general-purpose model
tokenizer = AutoTokenizer.from_pretrained("e:/Git_projects/EstNLTK/EstNLTK_model_training/models/NER_mudel_v2")
model = AutoModelForTokenClassification.from_pretrained("e:/Git_projects/EstNLTK/EstNLTK_model_training/models/NER_mudel_v2")

# Load MoE variant (if implemented as separate checkpoint)
moe_model = AutoModelForTokenClassification.from_pretrained("e:/Git_projects/EstNLTK/EstNLTK_model_training/models/NER_mudel_v2_muna_1")
```

## Model Checkpoints

All models include:

- `config.json` — Model architecture configuration
- `pytorch_model.bin` — Model weights (or `model.safetensors`)
- `tokenizer.json` — Tokenizer definition
- `special_tokens_map.json` — Special token mappings

Checkpoints are compatible with Hugging Face `transformers` library (v4.20+).
