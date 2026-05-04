# EstNLTK_model_training

This repository contains code and documentation for training a new morphological tagger for Estonian using the EstNLTK library. The goal is to improve the accuracy of morphological analysis for Estonian text, particularly in cases of homonymy and other challenging linguistic phenomena.

## Packaging local scripts

This repository now includes a minimal `pyproject.toml` so local Python modules under `scripts/` can be installed in editable mode:

```bash
pip install -e .
```

After installation, imports such as `from scripts.model.bert_morph_tagger import BertMorphTagger` work without notebook-specific `sys.path.append("../")` hacks.

### Important note for super-repositories

If multiple subprojects in the same Python environment expose a top-level package named `scripts`, import collisions can occur. For strict isolation, prefer one virtual environment per subproject or migrate to a unique top-level package name per project.
