"""Global configuration for EstNLTK model training notebooks and scripts."""

import pathlib
import random
import numpy as np


def _find_project_root(start: pathlib.Path | None = None) -> pathlib.Path:
    """Find project root by walking up until a marker file/folder is found."""
    markers = {"pyproject.toml"}
    current = (start or pathlib.Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if any((candidate / m).exists() for m in markers):
            return candidate
    raise RuntimeError("Project root not found.")


# Configurations
SEED = 42

# Paths
ROOT = _find_project_root()

DATA_DIR = ROOT / "data"
ENC2017_ROOT = DATA_DIR / "enc2017"
UD_ET_EDT_ROOT = DATA_DIR / "ud_et_edt"
HOMONYMS_ROOT = DATA_DIR / "homonymous_word_forms"

ENC2017_DIRS = {
    "processed": ENC2017_ROOT / "processed",
    "raw": ENC2017_ROOT / "raw",
}

UD_ET_EDT_DIRS = {
    "processed": UD_ET_EDT_ROOT / "processed",
    "raw": UD_ET_EDT_ROOT / "raw",
}

HOMONYMS_DIRS = {
    "processed": HOMONYMS_ROOT / "processed",
    "annotations": HOMONYMS_ROOT / "annotations",
}

OUTPUT_DIR = ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
HOMONYMS_PLOTS_DIR = PLOTS_DIR / "homonyms"

MODEL_DIR = ROOT / "models"


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility across random, numpy, and any other libraries used.

    Parameters
    ----------
    seed : int
        The random seed to set.
    """
    global random_seed
    random_seed = seed
    random.seed(seed)
    np.random.seed(seed)
