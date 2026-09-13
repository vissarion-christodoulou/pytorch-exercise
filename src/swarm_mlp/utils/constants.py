"""Values that more than one module has to agree on.

Two kinds of thing live here, and both are here for the same reason: a silent
disagreement between two copies of them is indistinguishable from a bug.

The training defaults are shared by the baseline and the distributed system.
If the worker's learning rate drifted from the reference's, the loss curves
would separate and every explanation - wrong gradient weighting, a broken
all-reduce, a bad split - would look more plausible than the real one.

The paths are resolved from this file rather than from the working directory,
because workers and trainers are launched from wherever the operator happens to
be standing. They were previously recomputed in four separate modules, which is
three chances to get the directory depth wrong.
"""

from __future__ import annotations

from pathlib import Path

#: The repository root: <root>/src/swarm_mlp/utils/constants.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: MNIST downloads land here; .gitignore already excludes it.
DATA_ROOT = PROJECT_ROOT / "data"

#: Curves and plots are written here.
RESULTS_DIR = PROJECT_ROOT / "results"

EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SEED = 0
