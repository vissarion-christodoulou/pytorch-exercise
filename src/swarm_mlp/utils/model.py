"""The model under training, and the pipeline stages carved out of it.

``SimpleMLP`` is reproduced verbatim from the assignment - the architecture is
mandated, not chosen, and the pipeline split is defined in terms of these exact
attribute names. It lives in its own module because both sides of the project
need it: the single-process reference trains the whole thing, and the workers
each own a contiguous slice of it. One definition, imported twice, is what
keeps the two runs comparable.

This module deliberately imports no hivemind. Stage shapes are plain tuples;
``worker.py`` is what turns them into hivemind schema objects.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def seed_everything(seed: int) -> None:
    """Seed every RNG that can affect weight initialisation.

    Data order is *not* covered here - it comes from an explicit generator in
    ``mnist_train_loader`` - so that the two can be varied independently.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(seed: int, device: str = "cpu") -> SimpleMLP:
    """A freshly initialised model, deterministic in ``seed``.

    Separate from the training loop because the distributed side needs the same
    initial weights: a worker hosts a slice of *this* model, seeded this way, so
    that both runs start from an identical point.

    ``seed`` is required rather than defaulted to ``reference.SEED``: this module
    must not import ``reference``, which imports this one.
    """
    seed_everything(seed)
    return SimpleMLP().to(device)


# Per-stage tensor shapes, batch dimension omitted: (input shape, output shape).
# `worker.py` reads this to build its hivemind schema, and `trainer.py` reads it
# to assert what it is about to send - hivemind validates nested *structure*
# only, never shapes, so a wrong shape would otherwise fail obscurely on the
# far side of an RPC.
STAGE_SHAPES: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "stage0": ((1, 28, 28), (256,)),
    "stage1": ((256,), (10,)),
}


def build_stage(stage: str, seed: int, device: str = "cpu") -> nn.Module:
    """The module a worker for ``stage`` hosts, deterministically initialised.

    ``"stage0"`` and ``"stage1"`` are the two halves of one ``SimpleMLP``,
    carved at the fc1/fc2 boundary:

        stage0:  Flatten -> fc1 -> relu1     (1, 28, 28) -> (256,)
        stage1:  fc2 -> relu2 -> fc3         (256,)      -> (10,)

    Every stage is sliced out of ONE ``build_model(seed)``, so two replicas of
    the same stage constructed with the same seed hold bit-identical weights,
    and the union of stage0 and stage1 is exactly the reference's model. That is
    what makes the replicas' gradients averageable and the curves comparable.

    The boundary was chosen at fc1/fc2 because it is the only split where the
    activation crossing the wire (256 floats per sample) is smaller than the
    input (784), so the pipeline does not cost more bandwidth than it saves.
    """
    if stage not in STAGE_SHAPES:
        raise ValueError(f"unknown stage {stage!r}; known stages: {sorted(STAGE_SHAPES)}")

    # Nothing may consume torch RNG between seeding and construction, or these
    # weights stop matching the reference's.
    model = build_model(seed, device=device)

    if stage == "stage0":
        # nn.Sequential holds references to the very modules `model` built, so
        # these are the reference's parameter tensors, not copies of them.
        return nn.Sequential(model.flatten, model.fc1, model.relu1)
    if stage == "stage1":
        return nn.Sequential(model.fc2, model.relu2, model.fc3)
    raise AssertionError(f"stage {stage!r} is in STAGE_SHAPES but has no constructor")


#: The stages of the pipeline, in the order activations flow through them.
PIPELINE: tuple[str, ...] = ("stage0", "stage1")
