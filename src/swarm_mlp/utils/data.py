"""MNIST input pipeline.

Shared, like the model, because comparability is the whole point of the
reference run: if the reference and the distributed trainer ever disagree about
the transform or the shuffling order, the two loss curves stop being evidence
about the distributed system and start being evidence about the data pipeline.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from swarm_mlp.utils.constants import DATA_ROOT

# ToTensor() and nothing else, matching the reference article: pixels land in
# [0, 1] and are not standardised. The usual Normalize((0.1307,), (0.3081,))
# would train slightly faster, but the assignment compares two loss curves to
# each other rather than to any published number, so the simpler pipeline is
# one less thing that has to agree across five processes.
TRANSFORM = transforms.ToTensor()


def mnist_train_dataset(data_root: Path | str | None = None) -> datasets.MNIST:
    """The 60k-image training split, downloading it on first use."""
    return datasets.MNIST(
        root=str(data_root or DATA_ROOT),
        train=True,
        transform=TRANSFORM,
        download=True,
    )


def mnist_train_loader(
    *,
    batch_size: int,
    seed: int,
    data_root: Path | str | None = None,
) -> DataLoader:
    """A shuffled training loader whose batch order is a function of ``seed``.

    The generator is explicit rather than relying on the global RNG: the
    distributed trainer will seed model initialisation and data order
    separately, and sharing one global stream between them would make the data
    order depend on how many parameters happened to be initialised first.

    ``num_workers=0`` keeps loading in-process. MNIST is small enough that
    worker processes buy nothing, and later on the trainer runs underneath
    hivemind, which forks liberally on its own account.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset=mnist_train_dataset(data_root),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
