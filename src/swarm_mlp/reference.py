"""Single-process baseline training run.

This is the control for the whole project. The assignment asks that the
distributed system's loss "track and converge similar to the pytorch example",
so everything here exists to make that comparison sharp rather than vague:

* **Dense sampling.** The loss is recorded after every optimiser step, not once
  per epoch. Five points per run cannot distinguish a subtly wrong all-reduce
  from noise; a few thousand can.
* **Samples, not steps, on the x-axis.** Four workers stepping on a shared
  target-batch-size trigger do not have a step counter that corresponds to this
  one. Samples consumed is the axis both sides can agree on.
* **Seeded and reproducible.** Same seed, same initial weights, same batch
  order. If the distributed run starts from the same place and follows the same
  data, the curves should nearly coincide - a much stronger test than "both
  go down".

There is deliberately no evaluation pass. Only training loss is compared, and a
test loop would double the runtime while adding a second thing that could
differ between the two implementations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from swarm_mlp.data import mnist_train_loader
from swarm_mlp.model import SimpleMLP

# Defaults shared by whatever calls this. When the distributed trainer lands it
# should import these rather than restate them; a silent disagreement about the
# learning rate would look exactly like a bug in the gradient averaging.
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SEED = 0


@dataclass
class LossCurve:
    """A training run's loss against the number of samples consumed.

    Held in memory and returned rather than written to disk: the comparison
    script calls the producing function and plots what comes back.
    """

    samples: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)

    def record(self, samples: int, loss: float, accuracy: float) -> None:
        self.samples.append(samples)
        self.loss.append(loss)
        self.accuracy.append(accuracy)

    def __len__(self) -> int:
        return len(self.samples)


def seed_everything(seed: int) -> None:
    """Seed every RNG that can affect weight initialisation.

    Data order is *not* covered here - it comes from an explicit generator in
    ``mnist_train_loader`` - so that the two can be varied independently.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(seed: int = SEED, device: str = "cpu") -> SimpleMLP:
    """A freshly initialised model, deterministic in ``seed``.

    Separate from the training loop because the distributed side will need the
    same initial weights: workers hold slices of *this* model, seeded this way,
    so that both runs start from an identical point.
    """
    seed_everything(seed)
    return SimpleMLP().to(device)


def train_reference(
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    seed: int = SEED,
    device: str = "cpu",
    log_every: int = 100,
) -> LossCurve:
    """Train ``SimpleMLP`` on MNIST and return its training loss curve.

    Computes and returns; writes nothing. Progress goes to stdout so a long run
    is not silent, but the curve itself is the return value.
    """
    model = build_model(seed=seed, device=device)
    loader = mnist_train_loader(batch_size=batch_size, seed=seed)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    curve = LossCurve()
    samples_seen = 0

    model.train()
    for epoch in range(epochs):
        for step, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            samples_seen += labels.size(0)

            # Batch accuracy is noisy, but it is nearly free from logits we
            # already have and it separates "the loss is falling" from "the
            # model is learning" - a distinction worth having when the
            # distributed version starts misbehaving.
            batch_accuracy = outputs.argmax(dim=1).eq(labels).float().mean().item()
            curve.record(samples_seen, loss.item(), batch_accuracy)

            if log_every and step % log_every == 0:
                print(
                    f"    epoch {epoch + 1}/{epochs}  step {step:>4}  "
                    f"samples {samples_seen:>6}  loss {loss.item():.4f}  "
                    f"acc {batch_accuracy:.3f}"
                )

    return curve
