"""Single-process baseline training run.

This is the control for the whole project. The assignment asks that the
distributed system's loss "track and converge similar to the pytorch example",
so everything here exists to make that comparison sharp rather than vague:

* **Dense sampling.** The loss is recorded after every optimiser step, not once
  per epoch. Five points per run cannot distinguish a subtly wrong all-reduce
  from noise; a few thousand can.
* **Samples, not steps, on the x-axis.** Four workers stepping once per group
  of batches do not have a step counter that corresponds to this one. Samples
  consumed is the axis both sides can agree on.
* **Seeded and reproducible.** Same seed, same initial weights, same batch
  order. If the distributed run starts from the same place and follows the same
  data, the curves should nearly coincide - a much stronger test than "both
  go down".

There is deliberately no evaluation pass. Only training loss is compared, and a
test loop would double the runtime while adding a second thing that could
differ between the two implementations.
"""

from __future__ import annotations

import torch.nn as nn
import torch.optim as optim

from swarm_mlp.utils.constants import BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED
from swarm_mlp.utils.curves import LossCurve
from swarm_mlp.utils.data import mnist_train_loader
from swarm_mlp.utils.model import build_model


def train_reference(
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    seed: int = SEED,
    device: str = "cpu",
    log_every: int = 100,
    max_steps: int | None = None,
) -> LossCurve:
    """Train ``SimpleMLP`` on MNIST and return its training loss curve.

    Computes and returns; writes nothing. Progress goes to stdout so a long run
    is not silent, but the curve itself is the return value.

    ``max_steps`` stops after that many recorded steps, so a comparison against
    a short distributed run does not have to pay for a full epoch. ``None``
    means run every epoch to the end.
    """
    model = build_model(seed=seed, device=device)
    loader = mnist_train_loader(batch_size=batch_size, seed=seed)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    curve = LossCurve(
        meta={
            "source": "reference",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
        }
    )
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

            if max_steps is not None and len(curve) >= max_steps:
                return curve

    return curve
