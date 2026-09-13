"""The loss curve both training implementations produce.

Lives in its own module because it is the one thing the reference run and the
distributed run have to agree about. Everything else differs between them - one
is a for-loop, the other is five processes and a DHT - but both emit the same
three parallel series, and the comparison in ``swarm_mlp.utils.compare`` is only
meaningful because the shape is shared rather than reimplemented twice.

Persistence is deliberately kept out of the training functions. ``train_reference``
and ``trainer.train`` compute and return a curve and write nothing; only the CLIs
call ``save``. The reason the file format exists at all is that the distributed
curve is produced *inside a trainer process*, so it has to cross a process
boundary to reach the comparison script - and a JSON file is the smallest thing
that does that honestly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LossCurve:
    """A training run's loss against the number of samples consumed.

    The x-axis is *samples*, not optimiser steps. Four workers stepping once per
    group of batches have no step counter that corresponds to the reference's;
    samples consumed is the axis both sides can agree on.
    """

    samples: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)

    # Free-form, but producers populate at least source/epochs/batch_size/seed/
    # learning_rate so that a comparison can reconstruct the matching reference
    # run from the distributed curve alone, rather than by convention.
    meta: dict[str, Any] = field(default_factory=dict)

    def record(self, samples: int, loss: float, accuracy: float) -> None:
        self.samples.append(samples)
        self.loss.append(loss)
        self.accuracy.append(accuracy)

    def __len__(self) -> int:
        return len(self.samples)

    def save(self, path: Path | str) -> Path:
        """Write the curve as JSON, creating parent directories as needed.

        Floats go through ``json`` unrounded: Python's float repr is the
        shortest string that round-trips exactly, so ``load(save(c))`` returns
        bit-identical values. Rounding here would silently cap the precision of
        every later comparison.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "samples": self.samples,
                    "loss": self.loss,
                    "accuracy": self.accuracy,
                    "meta": self.meta,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> LossCurve:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            samples=payload["samples"],
            loss=payload["loss"],
            accuracy=payload["accuracy"],
            meta=payload.get("meta", {}),
        )
