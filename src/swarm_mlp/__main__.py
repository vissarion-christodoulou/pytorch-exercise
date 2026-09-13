"""CLI for the single-process reference run: train, then plot the loss curve.

Kept deliberately thin. All it does is turn command-line arguments into a call
to ``train_reference`` and render what comes back, so that the comparison script
can obtain the same curve by importing the function - without dragging argparse
and matplotlib along with it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from swarm_mlp.baseline_reference.reference import train_reference
from swarm_mlp.utils.constants import BATCH_SIZE, EPOCHS, LEARNING_RATE, RESULTS_DIR, SEED
from swarm_mlp.utils.plotting import plot_curve


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="swarm-mlp",
        description="Train the reference SimpleMLP on MNIST and plot its loss curve.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "reference_loss.png",
        help="where to write the plot (default: results/reference_loss.png)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=50,
        help="moving-average window, in optimiser steps (default: 50)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    print(
        f"Reference run: {args.epochs} epochs, batch {args.batch_size}, "
        f"lr {args.learning_rate}, seed {args.seed}"
    )
    curve = train_reference(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    tail = min(len(curve), args.smooth)
    print(
        f"\n{len(curve)} steps, {curve.samples[-1]} samples consumed. "
        f"Final {tail}-step mean: loss {np.mean(curve.loss[-tail:]):.4f}, "
        f"accuracy {np.mean(curve.accuracy[-tail:]):.4f}"
    )

    written = plot_curve(curve, args.output, smooth=args.smooth)
    print(f"Plot written to {written}")


if __name__ == "__main__":
    main()
