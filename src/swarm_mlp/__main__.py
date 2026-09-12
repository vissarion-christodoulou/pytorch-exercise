"""CLI for the single-process reference run: train, then plot the loss curve.

Kept deliberately thin. All it does is turn command-line arguments into a call
to ``train_reference`` and render what comes back, so that the comparison script
can obtain the same curve by importing the function - without dragging argparse
and matplotlib along with it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Select the non-interactive backend before pyplot is imported. There is no
# display inside WSL2, so this is the only thing that can work; saying so
# explicitly beats relying on matplotlib's fallback and then wondering why
# plt.show() does nothing.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  - must follow matplotlib.use
import numpy as np  # noqa: E402

from swarm_mlp.reference import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    LossCurve,
    train_reference,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def rolling_mean(values: list[float], window: int) -> np.ndarray:
    """Centred moving average, for readability on top of the raw series.

    Per-step loss on a batch of 64 is dominated by which digits happened to be
    in the batch. The smoothed line is what you actually compare between runs;
    the raw series is kept underneath it so the noise is visible rather than
    hidden.

    The window shrinks at the two ends rather than being padded. The obvious
    ``np.convolve(..., mode="same")`` implicitly pads with zeros, which drags
    both endpoints of the smoothed line toward zero - producing a fake dip at
    the start of the loss curve and a fake collapse at the end of the accuracy
    curve. Overlaying two such curves would show a divergence that exists only
    in the smoothing.
    """
    array = np.asarray(values, dtype=float)
    size = array.size
    if window <= 1 or size == 0:
        return array

    # `window % 2` rather than a flat +1: an even window cannot be centred
    # symmetrically on a point, so only an odd one gets the extra slot past the
    # centre. A flat +1 would average window + 1 points whenever the window is
    # even, quietly contradicting the figure in the legend.
    half = window // 2

    prefix = np.concatenate([[0.0], np.cumsum(array)])
    index = np.arange(size)
    lo = np.maximum(0, index - half)
    hi = np.minimum(size, index + half + (window % 2))
    return (prefix[hi] - prefix[lo]) / (hi - lo)


def plot_curve(curve: LossCurve, output: Path, smooth: int = 50) -> Path:
    fig, (loss_ax, acc_ax) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, height_ratios=[2, 1]
    )

    loss_ax.plot(curve.samples, curve.loss, color="#4c72b0", alpha=0.2, linewidth=0.8)
    loss_ax.plot(
        curve.samples,
        rolling_mean(curve.loss, smooth),
        color="#4c72b0",
        linewidth=1.8,
        label=f"training loss (mean of {smooth} steps)",
    )
    loss_ax.set_ylabel("cross-entropy loss")
    loss_ax.set_title("SimpleMLP on MNIST - single-process reference")
    loss_ax.legend(loc="upper right")
    loss_ax.grid(alpha=0.3)

    acc_ax.plot(curve.samples, curve.accuracy, color="#55a868", alpha=0.2, linewidth=0.8)
    acc_ax.plot(
        curve.samples,
        rolling_mean(curve.accuracy, smooth),
        color="#55a868",
        linewidth=1.8,
        label=f"batch accuracy (mean of {smooth} steps)",
    )
    acc_ax.set_xlabel("training samples consumed")
    acc_ax.set_ylabel("accuracy")
    acc_ax.legend(loc="lower right")
    acc_ax.grid(alpha=0.3)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


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
