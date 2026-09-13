"""Rendering for loss curves.

Separated from the CLIs so that ``swarm_mlp.utils.compare`` and ``swarm_mlp.baseline_reference.__main__``
render the same way, and so that importing a training function does not drag
matplotlib in with it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Select the non-interactive backend before pyplot is imported. There is no
# display inside WSL2, so this is the only thing that can work; saying so
# explicitly beats relying on matplotlib's fallback and then wondering why
# plt.show() does nothing.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  - must follow matplotlib.use
import numpy as np  # noqa: E402

from swarm_mlp.utils.curves import LossCurve  # noqa: E402

REFERENCE_COLOR = "#4c72b0"
DISTRIBUTED_COLOR = "#c44e52"
ACCURACY_COLOR = "#55a868"


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

    loss_ax.plot(curve.samples, curve.loss, color=REFERENCE_COLOR, alpha=0.2, linewidth=0.8)
    loss_ax.plot(
        curve.samples,
        rolling_mean(curve.loss, smooth),
        color=REFERENCE_COLOR,
        linewidth=1.8,
        label=f"training loss (mean of {smooth} steps)",
    )
    loss_ax.set_ylabel("cross-entropy loss")
    loss_ax.set_title("SimpleMLP on MNIST - single-process reference")
    loss_ax.legend(loc="upper right")
    loss_ax.grid(alpha=0.3)

    acc_ax.plot(curve.samples, curve.accuracy, color=ACCURACY_COLOR, alpha=0.2, linewidth=0.8)
    acc_ax.plot(
        curve.samples,
        rolling_mean(curve.accuracy, smooth),
        color=ACCURACY_COLOR,
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


def plot_comparison(
    reference: LossCurve,
    distributed: LossCurve,
    output: Path,
    *,
    smooth: int = 50,
    tolerance: float = 1e-3,
    title: str = "",
) -> Path:
    """Overlay two curves and, below them, the per-step absolute difference.

    The third panel is the one a reviewer actually reads: two loss curves drawn
    on top of each other look identical long before they are, because the
    interesting disagreements are three orders of magnitude below the line
    width. Plotting |delta| on a log axis against the tolerance makes the claim
    checkable instead of aesthetic.
    """
    fig, (loss_ax, acc_ax, delta_ax) = plt.subplots(
        3, 1, figsize=(9, 10), sharex=True, height_ratios=[3, 2, 2]
    )

    for curve, color, name in (
        (reference, REFERENCE_COLOR, "reference"),
        (distributed, DISTRIBUTED_COLOR, "distributed"),
    ):
        loss_ax.plot(curve.samples, curve.loss, color=color, alpha=0.15, linewidth=0.8)
        loss_ax.plot(
            curve.samples,
            rolling_mean(curve.loss, smooth),
            color=color,
            linewidth=1.8,
            label=f"{name} (mean of {smooth} steps)",
        )
        acc_ax.plot(curve.samples, curve.accuracy, color=color, alpha=0.15, linewidth=0.8)
        acc_ax.plot(
            curve.samples,
            rolling_mean(curve.accuracy, smooth),
            color=color,
            linewidth=1.8,
            label=f"{name} (mean of {smooth} steps)",
        )

    loss_ax.set_ylabel("cross-entropy loss")
    loss_ax.set_title(title or "Distributed vs single-process reference")
    loss_ax.legend(loc="upper right")
    loss_ax.grid(alpha=0.3)

    acc_ax.set_ylabel("batch accuracy")
    acc_ax.legend(loc="lower right")
    acc_ax.grid(alpha=0.3)

    common = min(len(reference), len(distributed))
    delta = np.abs(
        np.asarray(reference.loss[:common]) - np.asarray(distributed.loss[:common])
    )
    samples = reference.samples[:common]

    # A perfect match plots as all-zeros, which a log axis cannot show. Say so
    # in the panel rather than rendering an empty box.
    if np.all(delta == 0.0):
        delta_ax.plot(samples, np.zeros_like(delta), color="#8172b2", linewidth=1.5)
        delta_ax.set_ylim(-1, 1)
        delta_ax.text(
            0.5,
            0.5,
            "identical: max |delta loss| = 0",
            transform=delta_ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
    else:
        delta_ax.semilogy(samples, np.maximum(delta, 1e-18), color="#8172b2", linewidth=0.9)
        delta_ax.axhline(
            tolerance, color="#937860", linestyle="--", linewidth=1.2,
            label=f"tolerance {tolerance:g}",
        )
        delta_ax.legend(loc="upper right")

    delta_ax.set_xlabel("training samples consumed")
    delta_ax.set_ylabel("|delta loss| per step")
    delta_ax.grid(alpha=0.3)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
