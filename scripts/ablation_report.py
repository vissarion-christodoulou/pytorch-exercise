"""Build the wire-precision ablation plot from the five curves in results/.

Run after the five configurations have been trained (see the README). Reads
``results/ablation_<config>.json`` and writes ``results/ablation_precision.png``.

The measured table that accompanies the plot lives in
``results/ablation_precision.txt``. It is assembled by hand from three sources
the trainer and the checkers already print - throughput and final loss from the
trainer's own log, accuracy against the reference from
``swarm_mlp.utils.compare``, and replica agreement from
``python -m swarm_mlp.distributed_training`` - so there is nothing to automate
here beyond the picture.
"""

from __future__ import annotations

import sys

from swarm_mlp.utils.constants import RESULTS_DIR
from swarm_mlp.utils.curves import LossCurve
from swarm_mlp.utils.plotting import plot_ablation

#: Drawn in this order, baseline first so it sits at the bottom of the legend.
CONFIGS = ("baseline", "grad_fp16", "grad_int8", "act_fp16", "act_int8")


def main() -> int:
    curves = {}
    for name in CONFIGS:
        path = RESULTS_DIR / f"ablation_{name}.json"
        if not path.exists():
            print(f"missing {path} - train that configuration first (see the README)")
            return 1
        curves[name] = LossCurve.load(path)

    written = plot_ablation(
        curves,
        RESULTS_DIR / "ablation_precision.png",
    )
    print(f"plot written to {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
