"""Compare a distributed loss curve against the single-process reference.

This is the script that turns "the curves look similar" into a number. It
reconstructs the matching reference run from the distributed curve's own
metadata rather than from a convention, so a mismatched comparison is not
possible by accident, and it exits non-zero when the curves disagree by more
than the tolerance - so it can be used as a check, not just as a picture.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from swarm_mlp.baseline_reference.reference import train_reference
from swarm_mlp.utils.constants import LEARNING_RATE, RESULTS_DIR
from swarm_mlp.utils.curves import LossCurve
from swarm_mlp.utils.observability import configure_logging
from swarm_mlp.utils.plotting import plot_comparison


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m swarm_mlp.utils.compare",
        description="Overlay a distributed loss curve on the reference and report the difference.",
    )
    parser.add_argument("distributed", type=Path, help="a curve written by the trainer")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="a saved reference curve; recomputed from the distributed run's metadata if omitted",
    )
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "compare_pipeline.png")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--smooth", type=int, default=50)
    parser.add_argument("--log-level", default=None)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    logger = configure_logging("compare", args.log_level)

    distributed = LossCurve.load(args.distributed)

    if args.reference is not None:
        reference = LossCurve.load(args.reference)
    else:
        meta = distributed.meta
        logger.info(
            "recomputing the reference from the distributed run's metadata "
            "(%d steps, batch %s, seed %s)",
            len(distributed),
            meta.get("batch_size"),
            meta.get("seed"),
        )
        reference = train_reference(
            epochs=meta.get("epochs", 1),
            batch_size=meta.get("batch_size"),
            # The trainer has no optimiser, so it records None; the worker used
            # the shared default and so must the reference.
            learning_rate=meta.get("learning_rate") or LEARNING_RATE,
            seed=meta.get("seed"),
            max_steps=len(distributed),
            log_every=0,
        )

    common = min(len(reference), len(distributed))
    if reference.samples[:common] != distributed.samples[:common]:
        logger.error(
            "sample axes differ over the common prefix - the runs consumed data "
            "differently and are not comparable"
        )
        return 2

    deltas = [abs(r - d) for r, d in zip(reference.loss[:common], distributed.loss[:common])]
    max_abs = max(deltas)
    mean_abs = sum(deltas) / len(deltas)
    first_over = next((i for i, v in enumerate(deltas) if v > args.tolerance), None)

    tail = min(common, 50)
    ref_tail = sum(reference.loss[common - tail : common]) / tail
    dist_tail = sum(distributed.loss[common - tail : common]) / tail

    print(f"common prefix: {common} steps")
    print(
        f"max |delta loss| = {max_abs:.3e}   mean |delta loss| = {mean_abs:.3e}   "
        f"first step over {args.tolerance:g}: {first_over if first_over is not None else 'none'}"
    )
    print(f"reference   final {tail}-step mean loss {ref_tail:.4f}")
    print(f"distributed final {tail}-step mean loss {dist_tail:.4f}")

    written = plot_comparison(
        reference,
        distributed,
        args.output,
        smooth=args.smooth,
        tolerance=args.tolerance,
        title="Distributed pipeline vs single-process reference",
    )
    print(f"plot written to {written}")

    if max_abs <= args.tolerance:
        print(f"PASS (tolerance {args.tolerance:g})")
        return 0
    print(f"FAIL: max |delta loss| {max_abs:.3e} exceeds tolerance {args.tolerance:g}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
