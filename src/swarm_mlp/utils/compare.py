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
from datetime import datetime
from pathlib import Path

from swarm_mlp.baseline_reference.reference import train_reference
from swarm_mlp.utils.constants import CURVE_TIMESTAMP_FORMAT, LEARNING_RATE, RESULTS_DIR
from swarm_mlp.utils.curves import LossCurve
from swarm_mlp.utils.observability import configure_logging
from swarm_mlp.utils.plotting import plot_comparison


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m swarm_mlp.utils.compare",
        description="Overlay a distributed loss curve on the reference and report the difference.",
    )
    parser.add_argument("--distributed", type=Path, help="a curve written by the trainer")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="a saved reference curve; recomputed from the distributed run's metadata if omitted",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where to write the plot "
        "(default: results/compare_pipeline_<timestamp of the distributed curve>.png)",
    )
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--log-level", default=None)
    return parser.parse_args(argv)


def fold_to_groups(curve: LossCurve, batch_size: int, batches_per_reduce: int) -> LossCurve:
    """Collapse a per-batch curve into one point per all-reduce group.

    The workers step once per group, never once per batch, so a group is the
    distributed run's optimiser step and ``target_batch_size`` is its effective
    batch. The reference is rebuilt at that batch size to match - which leaves
    the two curves sharing an x-axis *unit* but not an x-axis: the distributed
    one carries ``batches_per_reduce`` times as many points, each covering that
    fraction of the data. Folding puts them back on the same index.

    A group closes when it is full *or* when the epoch ends, because that is
    what the trainer does. MNIST's epoch ends on a short batch (60000 =
    937*64 + 32), so a short batch marks the boundary as reliably as a full
    group does - and chunking the flat list instead would straddle it and drift
    the sample axis out of step for every epoch after the first.

    Losses are averaged weighted by batch size rather than uniformly: the short
    batch holds half the samples of a full one, and the reference's loss for
    the step it lines up with is the mean over every sample in it.
    """
    counts = [b - a for a, b in zip([0, *curve.samples], curve.samples)]
    folded = LossCurve(meta={**curve.meta, "folded_from_batches": len(curve)})
    start = 0
    for i, n in enumerate(counts):
        if i - start + 1 < batches_per_reduce and n >= batch_size:
            continue
        stop = i + 1
        weights = counts[start:stop]
        total = sum(weights)
        folded.record(
            # The group's x is where it *ends*, which is exactly the cumulative
            # sample count the reference reports after the matching step.
            curve.samples[i],
            sum(v * w for v, w in zip(curve.loss[start:stop], weights)) / total,
            sum(v * w for v, w in zip(curve.accuracy[start:stop], weights)) / total,
        )
        start = stop
    # Anything left over is a group the trainer never reduced on (a --max-steps
    # cut mid-group); it has no counterpart in the reference, so it is dropped.
    return folded


def stamped_path(distributed: Path, name: str, suffix: str) -> Path:
    """``results/<name>_<stamp><suffix>``, stamped to match the curve described.

    The trainer names curves ``distributed_pipeline_<stamp>.json``, so reusing
    that stamp keeps every artefact of a comparison - the plot and the written
    summary - filed with the run it came from, and with each other, rather than
    with whichever run happened to be compared last.

    A curve named anything else - a renamed file, a curve from before the
    trainer stamped them - gets the plain name. Substituting *now* would be
    worse than omitting it: the artefact would claim a provenance it does not
    have, and the one thing you cannot recover afterwards is which run it
    described.
    """
    _, _, stamp = distributed.stem.rpartition("_")
    try:
        datetime.strptime(stamp, CURVE_TIMESTAMP_FORMAT)
    except ValueError:
        return RESULTS_DIR / f"{name}{suffix}"
    return RESULTS_DIR / f"{name}_{stamp}{suffix}"


def main() -> int:
    args = parse_args()
    logger = configure_logging("compare", args.log_level)

    distributed = LossCurve.load(args.distributed)
    meta = distributed.meta

    # One point per optimiser step on both sides, or the comparison lines a
    # 64-sample batch up against a 640-sample one and the axes never meet.
    group_size = meta.get("batches_per_reduce", 1)
    if group_size > 1:
        batches = len(distributed)
        distributed = fold_to_groups(distributed, meta["batch_size"], group_size)
        logger.info(
            "folded %d batches into %d all-reduce groups of up to %s samples",
            batches,
            len(distributed),
            meta.get("target_batch_size", group_size * meta["batch_size"]),
        )

    if args.reference is not None:
        reference = LossCurve.load(args.reference)
    else:
        logger.info(
            "recomputing the reference from the distributed run's metadata "
            "(%d steps, batch %s, seed %s)",
            len(distributed),
            meta.get("target_batch_size"),
            meta.get("seed"),
        )
        reference = train_reference(
            epochs=meta.get("epochs", 1),
            batch_size=meta.get("target_batch_size"),
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

    summary: list[str] = []

    def emit(line: str) -> None:
        """Print a line and keep it, so stdout and the report cannot disagree."""
        print(line)
        summary.append(line)

    # The inputs go in the file too: without them the numbers below are a result
    # with no experiment attached, which is the one thing a saved report must
    # not be.
    emit(f"distributed:   {args.distributed}")
    emit(f"reference:     {args.reference or 'recomputed from the distributed run metadata'}")
    emit(f"common prefix: {common} steps")
    emit(
        f"max |delta loss| = {max_abs:.3e}   mean |delta loss| = {mean_abs:.3e}   "
        f"first step over {args.tolerance:g}: {first_over if first_over is not None else 'none'}"
    )
    emit(f"reference   final {tail}-step mean loss {ref_tail:.4f}")
    emit(f"distributed final {tail}-step mean loss {dist_tail:.4f}")

    written = plot_comparison(
        reference,
        distributed,
        args.output or stamped_path(args.distributed, "compare_pipeline", ".png"),
        smooth=args.smooth,
        tolerance=args.tolerance,
        title="Distributed pipeline vs single-process reference",
    )
    emit(f"plot written to {written}")

    if max_abs <= args.tolerance:
        emit(f"PASS (tolerance {args.tolerance:g})")
        status = 0
    else:
        emit(f"FAIL: max |delta loss| {max_abs:.3e} exceeds tolerance {args.tolerance:g}")
        status = 1

    report = stamped_path(args.distributed, "comparison", ".txt")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"summary written to {report}")
    return status


if __name__ == "__main__":
    sys.exit(main())
