"""Confirm that the two replicas of each stage finished a run identically.

Every worker writes its final weights on shutdown (``worker.dump_parameters``).
This reads the four dumps from one run and checks the only invariant the
distributed system cannot demonstrate from its own logs: that ``stage0.0`` and
``stage0.1`` hold the same parameters, and likewise for ``stage1``.

That invariant is what the whole design rests on. The replicas of a stage never
exchange weights - they start from one ``build_stage(stage, SEED)`` and apply
the same averaged gradient to the same optimiser state, so they must stay in
lockstep arithmetically. Anything that breaks it (a round one replica took and
the other missed, an all-reduce that closed with one peer, a step on an
unaveraged gradient) still writes a perfectly ordinary-looking step to the log.
The weights are where it shows.

The bar is float32 rounding, not exact equality, and that is a fact about
hivemind rather than a concession. 
About a fifth of the values end up one ULP apart on every single round.

So the question is whether the gap stays at rounding level or compounds. Real
divergence is not subtle: a missed round leaves one replica a whole optimiser
step ahead, which is a difference of order the learning rate - four or five
orders of magnitude above the noise floor. ``DIVERGENCE_THRESHOLD`` sits in that
gap, and the measured drift is printed either way, so a pass that is quietly
getting worse is still visible.

    python -m swarm_mlp.distributed_training                  # the newest run
    python -m swarm_mlp.distributed_training 20260913-111508  # a named one

Exits non-zero if any pair disagrees, so it can gate a run in a script.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from swarm_mlp.distributed_training.constants import REPLICAS_PER_STAGE
from swarm_mlp.utils.constants import CURVE_TIMESTAMP_FORMAT, LEARNING_RATE, RESULTS_DIR
from swarm_mlp.utils.model import PIPELINE

#: A worker's dump, as ``worker.dump_parameters`` names it.
DUMP_GLOB = "worker_{stage}_{index}_*.json"

#: Stamps further apart than this are almost certainly different runs. Workers
#: stamp at their own shutdown, so four of them stopped together still differ by
#: a second or two; a gap of minutes means somebody compared a stale dump.
SUSPICIOUS_SPREAD_SECONDS = 120

#: The smallest real divergence is one replica taking an optimiser step the
#: other did not. Adam's update is ``lr * m_hat / (sqrt(v_hat) + eps)`` and that
#: ratio is bounded near 1, so a step moves a weight by roughly the learning
#: rate: 1e-3. Rounding drift after a few hundred steps measures ~1e-7. A
#: hundredth of a step sits two orders above the noise and two below the signal.
#:
#: Anchored to the step size rather than to the weight magnitude because the
#: drift's scale comes from the gradients and the step count, not from how large
#: the weight happens to be. It does creep up with steps - roughly linearly, so
#: a run hundreds of times longer than this project's would want it revisited.
DIVERGENCE_THRESHOLD = LEARNING_RATE / 100


def stamp_of(path: Path) -> datetime:
    """The timestamp a dump filename carries.

    Raises rather than skipping: a file matching the glob that does not parse
    is a naming drift between this module and the worker, and silently ignoring
    it would present a comparison of the wrong two files as a pass.
    """
    _, _, stamp = path.stem.rpartition("_")
    return datetime.strptime(stamp, CURVE_TIMESTAMP_FORMAT)


def find_dump(stage: str, index: int, target: datetime | None) -> Path:
    """The dump for one worker: nearest to ``target``, or the newest if None.

    Nearest rather than exact, because the four workers do not share a stamp.
    Each one writes as it shuts down, so stopping the swarm gives four stamps a
    second or two apart and no single one of them names the run. The stamp the
    caller passes identifies a run; this picks each worker's contribution to it.
    """
    candidates = sorted(RESULTS_DIR.glob(DUMP_GLOB.format(stage=stage, index=index)))
    if not candidates:
        raise FileNotFoundError(
            f"no dump for {stage}.{index} in {RESULTS_DIR} - did that worker shut down cleanly? "
            "Weights are written on SIGINT/SIGTERM, not on SIGKILL"
        )
    if target is None:
        return max(candidates, key=stamp_of)
    return min(candidates, key=lambda path: abs(stamp_of(path) - target))


def compare(left: dict, right: dict) -> tuple[list[str], list[str]]:
    """How the two dumps' parameters differ: ``(failures, measurements)``.

    Both lists are returned on every call. The measurements are the point of the
    exercise as much as the verdict is: the drift is expected to be non-zero and
    what matters is its size, so it gets printed whether or not it passes.
    """
    left_params, right_params = left["params"], right["params"]
    if left_params.keys() != right_params.keys():
        return ([f"different parameter names: {sorted(left_params)} vs {sorted(right_params)}"], [])

    failures, measurements = [], []
    for name, left_param in left_params.items():
        right_param = right_params[name]
        if left_param["shape"] != right_param["shape"]:
            failures.append(f"{name}: shape {left_param['shape']} vs {right_param['shape']}")
            continue

        pairs = list(zip(left_param["values"], right_param["values"]))
        differing = sum(1 for a, b in pairs if a != b)
        worst_delta, worst_scale = 0.0, 0.0
        for a, b in pairs:
            delta = abs(a - b)
            if delta > worst_delta:
                worst_delta, worst_scale = delta, max(abs(a), abs(b))

        measurements.append(
            f"{name}: {differing}/{len(pairs)} values differ, max |delta| {worst_delta:.3e} "
            f"= {worst_delta / LEARNING_RATE:.1e} of one optimiser step"
        )
        if worst_delta > DIVERGENCE_THRESHOLD:
            failures.append(
                f"{name}: max |delta| {worst_delta:.3e} exceeds {DIVERGENCE_THRESHOLD:.3e} "
                f"(at |value| {worst_scale:.3e})"
            )
    return failures, measurements


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m swarm_mlp.distributed_training",
        description="Check that the replicas of each stage ended a run with identical weights.",
    )
    parser.add_argument(
        "timestamp",
        nargs="?",
        default=None,
        # argparse %-expands help text, so the strftime codes need doubling.
        help=f"the run to check, as {CURVE_TIMESTAMP_FORMAT.replace('%', '%%')} "
        "(default: the most recent dumps)",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    target = (
        datetime.strptime(args.timestamp, CURVE_TIMESTAMP_FORMAT) if args.timestamp else None
    )

    dumps = {
        (stage, index): find_dump(stage, index, target)
        for stage in PIPELINE
        for index in range(REPLICAS_PER_STAGE)
    }

    summary: list[str] = []

    def emit(line: str) -> None:
        """Print a line and keep it, so stdout and the report cannot disagree."""
        print(line)
        summary.append(line)

    # The four filenames go in the report before any number does. A verdict
    # about "the replicas" is worthless without a record of which four files it
    # actually read - especially here, where the four stamps need not match.
    for (stage, index), path in dumps.items():
        emit(f"{stage}.{index}  {path.name}")

    stamps = [stamp_of(path) for path in dumps.values()]
    spread = (max(stamps) - min(stamps)).total_seconds()
    if spread > SUSPICIOUS_SPREAD_SECONDS:
        emit(
            f"\nWARNING: these dumps span {spread:.0f}s, so they may not be one run. "
            f"Pass a timestamp to pin the run down."
        )

    failures = 0
    for stage in PIPELINE:
        loaded = [json.loads(dumps[(stage, i)].read_text()) for i in range(REPLICAS_PER_STAGE)]
        steps = {dump["steps"] for dump in loaded}
        problems, measurements = compare(loaded[0], loaded[1])

        names = " vs ".join(f"{d['stage']}.{d['index']}" for d in loaded)
        emit(f"\n{stage}: {names}")
        emit(f"  steps {sorted(steps)}, samples {[d['samples_total'] for d in loaded]}")
        for measurement in measurements:
            emit(f"  {measurement}")
        if problems:
            failures += 1
            emit("  DIVERGED - this is larger than float32 rounding can explain:")
            for problem in problems:
                emit(f"    {problem}")
        else:
            emit("  agree to within float32 rounding")
        if len(steps) > 1:
            # Not a mismatch on its own: a replica can be signalled a round it
            # had no samples for and correctly decline to step. Worth printing
            # next to identical weights, though, because the two together say
            # the decline was handled right.
            emit(f"  note: replicas took different numbers of steps ({sorted(steps)})")

    emit("")
    if failures:
        emit(f"FAIL: {failures} of {len(PIPELINE)} stages diverged")
        status = 1
    else:
        emit(f"PASS: the replicas of all {len(PIPELINE)} stages stayed in lockstep")
        status = 0

    # Stamped from the first worker in PIPELINE order rather than from the
    # argument, which is optional and, when given, names a run rather than any
    # one file. Picking a real dump's stamp means the report is always named
    # after something that exists on disk.
    stamp = stamp_of(dumps[(PIPELINE[0], 0)]).strftime(CURVE_TIMESTAMP_FORMAT)
    report = RESULTS_DIR / f"distributed_{stamp}.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"summary written to {report}")
    return status


if __name__ == "__main__":
    sys.exit(main())
