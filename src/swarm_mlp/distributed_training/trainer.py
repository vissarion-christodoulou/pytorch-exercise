"""The trainer service: samples data and routes it through the workers.

The trainer holds no weights, no optimiser and no gradients of its own. It owns
1) the dataset
2) the loss
3) the worker load balancing
4) when each stage all-reduces

At any given point it holds a queue of available workers per stage and assigns
them batches. It computes the loss when a last-stage worker completes a forward
pass, and calls ``.backward()``.

That last call is doing more than it appears to. ``logits`` came out of
hivemind's ``RemoteExpert``, whose forward is an ``autograd.Function``, so
``loss.backward()`` issues a *remote* backward RPC carrying the saved inputs and
the gradient of the loss with respect to the logits. The worker recomputes its
forward, backpropagates into its own parameters and accumulates. The input
gradients become the gradients flowing further back up the pipeline.

The workers only step when the trainer tells them to, which it does once a whole
group of batches has been through both the forward and the backward pass.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
import time
from pathlib import Path
from typing import Sequence

import torch.nn as nn

import hivemind
from hivemind.moe.server import get_experts

from swarm_mlp.distributed_training.constants import (
    DEFAULT_BATCHES_PER_REDUCE,
    POLL_INTERVAL_FOR_FREE_WORKER, 
    REPLICAS_PER_STAGE,
    RESOLVE_EXPERT_TIMEOUT, 
    TRAINER_LOG_EVERY
)
from swarm_mlp.distributed_training.control import DEFAULT_SIGNAL_TIMEOUT, signal_reduce
from swarm_mlp.utils.constants import (
    BATCH_SIZE,
    CURVE_TIMESTAMP_FORMAT,
    EPOCHS,
    RESULTS_DIR,
    SEED,
)
from swarm_mlp.utils.curves import LossCurve
from swarm_mlp.utils.data import mnist_train_loader
from swarm_mlp.utils.model import PIPELINE, STAGE_SHAPES
from swarm_mlp.utils.observability import configure_logging, silence_teardown_noise



def resolve_expert(
    dht: hivemind.DHT, expert_uid: str, timeout: float = 60.0, poll: float = 0.5
):
    """Look up ``expert_uid`` in the DHT, waiting for it to be declared.

    Note the ``[0] is not None`` test: ``get_experts`` returns a list with one
    entry per requested uid and puts ``None`` in the slots it could not resolve,
    so the list itself is always truthy
    """
    deadline = time.monotonic() + timeout
    while True:
        expert = get_experts(dht, [expert_uid])[0]
        if expert is not None:
            return expert
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"expert {expert_uid!r} was not declared in the DHT within {timeout:.0f}s - "
                "is the worker running, and were its --initial-peers passed here?"
            )
        time.sleep(poll)


class StagePool:
    """The free workers hosting one pipeline stage, and who is busy right now.

    A batch asks the pool for any idle replica; if all of them are busy it waits,
    which is the queue for this stage. 

    No lock guards ``_free``. asyncio only switches coroutines at ``await``
    points, and there is no await between testing the list and mutating it, so
    each claim is atomic with respect to every other batch in flight. (Verified
    the hard way: inserting a single ``await`` between the test and the ``pop``
    produces an immediate IndexError under concurrency.)
    """

    def __init__(self, stage: str, experts: Sequence[object]) -> None:
        self.stage = stage
        self.experts = list(experts)
        self._free = list(range(len(self.experts)))
        #: batches handled per replica; the load-balance evidence
        self.handled = [0] * len(self.experts)
        #: poll iterations spent with every replica busy. Times POLL_INTERVAL_FOR_FREE_WORKER
        #: this is roughly how long batches sat waiting for a free worker..
        self.polls_blocked = 0

    async def _claim(self) -> int:
        while True:
            if self._free:
                return self._free.pop(0)
            self.polls_blocked += 1
            await asyncio.sleep(POLL_INTERVAL_FOR_FREE_WORKER)

    @asynccontextmanager
    async def use(self):
        index = await self._claim()
        self.handled[index] += 1
        try:
            yield self.experts[index]
        finally:
            # Released as soon as this stage's forward returns, not when the
            # whole batch finishes: that is what lets this replica start the
            # next batch while the current one is still in a later stage.
            self._free.append(index)


async def train_pipeline(
    *,
    initial_peers: Sequence[str],
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    batches_per_reduce: int = DEFAULT_BATCHES_PER_REDUCE,
) -> LossCurve:
    """Drive MNIST through a multi-stage, multi-replica pipeline of workers.

    ``PIPELINE`` names the pipeline stages in the order activations flow, and
    ``REPLICAS_PER_STAGE`` workers host each of them. Every ``<stage>.<r>`` for r
    in ``range(REPLICAS_PER_STAGE)`` must already be serving, so the 2x2 topology
    wants four workers: stage0.0, stage0.1, stage1.0, stage1.1.

    **Scheduling.** Batches are processed in groups of ``batches_per_reduce``,
    which is the unit the workers all-reduce over. Inside a group every batch is
    its own coroutine: it takes any free replica of stage0, hands the activations
    to any free replica of stage1, computes the loss, and calls backward. When
    every batch in the group has finished its backward pass, the trainer signals
    each worker to all-reduce and step.

    Forward and backward are fused per batch rather than run as two phases over
    the group. Since no worker steps until the threshold is crossed, every batch
    in a group sees identical weights either way - the two are numerically the
    same, and fusing avoids holding a group's worth of autograd graphs open.

    Computes and returns; writes nothing. The CLI is what persists a curve.
    """
    logger = configure_logging("trainer")
    dht = hivemind.DHT(initial_peers=list(initial_peers), start=True)

    try:
        started = time.monotonic()
        pools = []
        for stage in PIPELINE:
            experts = []
            for replica in range(REPLICAS_PER_STAGE):
                uid = f"{stage}.{replica}"
                expert = resolve_expert(dht, uid, timeout=RESOLVE_EXPERT_TIMEOUT)
                logger.info("resolved %s -> %s", uid, expert.peer_id)
                experts.append(expert)
            pools.append(StagePool(stage, experts))
        logger.info(
            "%d stages x %d replicas resolved in %.1fs; all-reducing every %d batches "
            "(%d samples per stage)",
            len(PIPELINE),
            REPLICAS_PER_STAGE,
            time.monotonic() - started,
            batches_per_reduce,
            batches_per_reduce * batch_size,
        )

        in_shape = STAGE_SHAPES[PIPELINE[0]][0]
        loader = mnist_train_loader(batch_size=batch_size, seed=SEED)
        criterion = nn.CrossEntropyLoss()

        curve = LossCurve(
            meta={
                "source": "distributed-pipeline",
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": None,  # owned by the workers
                "seed": SEED,
                "stages": list(PIPELINE),
                "replicas": REPLICAS_PER_STAGE,
                "batches_per_reduce": batches_per_reduce,
                "target_batch_size": batches_per_reduce * batch_size,
                "note": (
                    "learning rate and the all-reduce threshold live on the workers; "
                    "the trainer holds no weights and no optimiser"
                ),
            }
        )

        samples_seen = 0
        completed = 0
        window_started = time.monotonic()
        # ordinal -> (samples, loss, accuracy); filled concurrently, read in order
        results: dict[int, tuple[int, float, float]] = {}

        async def process_batch(ordinal: int, images, labels) -> None:
            nonlocal completed, window_started

            # hivemind validates nested *structure* only, never shapes, so a wrong
            # shape would sail through the client and fail obscurely on the far
            # side of the RPC. Check it here, where the traceback is.
            if tuple(images.shape[1:]) != in_shape:
                raise ValueError(
                    f"batch shape {tuple(images.shape)} does not match "
                    f"stage {PIPELINE[0]!r} input {in_shape}"
                )

            # RemoteExpert.forward and Tensor.backward are blocking: hivemind hands
            # the RPC to its own event-loop thread and waits on the result. Awaiting
            # them directly would block this trainer's event loop and serialise every
            # worker - which is exactly the stall the concurrency exists to prevent.
            activations = images
            for pool in pools:
                async with pool.use() as expert:
                    activations = await asyncio.to_thread(expert, activations)

            loss = criterion(activations, labels)
            # One backward walks the whole chain: autograd unwinds the last stage's
            # _RemoteModuleCall, which issues its backward RPC, then the previous
            # stage's, and so on back to the input. The gradients land on exactly
            # the replicas that ran the forward and each worker accumulates them 
            # without stepping.
            await asyncio.to_thread(loss.backward)

            accuracy = activations.argmax(dim=1).eq(labels).float().mean().item()
            results[ordinal] = (labels.size(0), loss.item(), accuracy)

            completed += 1
            if completed % TRAINER_LOG_EVERY == 0:
                now = time.monotonic()
                rate = TRAINER_LOG_EVERY / max(now - window_started, 1e-9)
                window_started = now
                logger.info(
                    "batch %5d  loss %.4f  acc %.3f  (%.1f batches/s)  load %s",
                    completed,
                    loss.item(),
                    accuracy,
                    rate,
                    " ".join(f"{p.stage}={p.handled}" for p in pools),
                )

        async def signal_group_complete(round_id: int) -> None:
            """Tell every worker its stage has finished a group.

            The trainer dealt the batches, so it knows the group is complete and
            exactly how large it was, and it says so. No worker has to infer the
            moment, and - crucially - no worker has to be holding an incoming
            backward in order to find out. Every replica is told at the same
            instant, so they arrive at the all-reduce barrier together instead
            of whenever their next batch happens to land.

            The calls are issued together rather than in sequence: a round cannot
            close until every replica of a stage has joined, so signalling them
            one at a time would serialise exactly what needs to overlap.
            """
            experts = [expert for pool in pools for expert in pool.experts]
            acks = await asyncio.gather(
                *(
                    asyncio.wrap_future(
                        signal_reduce(dht, e.peer_id, e.uid, round_id, DEFAULT_SIGNAL_TIMEOUT)
                    )
                    for e in experts
                ),
                return_exceptions=True,
            )

            for expert, ack in zip(experts, acks):
                if isinstance(ack, BaseException):
                    # A dropped signal costs this worker one group of gradients:
                    # it keeps accumulating and folds them into the next round,
                    # which the sample weighting handles correctly. Loud, but not
                    # fatal - the run continues.
                    logger.error(
                        "reduce signal to %s failed (%s: %s); its gradients roll "
                        "into the next round",
                        expert.uid,
                        type(ack).__name__,
                        ack,
                    )
                    continue

                # A worker can accept the signal and still decline the round.
                # A stage that keeps declining is a stage whose replicas are
                # drifting apart, so it is worth a line.
                report = json.loads(ack.metadata) if ack.metadata else {}
                if not report.get("reduced"):
                    logger.warning(
                        "round %d: %s did not reduce (%s)",
                        round_id,
                        expert.uid,
                        report.get("reason", "no reason given"),
                    )
                    raise Exception

        ordinal = 0
        round_id = 0
        for epoch in range(epochs):
            group: list[tuple[int, object, object]] = []
            for images, labels in loader:
                group.append((ordinal, images, labels))
                ordinal += 1
                if len(group) < batches_per_reduce:
                    continue
                await _run_group(group, process_batch)
                await signal_group_complete(round_id)
                round_id += 1
                group = []
            if group:
                # A trailing group shorter than REPLICAS_PER_STAGE cannot
                # give every replica a batch, which makes stepping harder.
                # Quick and dirty solution: skip a tail of at most
                # REPLICAS_PER_STAGE-1 batches
                if len(group) >= REPLICAS_PER_STAGE:
                    await _run_group(group, process_batch)
                    await signal_group_complete(round_id)
                    round_id += 1
                else:
                    logger.info(
                        "dropping a trailing group of %d batches: fewer than the "
                        "%d replicas that must each contribute to a round",
                        len(group),
                        REPLICAS_PER_STAGE,
                    )

        for key in sorted(results):
            batch_samples, batch_loss, batch_accuracy = results[key]
            samples_seen += batch_samples
            curve.record(samples_seen, batch_loss, batch_accuracy)

        logger.info(
            "batches handled per replica: %s",
            {p.stage: p.handled for p in pools},
        )
        logger.info(
            "time spent waiting for a free worker: %s",
            {p.stage: f"{p.polls_blocked * POLL_INTERVAL_FOR_FREE_WORKER:.1f}s" for p in pools},
        )
        return curve
    finally:
        dht.shutdown()


async def _run_group(group, process_batch) -> None:
    """Run one all-reduce group's batches concurrently and wait for all of them."""
    await asyncio.gather(
        *(process_batch(ordinal, images, labels) for ordinal, images, labels in group)
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m swarm_mlp.distributed_training.trainer",
        description="Sample MNIST batches and drive them through a remote stage.",
    )
    parser.add_argument("--initial-peers", nargs="+", required=True, help="multiaddrs of live peers")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--batches-per-reduce",
        type=int,
        default=10,
        help="batches per all-reduce group. The trainer owns this interval "
        "outright - the workers accumulate until told, so nothing needs to be "
        "kept in sync with them (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where to write the loss curve "
        "(default: results/distributed_pipeline_<timestamp>.json)",
    )
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)
    if args.output is None:
        # Stamped, so a run never silently overwrites the curve from the last
        # one - these take minutes to produce and are the only record of a
        # configuration that has since been changed. compare.py reads this stamp
        # back off the filename, which is why the format is shared rather than
        # spelled out here.
        stamp = time.strftime(CURVE_TIMESTAMP_FORMAT)
        args.output = RESULTS_DIR / f"distributed_pipeline_{stamp}.json"
    return args


def main() -> None:
    args = parse_args()
    logger = configure_logging("trainer", args.log_level)
    silence_teardown_noise()

    curve = asyncio.run(
        train_pipeline(
            initial_peers=args.initial_peers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            batches_per_reduce=args.batches_per_reduce
        )
    )

    tail = min(len(curve), 50)
    mean_loss = sum(curve.loss[-tail:]) / tail
    mean_accuracy = sum(curve.accuracy[-tail:]) / tail
    logger.info(
        "%d steps, %d samples. Final %d-step mean: loss %.4f, accuracy %.4f",
        len(curve),
        curve.samples[-1],
        tail,
        mean_loss,
        mean_accuracy,
    )
    logger.info("curve written to %s", curve.save(args.output))


if __name__ == "__main__":
    main()
