"""The trainer service: samples data and routes it through the workers.

The trainer holds no weights, no optimiser and no gradients of its own. It owns
the dataset and the loss, and that is all. Everything it does is visible in the
inner loop: pull a batch, call the remote stage, compute cross-entropy against
the labels, and call ``.backward()``.

That last call is doing more than it appears to. ``logits`` came out of
hivemind's ``RemoteExpert``, whose forward is an ``autograd.Function``, so
``loss.backward()`` issues a *remote* backward RPC carrying the saved inputs and
the gradient of the loss with respect to the logits. The worker recomputes its
forward, backpropagates into its own parameters, accumulates, and - on its own
schedule, not the trainer's - steps. The input gradients it returns are
discarded here; with a second stage they become the gradients flowing further
back up the pipeline.

With two stages and two replicas each, ``train_pipeline`` below drives one
*lane* per replica index: lane r sends its batch through ``stage0.r`` and then
``stage1.r``, and the lanes run concurrently. That concurrency is not an
optimisation, it is a correctness requirement. A worker all-reduces inside
``on_backward`` and blocks there until its same-stage peer arrives at the same
barrier. If the trainer drove one batch at a time, the first replica to reach
the threshold would wait for a peer that is not being sent any work, and the run
would stall until the averaging timeout. Keeping every replica fed is what makes
the barrier close.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import time
from pathlib import Path
from typing import Sequence

import torch.nn as nn

import hivemind
from hivemind.moe.server import get_experts

from swarm_mlp.control import DEFAULT_SIGNAL_TIMEOUT, signal_reduce
from swarm_mlp.curves import LossCurve
from swarm_mlp.data import mnist_train_loader
from swarm_mlp.model import PIPELINE, STAGE_SHAPES
from swarm_mlp.observability import configure_logging, silence_teardown_noise
from swarm_mlp.reference import BATCH_SIZE, SEED, EPOCHS

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def resolve_expert(
    dht: hivemind.DHT, expert_uid: str, timeout: float = 60.0, poll: float = 0.5
):
    """Look up ``expert_uid`` in the DHT, waiting for it to be declared.

    A worker publishes itself on a timer, so a trainer started in the same second
    will legitimately see ``None`` for a moment. Polling beats requiring the
    operator to time the two commands.

    Note the ``[0] is not None`` test: ``get_experts`` returns a list with one
    entry per requested uid and puts ``None`` in the slots it could not resolve,
    so the list itself is always truthy and ``if experts:`` would return a
    ``None`` expert that fails much later, somewhere less informative.
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


#: How long an idle coroutine waits before re-checking a stage's free list.
#: Zero would spin the event loop hot and starve the threads doing the RPCs.
POLL_INTERVAL = 0.002


class StagePool:
    """The free workers hosting one pipeline stage, and who is busy right now.

    A batch asks the pool for any idle replica; if all of them are busy it waits,
    which is the queue for this stage. ``pin`` asks for one specific replica
    instead - see ``train_pipeline`` for why every group needs a few of those.

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
        self.waits = 0

    async def _claim(self, pin: int | None) -> int:
        while True:
            if pin is None:
                if self._free:
                    return self._free.pop(0)
            elif pin in self._free:
                self._free.remove(pin)
                return pin
            self.waits += 1
            await asyncio.sleep(POLL_INTERVAL)

    @asynccontextmanager
    async def use(self, pin: int | None = None):
        index = await self._claim(pin)
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
    stages: Sequence[str] = PIPELINE,
    replicas: int = 2,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    batches_per_reduce: int = 10,
    signal_timeout: float = DEFAULT_SIGNAL_TIMEOUT,
    seed: int = SEED,
    max_steps: int | None = None,
    log_every: int = 50,
    resolve_timeout: float = 60.0,
    dht: hivemind.DHT | None = None,
) -> LossCurve:
    """Drive MNIST through a multi-stage, multi-replica pipeline of workers.

    ``stages`` names the pipeline in the order activations flow; ``replicas`` is
    how many workers host each stage. Every ``<stage>.<r>`` for r in
    ``range(replicas)`` must already be serving, so the default 2x2 topology
    wants four workers: stage0.0, stage0.1, stage1.0, stage1.1.

    **Scheduling.** Batches are processed in groups of ``batches_per_reduce``,
    which is the unit the workers all-reduce over. Inside a group every batch is
    its own coroutine: it takes any free replica of stage0, hands the activations
    to any free replica of stage1, computes the loss, and calls backward. The
    group ends when all of its batches have finished their backward pass, by
    which point each stage has crossed its collective threshold and averaged.

    Forward and backward are fused per batch rather than run as two phases over
    the group. Since no worker steps until the threshold is crossed, every batch
    in a group sees identical weights either way - the two are numerically the
    same, and fusing avoids holding a group's worth of autograd graphs open.

    **Why the first few batches of each group are pinned.** A worker reaches its
    all-reduce only from inside ``on_backward``, so a replica that was handed no
    work in a group never calls ``step()`` - and its peer, which did cross the
    threshold, blocks at a barrier that can no longer close. Free selection alone
    permits exactly that (a fast replica taking all ten batches), so each group
    deals its first ``replicas`` batches round-robin, one to each replica, and
    only then lets the rest go to whoever is idle. Load-aware for the bulk,
    guaranteed to close for the barrier.

    Computes and returns; writes nothing. The CLI is what persists a curve.
    """
    if batches_per_reduce < replicas:
        # Each group deals its first `replicas` batches one per replica; a group
        # smaller than that leaves some replica with no work, and it will then
        # never reach the barrier its peers are blocking on. The trailing-group
        # path already refuses this case; the main path must too.
        raise ValueError(
            f"batches_per_reduce ({batches_per_reduce}) must be at least replicas "
            f"({replicas}), or some replica gets no batch in a group and cannot "
            "join the all-reduce"
        )

    logger = configure_logging("trainer")

    owns_dht = dht is None
    if owns_dht:
        dht = hivemind.DHT(initial_peers=list(initial_peers), start=True)

    try:
        started = time.monotonic()
        pools = []
        for stage in stages:
            if stage not in STAGE_SHAPES:
                raise ValueError(f"unknown stage {stage!r}; known: {sorted(STAGE_SHAPES)}")
            experts = []
            for replica in range(replicas):
                uid = f"{stage}.{replica}"
                expert = resolve_expert(dht, uid, timeout=resolve_timeout)
                logger.info("resolved %s -> %s", uid, expert.peer_id)
                experts.append(expert)
            pools.append(StagePool(stage, experts))
        logger.info(
            "%d stages x %d replicas resolved in %.1fs; all-reducing every %d batches "
            "(%d samples per stage)",
            len(stages),
            replicas,
            time.monotonic() - started,
            batches_per_reduce,
            batches_per_reduce * batch_size,
        )

        in_shape = STAGE_SHAPES[stages[0]][0]
        loader = mnist_train_loader(batch_size=batch_size, seed=seed)
        criterion = nn.CrossEntropyLoss()

        curve = LossCurve(
            meta={
                "source": "distributed-pipeline",
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": None,  # owned by the workers
                "seed": seed,
                "stages": list(stages),
                "replicas": replicas,
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

        async def process_batch(ordinal: int, images, labels, pin: int | None) -> None:
            nonlocal completed, window_started

            # hivemind validates nested *structure* only, never shapes, so a wrong
            # shape would sail through the client and fail obscurely on the far
            # side of the RPC. Check it here, where the traceback is.
            if tuple(images.shape[1:]) != in_shape:
                raise ValueError(
                    f"batch shape {tuple(images.shape)} does not match "
                    f"stage {stages[0]!r} input {in_shape}"
                )

            # RemoteExpert.forward and Tensor.backward are blocking: hivemind hands
            # the RPC to its own event-loop thread and waits on the result. Awaiting
            # them directly would block this trainer's event loop and serialise every
            # worker - which is exactly the stall the concurrency exists to prevent.
            activations = images
            for pool in pools:
                async with pool.use(pin=pin) as expert:
                    activations = await asyncio.to_thread(expert, activations)

            loss = criterion(activations, labels)
            # One backward walks the whole chain: autograd unwinds the last stage's
            # _RemoteModuleCall, which issues its backward RPC, then the previous
            # stage's, and so on back to the input. The gradients land on exactly
            # the replicas that ran the forward - hivemind recomputes the forward
            # from the saved inputs, so they could not go anywhere else - and each
            # worker accumulates them without stepping.
            await asyncio.to_thread(loss.backward)

            accuracy = activations.argmax(dim=1).eq(labels).float().mean().item()
            results[ordinal] = (labels.size(0), loss.item(), accuracy)

            completed += 1
            if log_every and completed % log_every == 0:
                now = time.monotonic()
                rate = log_every / max(now - window_started, 1e-9)
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
            """Tell every worker its stage has finished a group. Push mode only.

            This is the whole point of push mode: the trainer dealt the batches,
            so it knows the group is complete and exactly how large it was, and
            it says so. No worker has to infer the moment from a gossiped count,
            and - crucially - no worker has to be holding an incoming backward in
            order to find out. Every replica is told at the same instant, so they
            arrive at the all-reduce barrier together instead of whenever their
            next batch happens to land.

            The calls are issued together rather than in sequence: a round cannot
            close until every replica of a stage has joined, so signalling them
            one at a time would serialise exactly what needs to overlap.
            """
            pending = [
                asyncio.wrap_future(
                    signal_reduce(dht, expert.peer_id, expert.uid, round_id, signal_timeout)
                )
                for pool in pools
                for expert in pool.experts
            ]
            acks = await asyncio.gather(*pending, return_exceptions=True)

            for expert, ack in zip(
                [e for pool in pools for e in pool.experts], acks
            ):
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

        ordinal = 0
        round_id = 0
        for epoch in range(epochs):
            group: list[tuple[int, object, object]] = []
            for images, labels in loader:
                if max_steps is not None and ordinal >= max_steps:
                    break
                group.append((ordinal, images, labels))
                ordinal += 1
                if len(group) < batches_per_reduce:
                    continue
                await _run_group(group, process_batch, replicas)
                await signal_group_complete(round_id)
                round_id += 1
                group = []
            if group:
                # A short final group still has to give every replica a batch, or
                # the stage cannot reach its threshold and the last partial
                # all-reduce never closes. Dropping it costs at most
                # batches_per_reduce-1 batches of an epoch.
                if len(group) >= replicas:
                    await _run_group(group, process_batch, replicas)
                    await signal_group_complete(round_id)
                    round_id += 1
                else:
                    logger.info(
                        "dropping a trailing group of %d batches: fewer than %d replicas",
                        len(group),
                        replicas,
                    )

        for key in sorted(results):
            batch_samples, batch_loss, batch_accuracy = results[key]
            samples_seen += batch_samples
            curve.record(samples_seen, batch_loss, batch_accuracy)

        logger.info("batches handled per replica: %s", {p.stage: p.handled for p in pools})
        return curve
    finally:
        if owns_dht:
            dht.shutdown()


async def _run_group(group, process_batch, replicas: int) -> None:
    """Run one all-reduce group's batches concurrently and wait for all of them.

    The first ``replicas`` batches are pinned one per replica so that every
    worker in every stage takes part in the round; the rest go to whichever
    replica is idle.
    """
    await asyncio.gather(
        *(
            process_batch(ordinal, images, labels, pin=i if i < replicas else None)
            for i, (ordinal, images, labels) in enumerate(group)
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m swarm_mlp.trainer",
        description="Sample MNIST batches and drive them through a remote stage.",
    )
    parser.add_argument("--initial-peers", nargs="+", required=True, help="multiaddrs of live peers")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=list(PIPELINE),
        help=f"pipeline stages in flow order (default: {' '.join(PIPELINE)})",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=2,
        help="workers hosting each stage; lane r drives <stage>.r for every stage (default: 2)",
    )
    parser.add_argument(
        "--expert",
        default=None,
        help="drive this single expert uid instead of a pipeline (e.g. full.0)",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--batches-per-reduce",
        type=int,
        default=10,
        help="batches per all-reduce group; must match the workers' "
        "--target-batch-size / --batch-size (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where to write the loss curve "
        "(default: results/distributed_pipeline.json, or distributed_full.json with --expert)",
    )
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = RESULTS_DIR / (
            "distributed_full.json" if args.expert else "distributed_pipeline.json"
        )
    return args


def main() -> None:
    args = parse_args()
    logger = configure_logging("trainer", args.log_level)
    silence_teardown_noise()

    curve = asyncio.run(
        train_pipeline(
            initial_peers=args.initial_peers,
            stages=args.stages,
            replicas=args.replicas,
            epochs=args.epochs,
            batch_size=args.batch_size,
            batches_per_reduce=args.batches_per_reduce,
            seed=args.seed,
            max_steps=args.max_steps,
            log_every=args.log_every,
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
