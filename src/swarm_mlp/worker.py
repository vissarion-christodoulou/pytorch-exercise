"""The worker service: hosts one pipeline stage and serves forward/backward.

A worker owns a stage's weights and nothing else. It never sees a label, never
computes a loss, and never decides what data to process - it answers requests.
The trainer decides what to send; the worker decides when to step.

That last split is the whole point of this module. hivemind's ``ModuleBackend``
applies its optimiser after *every* backward call, which would make each remote
micro-batch its own optimiser step. The assignment needs the opposite:
accumulate locally, and step only once the workers of this stage have
*collectively* seen a target number of samples - at which point they all-reduce
their gradients with each other and step together.

``StageBackend`` below is that inversion, and it has two modes:

* **local** (``grad_averager is None``) - accumulate into private buffers and
  step on the local sample count. This is a single worker hosting a stage on its
  own, and it is what reproduces the single-process reference exactly.
* **averaged** - accumulate into the same private buffers and hold them until
  the trainer signals the round over the control channel. The trainer dealt the
  batches, so it is the only party that knows the group's exact total; this
  worker never estimates it and never decides for itself. On the signal it
  all-reduces with the other replicas of its stage through a
  ``hivemind.optim.GradientAverager`` and steps. Replicas begin from identical
  weights and apply identical averaged gradients, so they stay identical
  without ever exchanging parameters.

The all-reduce runs on hivemind's Runtime thread, inside ``on_backward``, and it
blocks. That is deliberate and it is also the sharpest constraint on the system:
a worker in the middle of an all-reduce is serving nothing. It is safe only
because the trainer keeps every replica of a stage busy concurrently, so the
replicas arrive at the barrier together. See ``trainer.py`` for the other half
of that argument.
"""

from __future__ import annotations

import argparse
import math
import signal
import threading
from typing import Sequence

import torch
import torch.nn as nn

import hivemind
from hivemind.moe.expert_uid import is_valid_uid
from hivemind.moe.server import ModuleBackend, Server
from hivemind.optim.grad_averager import GradientAverager
from hivemind.utils import BatchTensorDescriptor
from hivemind.utils.logging import get_logger

from swarm_mlp.control import ControlServer
from swarm_mlp.model import STAGE_SHAPES, build_stage
from swarm_mlp.observability import configure_logging, silence_teardown_noise
from swarm_mlp.reference import BATCH_SIZE, LEARNING_RATE, SEED

#: Default number of trainer batches a stage accumulates, collectively across
#: its replicas, before all-reducing. The operator sets the threshold in
#: *samples* (`--target-batch-size`); this is only the default multiplier used
#: when they give `--batches-per-reduce` instead.
BATCHES_PER_REDUCE = 10


class StageBackend(ModuleBackend):
    """A pipeline stage that accumulates gradients and steps at a target batch size.

    hivemind's ``ModuleBackend.on_backward`` steps the optimiser after every
    backward call. This subclass accumulates instead. Alone, it steps once it
    has seen ``target_batch_size`` samples itself. Given a ``GradientAverager``
    it never decides at all: it steps when the trainer calls ``reduce_now``.

    ``target_batch_size`` is therefore read in local mode *only*. In averaged
    mode the threshold lives in the trainer, which is the only party that can
    see what the whole group has processed.

    The optimiser is held here rather than passed to ``ModuleBackend.__init__``
    so that no base-class code path can reach it: the base class treats an
    optimiser as something to step, and this class treats it as something to
    step *on a schedule*.
    """

    def __init__(
        self,
        name: str,
        module: nn.Module,
        *,
        optimizer: torch.optim.Optimizer,
        target_batch_size: int,
        args_schema: tuple[BatchTensorDescriptor, ...],
        outputs_schema: BatchTensorDescriptor,
        grad_averager: GradientAverager | None = None,
        averaging_timeout: float = 120.0,
        **pool_kwargs,
    ) -> None:
        super().__init__(
            name,
            module,
            optimizer=None,
            args_schema=args_schema,
            outputs_schema=outputs_schema,
            **pool_kwargs,
        )
        self.optimizer = optimizer
        self.target_batch_size = target_batch_size
        self.grad_averager = grad_averager
        self.averaging_timeout = averaging_timeout
        self.local_epoch = 0
        # Set before the averager is torn down. The Server thread
        # can still deliver a backward after that point, and touching a dead
        # averager blocks forever on an MPFuture whose owner process is gone.
        self._stopping = threading.Event()
        # on_backward now runs on the Runtime thread while reduce_now runs on
        # the control channel's thread, so the accumulators finally do need a
        # lock. It is nearly uncontended in practice: the trainer only signals a
        # reduce once every backward of the group has returned.
        self._step_lock = threading.RLock()
        #: set by serve() in push mode; owns the thread serving rpc_reduce_now
        self.control: object | None = None
        #: last round id the trainer successfully drove us through (push mode)
        self.last_round = -1
        self.missed_rounds = 0

        self.samples_since_step = 0
        self.samples_total = 0
        self.backward_calls = 0
        self.steps = 0
        self.averaging_rounds = 0
        self.failed_rounds = 0
        self.solo_rounds = 0
        self.last_effective_batch = 0
        self.last_group_size = 0
        self.last_grad_norm = float("nan")

        self._params = [p for p in module.parameters() if p.requires_grad]
        # Private accumulators in BOTH modes. Delegating accumulation to
        # GradientAverager looks tidier but is wrong for variable batch sizes:
        # accumulate_grads_ scales each call by batch_size/anchor_batch_size
        # (the FIRST batch of the round) and load_accumulators_into_averager_
        # divides by the NUMBER of calls, so what reaches the all-reduce is
        # (mean_batch / first_batch) times the true mean gradient. Equal batches
        # hide it; hivemind's own Optimizer always passes a constant
        # batch_size_per_step, which is why the library gets away with it.
        self._accumulators = [torch.zeros_like(p) for p in self._params]

        # get_logger, not configure_logging: the entry point owns global logging
        # setup, and a backend constructed inside a test must not reconfigure it.
        self._logger = get_logger(f"swarm_mlp.worker.{name}")

    @property
    def averaging_enabled(self) -> bool:
        return self.grad_averager is not None

    def on_backward(self, batch_size: int) -> None:
        """Accumulate this call's gradients; step if the target has been reached.

        Called by hivemind's ``Runtime`` loop, which runs in the Server thread of
        the main process. It is the only caller on that thread, but the control
        channel can call ``reduce_now`` on another, so ``_step_lock`` guards the
        accumulators and counters below.

        Worker accumulates gradients until the trainer - which dealt the batches and therefore
        knows the exact total - tells it the group is complete.

        **Why the gradients are weighted by ``batch_size``.** The trainer's loss
        uses ``reduction="mean"``, so the gradient arriving from each backward is
        already divided by *that call's* batch size. Summing the gradients of a
        16-sample and a 48-sample batch and halving them would weight the small
        batch three times too heavily. Scaling each by its own batch size and
        dividing the total by the sample count reproduces exactly the gradient of
        the mean loss over all of them. ``GradientAverager`` normalises the same
        way - ``accumulate_grads_`` scales by ``batch_size / anchor_batch_size``
        and ``load_accumulators_into_averager_`` divides by the number of
        accumulations - so the two modes agree.
        """
        with self._step_lock:
            self._on_backward_locked(batch_size)

    def _on_backward_locked(self, batch_size: int) -> None:
        self.samples_since_step += batch_size
        self.samples_total += batch_size
        self.backward_calls += 1

        if self.averaging_enabled and self._stopping.is_set():
            return  # shutting down: accumulate nothing, join no barrier

        if self.averaging_enabled:
            self._accumulate_locally(batch_size)
            return
        else:
            self._accumulate_locally(batch_size)
            # `>=` with a reset to zero, not `% target == 0` and not a carried-over
            # remainder. Modulo on a running total only fires when batch sizes divide
            # the target exactly - with variable batches (which hivemind's TaskPool
            # produces by aggregating whatever requests arrive together) it silently
            # skips most steps. Carrying a remainder forward would be wrong too: the
            # gradient for those samples has just been applied.
            if self.samples_since_step < self.target_batch_size:
                return
            self._step_locally()

    # ------------------------------------------------------------------ local

    def _accumulate_locally(self, batch_size: int) -> None:
        for param, accumulator in zip(self._params, self._accumulators):
            if param.grad is not None:
                accumulator.add_(param.grad, alpha=float(batch_size))
                param.grad = None

    def _step_locally(self) -> None:
        sum_of_squares = 0.0
        for param, accumulator in zip(self._params, self._accumulators):
            param.grad = accumulator / self.samples_since_step
            sum_of_squares += float(param.grad.pow(2).sum())
        self.last_grad_norm = math.sqrt(sum_of_squares)

        self.optimizer.step()

        for param, accumulator in zip(self._params, self._accumulators):
            param.grad = None
            accumulator.zero_()

        self._finish_step(self.samples_since_step, group_size=1)

    # --------------------------------------------------------------- averaged

    def reduce_now(self, round_id: int = -1) -> dict:
        """All-reduce and step because the trainer says the group is complete.

        Called from the control channel's thread (see ``control.py``), never
        from the Runtime thread. Returns a small dict that becomes the trainer's
        ack, so a caller can tell a real round from a no-op.
        """
        with self._step_lock:
            if not self.averaging_enabled:
                return {"reduced": False, "reason": "averaging is disabled on this worker"}
            if self._stopping.is_set():
                return {"reduced": False, "reason": "worker is shutting down"}
            if self.samples_since_step == 0:
                # Nothing arrived for this worker in that group. Not an error:
                # with free-worker selection the trainer may legitimately have
                # given every batch to our peer. Returning instead of averaging
                # is also what keeps us out of a round we would contribute
                # nothing to - a group in which every peer has weight zero
                # divides by zero inside hivemind and yields NaN gradients
                # silently, with no exception and no log line.
                self.last_round = round_id
                return {"reduced": False, "reason": "no samples accumulated", "round": round_id}

            # If a signal was lost, the group stepped without us: our peers moved on,
            # our weights did not, and everything accumulated since was computed
            # against weights the group has left behind. Averaging that in would
            # corrupt the round.
            if 0 <= self.last_round < round_id - 1:
                self.missed_rounds += 1
                self._logger.error(
                    "missed round(s) %d..%d - the group stepped without us. Discarding "
                    "%d stale samples. NOTE: our weights are now behind our peers' and "
                    "nothing here resynchronises them",
                    self.last_round + 1,
                    round_id - 1,
                    self.samples_since_step,
                )
                self._reset_accumulators()
                self.samples_since_step = 0
                self.last_round = round_id
                return {"reduced": False, "reason": "missed a round", "round": round_id}

            samples = self.samples_since_step
            before = self.steps
            self._all_reduce_and_step()
            self.last_round = round_id
            return {
                "reduced": self.steps > before,
                "round": round_id,
                "samples": samples,
                "steps": self.steps,
                "failed_rounds": self.failed_rounds,
            }

    def _all_reduce_and_step(self) -> None:
        """All-reduce with this stage's other replicas, then step. Blocks."""

        local_samples = self.samples_since_step
        # A fallback only: the real collective total is recovered from the
        # averaging round's `gather` below, which is exact. This value survives
        # only if that gather comes back in a shape we cannot sum.
        group_samples = local_samples

        # Deliberately no "skip averaging if we look alone" shortcut here.
        # There is no cheap way to learn the group size before matchmaking has
        # run, and guessing is worse than waiting: stepping locally because a
        # peer had not been discovered yet diverges the replicas permanently
        # and nothing ever pulls them back together (measured: 6.3e-3 drift
        # where averaging gives <1e-6). Whether we averaged is decided by the
        # averaging result below, never by a guess made beforehand. A genuinely
        # solo stage should be run with --no-averaging; if it is not, the round
        # fails on the timeout and is reported rather than silently corrupting
        # the weights.

        # Finish the mean ourselves, then give the averager a single
        # accumulation of it: with one call the anchor equals the batch size and
        # the call count is 1, so both of hivemind's internal scalings collapse
        # to the identity and what we computed is exactly what gets reduced.
        for param, accumulator in zip(self._params, self._accumulators):
            param.grad = accumulator / local_samples
        self.grad_averager.accumulate_grads_(batch_size=local_samples)

        # Blocks until the other replicas of this stage arrive at the same
        # barrier. `weight` defaults to local_samples_accumulated, so a replica
        # that processed more samples counts proportionally more - which is what
        # keeps the averaged gradient equal to the gradient of the mean loss over
        # every sample the group saw, however unevenly they were split.
        # step() also loads the accumulators into the averager and resets them.
        try:
            # weight= is explicit rather than relying on the averager's own
            # sample counter, which we no longer drive call-by-call.
            # The return value is the gathered data from everyone in the group,
            # so its length is the REAL group size. Nothing cheaper is: a
            # DHT-gossiped peer count was quietly reporting 2 on rounds that
            # had in fact averaged with nobody, which is why the group size is
            # taken from the round itself and from nowhere else.
            # `gather` rides along with matchmaking and comes back as a dict of
            # peer -> that peer's payload, so sending our own sample count is
            # what lets every replica report the TRUE collective batch size
            # rather than guessing from a gossiped estimate. It is also the
            # number the assignment actually asks about: how many samples the
            # stage collectively saw before it stepped.
            gathered = self.grad_averager.step(
                weight=float(local_samples),
                gather=local_samples,
                timeout=self.averaging_timeout,
            )
            group_size = len(gathered) if hasattr(gathered, "__len__") else 1
            if isinstance(gathered, dict) and gathered:
                try:
                    group_samples = sum(int(v) for v in gathered.values())
                except (TypeError, ValueError):
                    pass  # keep the fallback below
        except BaseException as error:
            # This runs on the Runtime thread inside rpc_backward, so an escaping
            # exception is delivered to the *trainer* as a P2PHandlerError and
            # takes the whole run down with it. A round that could not find its
            # peers is a reason to drop one group of gradients, not to stop
            # training. The replicas may drift apart if this keeps happening -
            # nothing here re-synchronises weights - so it is logged at ERROR.
            self._logger.error(
                "all-reduce round failed (%s: %s); dropping %d locally accumulated "
                "samples and continuing%s",
                type(error).__name__,
                error,
                local_samples,
                # last_group_size stays 0 until a round actually closes with
                # peers in it, so this fires exactly when we have never once
                # met a replica - which is what a misconfigured solo worker
                # looks like, and is the difference between "the swarm is
                # flaky" and "you forgot a flag". It reads history rather than
                # guessing at the round that just failed: a worker that has
                # averaged before is having a bad round, not a bad config.
                ""
                if self.last_group_size >= 2
                else " - this stage has never averaged with a peer; run it with "
                "--no-averaging if it is meant to be the only replica",
            )
            self.grad_averager.reset_accumulated_grads_()
            self._reset_accumulators()
            self.failed_rounds += 1
            self._advance_epoch()
            self.samples_since_step = 0
            return

        with self.grad_averager.use_averaged_gradients():
            sum_of_squares = sum(
                float(p.grad.pow(2).sum()) for p in self._params if p.grad is not None
            )
            self.last_grad_norm = math.sqrt(sum_of_squares)
            self.optimizer.step()

        self._reset_accumulators()

        self._advance_epoch()

        self.averaging_rounds += 1
        if group_size < 2:
            # Averaging "succeeded" with nobody else in the group, so this step
            # applied only our own gradient while our peer applied only theirs -
            # the replicas have now diverged and nothing here resynchronises them.
            self.solo_rounds += 1
            self._logger.warning(
                "all-reduce round completed with a group of %d: this step was NOT "
                "averaged and the replicas of this stage have diverged",
                group_size,
            )
        self._finish_step(group_samples, group_size=group_size, local_samples=local_samples)

    # ----------------------------------------------------------------- shared

    def _advance_epoch(self) -> None:
        """Move to the next round.

        The epoch is local bookkeeping only. Replicas stay in step because the
        trainer signals every round to all of them at once, not because they
        share a counter anywhere.
        """
        self.local_epoch += 1

    def _reset_accumulators(self) -> None:
        for param, accumulator in zip(self._params, self._accumulators):
            param.grad = None
            accumulator.zero_()

    def _finish_step(self, effective_batch: int, *, group_size: int, local_samples: int | None = None) -> None:
        self.steps += 1
        self.last_effective_batch = effective_batch
        self.last_group_size = group_size
        self.samples_since_step = 0

        if local_samples is None:
            self._logger.info(
                "step %d  effective_batch %d  samples_total %d  grad_norm %.4f",
                self.steps,
                self.last_effective_batch,
                self.samples_total,
                self.last_grad_norm,
            )
        else:
            self._logger.info(
                "step %d  all-reduced with %d peers  group_batch %d (ours %d)  "
                "samples_total %d  grad_norm %.4f",
                self.steps,
                group_size,
                effective_batch,
                local_samples,
                self.samples_total,
                self.last_grad_norm,
            )

    def shutdown_averaging(self, shutdown_grace: float = 15.0) -> None:
        """Stop the control channel and averager, in that order.

        Order matters: the control channel must stop accepting reduce signals
        before the averager it would drive is torn down, and both must go before
        the DHT they depend on.
        """
        self._stopping.set()

        # Wait - bounded - for any round already running. rpc_reduce_now hands
        # the all-reduce to a thread, so stopping the control loop does NOT stop
        # it: the loop's executor is shut down with wait=False and the round
        # keeps going, holding _step_lock, against a GradientAverager we are
        # about to kill. That left the worker unable to exit for the full
        # averaging_timeout and deaf to Ctrl+C. Taking the lock first means the
        # round either finishes or raises its own timeout while the averager it
        # depends on is still alive.
        acquired = self._step_lock.acquire(timeout=shutdown_grace)
        if not acquired:
            self._logger.warning(
                "an all-reduce was still running after %.0fs; shutting down anyway",
                shutdown_grace,
            )
        try:
            self._shutdown_collectives()
        finally:
            if acquired:
                self._step_lock.release()

    def _shutdown_collectives(self) -> None:
        if self.control is not None:
            self.control.shutdown()
        if self.grad_averager is not None:
            self.grad_averager.shutdown()

    def get_stats(self) -> dict[str, int]:
        """Counters for logging and tests. Nothing in the system reads these yet."""
        return {
            "steps": self.steps,
            "samples_total": self.samples_total,
            "samples_since_step": self.samples_since_step,
            "backward_calls": self.backward_calls,
            "last_effective_batch": self.last_effective_batch,
            "averaging_rounds": self.averaging_rounds,
            "failed_rounds": self.failed_rounds,
            "missed_rounds": self.missed_rounds,
            "solo_rounds": self.solo_rounds,
            "last_group_size": self.last_group_size,
        }


def serve(
    *,
    stage: str,
    index: int,
    initial_peers: Sequence[str] = (),
    host_maddrs: Sequence[str] = ("/ip4/127.0.0.1/tcp/0",),
    identity_path: str | None = None,
    target_batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    seed: int = SEED,
    num_handlers: int = 2,
    update_period: float = 5.0,
    stats_interval: float | None = None,
    max_batch_size: int = BATCH_SIZE,
    averaging: bool = False,
    min_matchmaking_time: float = 2.0,
    request_timeout: float = 1.0,
    target_group_size: int | None = None,
    averaging_timeout: float = 120.0,
) -> tuple[hivemind.DHT, Server, StageBackend]:
    """Start a worker hosting ``<stage>.<index>`` and return its parts.

    With ``averaging=True`` the worker also joins the all-reduce group for its
    stage and stops deciding when to step: the trainer signals each round over
    the control channel, and two replicas sharing a stage each contribute
    whatever they were dealt before they average and step together.
    ``target_batch_size`` is inert in that mode - the trainer's
    ``--batches-per-reduce`` is the real threshold. It governs the step
    interval only under ``averaging=False``.

    The caller owns the lifetime: call ``server.shutdown()`` to stop, which also
    shuts down the DHT it was given, and ``backend.shutdown_averaging()``.
    """
    uid = f"{stage}.{index}"
    if not is_valid_uid(uid):
        raise ValueError(f"{uid!r} is not a valid hivemind expert uid (expected <prefix>.<int>)")
    if averaging and request_timeout >= min_matchmaking_time:
        raise ValueError(
            f"request_timeout ({request_timeout}) must be smaller than min_matchmaking_time "
            f"({min_matchmaking_time}) or averaging rounds will fail; see hivemind's Matchmaking docstring"
        )
    if averaging and min_matchmaking_time >= averaging_timeout:
        # DecentralizedAverager.step asserts scheduled_time < deadline, where
        # scheduled_time is now + min_matchmaking_time and deadline is now +
        # timeout, so this combination fails every round before any network work.
        raise ValueError(
            f"min_matchmaking_time ({min_matchmaking_time}) must be smaller than "
            f"averaging_timeout ({averaging_timeout}) or every round raises immediately"
        )
    if stage not in STAGE_SHAPES:
        raise ValueError(f"unknown stage {stage!r}; known stages: {sorted(STAGE_SHAPES)}")

    logger = configure_logging(f"worker[{uid}]")

    module = build_stage(stage, seed)

    dht_kwargs = {"host_maddrs": list(host_maddrs), "start": True}
    if initial_peers:
        dht_kwargs["initial_peers"] = list(initial_peers)
    if identity_path is not None:
        dht_kwargs["identity_path"] = identity_path
    dht = hivemind.DHT(**dht_kwargs)

    grad_averager = None
    if averaging:
        # Both keys are namespaced by stage, so stage0's replicas form one
        # all-reduce group and stage1's form another. They never mix.
        grad_averager = GradientAverager(
            module.parameters(),
            dht=dht,
            prefix=f"{stage}_grads",
            # hivemind's default is 5s of matchmaking per round, which dominates
            # wall-clock on a local run that averages every few hundred samples.
            # It cannot be lowered blindly though: hivemind requires
            # request_timeout < min_matchmaking_time and warns that otherwise
            # "matchmaking can cause deadlocks". Lowering only the matchmaking
            # window inverted that ordering against the 3s default request
            # timeout, and rounds duly started failing - so both move together.
            min_matchmaking_time=min_matchmaking_time,
            request_timeout=request_timeout,
            # Set this to the number of replicas hosting the stage and an
            # all-reduce round closes the moment they have all arrived, instead
            # of waiting out the declared expiration: hivemind's shortcut in
            # matchmaking.py is guarded by `target_group_size is not None and
            # len(followers) + 1 >= it`, so the default of None disables it.
            #
            # It is a big win and a real risk, and which one you get depends
            # entirely on how tightly the replicas arrive. Measured on the 2x2
            # topology:
            #   batches_per_reduce=10, 60 batches
            #     None -> 110.66s, 4 of 8 rounds failed on timeout
            #     2    ->   3.38s, 16 of 16 rounds clean        (32x faster)
            #   batches_per_reduce=4, 24 batches (the integration test)
            #     None -> passes
            #     2    -> AllreduceException: could not find a group
            #
            # The difference is arrival skew. None tolerates a late replica by
            # waiting; a target group size turns that wait into a hard failure.
            # So this stays off by default until the trainer synchronises
            # arrival - at which point it is the single largest speedup
            # available, since averaging is ~90% of wall clock.
            target_group_size=target_group_size,
            start=True,
        )

    in_shape, out_shape = STAGE_SHAPES[stage]
    backend = StageBackend(
        uid,
        module,
        optimizer=torch.optim.Adam(module.parameters(), lr=learning_rate),
        target_batch_size=target_batch_size,
        args_schema=(BatchTensorDescriptor(*in_shape),),
        outputs_schema=BatchTensorDescriptor(*out_shape),
        grad_averager=grad_averager,
        averaging_timeout=averaging_timeout,
        min_batch_size=1,
        # One trainer batch per call, deliberately. hivemind's TaskPool groups
        # whatever requests are queued together up to max_batch_size, and the
        # trainer issues backwards concurrently, so a generous cap really does
        # merge them (measured: one 128-sample call out of 79 in a 2x2 run).
        # A merged call is unrecoverable for us: autograd sums contributions
        # that were each normalised by their own sub-batch, so param.grad is
        # (1/b)*G_total rather than (1/B)*G_total, and there is no way to
        # recover the individual b's afterwards. Capping at the trainer's batch
        # size makes `total_size + task_size > max_batch_size` fire on the
        # second task, so every call is exactly one batch.
        max_batch_size=max_batch_size,
    )

    # `device` and `stats_report_interval` are Runtime parameters; Server forwards
    # its surplus kwargs straight through to the Runtime it constructs.
    server = Server(
        dht,
        {uid: backend},
        num_connection_handlers=num_handlers,
        update_period=update_period,
        device=torch.device("cpu"),
        stats_report_interval=stats_interval,
        start=True,
    )

    if averaging:
        # Registered here, in the worker's MAIN process, so the handler shares
        # memory with the Runtime thread that owns the accumulators. See the
        # module docstring in control.py for why a ConnectionHandler cannot.
        backend.control = ControlServer(dht, backend, uid)
        backend.control.start()

    logger.info("hosting %s (%d parameters)", uid, sum(p.numel() for p in module.parameters()))
    if averaging:
        # target_batch_size is deliberately absent here: it is inert in this
        # mode, and printing it beside "all-reduce group" invites the reader to
        # believe this worker is counting toward it. The trainer owns that.
        logger.info("all-reduce group %r, rounds triggered by the trainer", stage)
    else:
        logger.info(
            "averaging disabled (solo stage), stepping every %d samples", target_batch_size
        )
    logger.info("peer id %s, dht child pid %s", dht.peer_id, dht.pid)
    logger.info("trainers join with: --initial-peers %s", dht.get_visible_maddrs()[0])

    return dht, server, backend


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m swarm_mlp.worker",
        description="Host one pipeline stage of SimpleMLP as a hivemind peer.",
    )
    parser.add_argument("--stage", default="stage0", help="stage name (default: stage0)")
    parser.add_argument("--index", type=int, default=0, help="replica index within the stage")
    parser.add_argument("--initial-peers", nargs="*", default=[], help="multiaddrs of live peers")
    parser.add_argument("--host-maddrs", nargs="*", default=["/ip4/127.0.0.1/tcp/0"])
    parser.add_argument("--identity-path", default=None, help="private key file for a stable PeerID")
    parser.add_argument(
        "--target-batch-size",
        type=int,
        default=None,
        help="samples to accumulate before stepping. Applies to --no-averaging "
        "only: with averaging on, the trainer signals every round and this is "
        f"ignored (default: --batches-per-reduce x {BATCH_SIZE})",
    )
    parser.add_argument(
        "--batches-per-reduce",
        type=int,
        default=BATCHES_PER_REDUCE,
        help=f"batches per step; sets the target to N x {BATCH_SIZE} samples "
        f"(default: {BATCHES_PER_REDUCE}). Ignored if --target-batch-size is "
        "given, and ignored entirely unless --no-averaging - set the interval "
        "on the trainer instead.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="the trainer's batch size; also caps the task pool so concurrent "
        f"backwards are never merged into one call (default: {BATCH_SIZE})",
    )
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-handlers", type=int, default=2)
    parser.add_argument("--update-period", type=float, default=5.0)
    parser.add_argument(
        "--no-averaging",
        action="store_true",
        help="run this stage solo: accumulate and step locally, never all-reduce",
    )
    parser.add_argument(
        "--min-matchmaking-time",
        type=float,
        default=2.0,
        help="seconds each all-reduce round spends looking for peers (hivemind default: 5); "
        "must be larger than --request-timeout",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=1.0,
        help="seconds for a single matchmaking request (hivemind default: 3)",
    )
    parser.add_argument(
        "--target-group-size",
        type=int,
        default=None,
        help="number of workers hosting this stage; a round then closes as soon "
        "as that many have joined. Measured 32x faster, but it turns a late "
        "replica from a wait into a failed round, so it is off by default until "
        "the trainer synchronises arrival",
    )
    parser.add_argument("--averaging-timeout", type=float, default=120.0)
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=None,
        help="seconds between throughput reports (hivemind's and ours); off by default",
    )
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)
    if args.target_batch_size is None:
        args.target_batch_size = args.batches_per_reduce * args.batch_size
    return args


def main() -> None:
    args = parse_args()
    logger = configure_logging(f"worker[{args.stage}.{args.index}]", args.log_level)
    silence_teardown_noise()

    dht, server, backend = serve(
        stage=args.stage,
        index=args.index,
        initial_peers=args.initial_peers,
        host_maddrs=args.host_maddrs,
        identity_path=args.identity_path,
        target_batch_size=args.target_batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        num_handlers=args.num_handlers,
        update_period=args.update_period,
        stats_interval=args.stats_interval,
        max_batch_size=args.batch_size,
        averaging=not args.no_averaging,
        min_matchmaking_time=args.min_matchmaking_time,
        request_timeout=args.request_timeout,
        target_group_size=args.target_group_size,
        averaging_timeout=args.averaging_timeout,
    )

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    try:
        # Wake on the same cadence as the throughput report so our counters can be
        # read next to hivemind's per-pool numbers rather than minutes apart.
        while not stop.wait(args.stats_interval or 3600.0):
            if args.stats_interval:
                logger.info("stats: %s", backend.get_stats())
    finally:
        backend.shutdown_averaging()
        # Shuts down the DHT too. Expect a second "Server shutdown successfully"
        # and one "ConnectionHandler ... already dead" warning: our call makes
        # Runtime.run return, and Server.run's own finally then shuts down again.
        server.shutdown()
        logger.info(
            "worker stopped after %d steps (%d all-reduce rounds), %d samples",
            backend.steps,
            backend.averaging_rounds,
            backend.samples_total,
        )


if __name__ == "__main__":
    main()
