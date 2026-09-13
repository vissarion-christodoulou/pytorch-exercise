"""StageBackend's accumulation, driven without any network.

``ModuleBackend.backward`` is the entry point hivemind's Runtime uses, and it
can be called directly: constructing a backend creates its task pools but does
not start them, so these tests spawn no processes.

A worker never steps on its own - only ``reduce_now`` reaches the optimiser, and
only the trainer calls that - so what is testable without a network is the
accumulation and the guards around it. The stepping itself is covered by the
integration tests in ``test_pipeline.py``, which run real peers.

Each test feeds gradients of exactly the shape the trainer sends - the gradient
of a ``reduction="mean"`` cross-entropy with respect to the logits - because the
sample weighting in ``on_backward`` is only correct for that convention.
"""

from __future__ import annotations

import multiprocessing as mp

import torch
import torch.nn as nn

from hivemind.utils import BatchTensorDescriptor

from swarm_mlp.distributed_training.worker import StageBackend

IN_FEATURES, OUT_FEATURES, BATCH = 4, 3, 64


class _StubAverager:
    """A sentinel, not a fake.

    A backend needs a ``GradientAverager`` to exist, but every path these tests
    reach returns before touching it. A fuller fake would only invite the reader
    to believe the test exercises hivemind's averaging, which it does not - that
    lives in ``test_pipeline.py``, over a real DHT.
    """


def make_backend(
    seed: int = 0,
    optimizer_cls=torch.optim.Adam,
    lr: float = 1e-3,
    max_batch_size: int = 4096,
):
    torch.manual_seed(seed)
    module = nn.Sequential(nn.Flatten(), nn.Linear(IN_FEATURES, OUT_FEATURES))
    backend = StageBackend(
        "test.0",
        module,
        optimizer=optimizer_cls(module.parameters(), lr=lr),
        args_schema=(BatchTensorDescriptor(IN_FEATURES),),
        outputs_schema=BatchTensorDescriptor(OUT_FEATURES),
        grad_averager=_StubAverager(),
        min_batch_size=1,
        max_batch_size=max_batch_size,
    )
    return module, backend


def grad_wrt_logits(module: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """What the trainer sends back: d(mean cross-entropy) / d(logits)."""
    with torch.enable_grad():
        logits = module(inputs)
        loss = nn.CrossEntropyLoss()(logits, labels)
        (grad,) = torch.autograd.grad(loss, logits)
    return grad


def batch(n: int, seed: int = 123):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(n, IN_FEATURES, generator=generator)
    labels = torch.randint(0, OUT_FEATURES, (n,), generator=generator)
    return inputs, labels


def test_constructing_a_backend_starts_no_processes():
    before = len(mp.active_children())
    make_backend()
    assert len(mp.active_children()) == before


def test_weighted_accumulation_equals_one_full_batch():
    """Micro-batches of 16 + 48 must accumulate to exactly what one batch of 64 does.

    This is the test that pins the sample weighting: the trainer's loss uses
    ``reduction="mean"``, so each incoming gradient is already divided by its own
    call's batch size. Replacing ``acc += batch_size * grad`` with a plain sum
    over calls weights the 16-sample batch three times too heavily.

    Asserted on the accumulator rather than on the weights, because the worker
    no longer steps on its own - the accumulator *is* what the all-reduce is
    handed, divided by the sample count, so this is the value that matters.
    """
    split_module, split_backend = make_backend()
    whole_module, whole_backend = make_backend()

    inputs, labels = batch(BATCH)
    split_backend.backward(inputs[:16], grad_wrt_logits(split_module, inputs[:16], labels[:16]))
    split_backend.backward(inputs[16:], grad_wrt_logits(split_module, inputs[16:], labels[16:]))
    whole_backend.backward(inputs, grad_wrt_logits(whole_module, inputs, labels))

    assert split_backend.samples_since_step == whole_backend.samples_since_step == BATCH
    assert split_backend.steps == whole_backend.steps == 0, "neither may step on its own"
    for split_acc, whole_acc in zip(split_backend._accumulators, whole_backend._accumulators):
        assert torch.allclose(split_acc, whole_acc, atol=1e-6)


def test_get_stats_reports_every_counter():
    module, backend = make_backend()
    inputs, labels = batch(16)
    backend.backward(inputs, grad_wrt_logits(module, inputs, labels))

    assert backend.get_stats() == {
        "steps": 0,
        "samples_total": 16,
        "samples_since_step": 16,
        "backward_calls": 1,
        "last_effective_batch": 0,
        "averaging_rounds": 0,
        "failed_rounds": 0,
        "missed_rounds": 0,
        "solo_rounds": 0,
        "last_group_size": 0,
    }


def test_backward_pool_is_capped_to_one_trainer_batch():
    """The task pool must never merge two trainer backwards into one call.

    hivemind's TaskPool groups whatever requests are queued together, up to
    ``max_batch_size``, and the trainer issues backwards concurrently - so a
    generous cap really does merge them (measured: one 128-sample call out of 79
    in a 2x2 run). A merged call is unrecoverable here: autograd sums
    contributions that were each normalised by their own sub-batch, so
    ``param.grad`` is ``(1/b) * G_total`` rather than ``(1/B) * G_total``, and
    the individual ``b`` values are gone by the time ``on_backward`` sees it.
    Weighting by the merged size then counts every sub-batch twice over.

    Capping at the trainer's batch size makes hivemind's
    ``total_size + task_size > max_batch_size`` fire on the second task, so one
    call is always one batch. If anyone raises this cap for throughput, the
    gradients go silently wrong in proportion to how many requests happened to
    arrive together - hence this test.
    """
    _, backend = make_backend(max_batch_size=BATCH)
    pools = list(backend.get_pools())
    assert pools, "expected the backend to expose its task pools"
    for pool in pools:
        assert pool.max_batch_size == BATCH


def test_serve_caps_the_pool_at_the_trainer_batch_size_by_default():
    import inspect

    from swarm_mlp.utils.constants import BATCH_SIZE
    from swarm_mlp.distributed_training.worker import serve

    assert inspect.signature(serve).parameters["max_batch_size"].default == BATCH_SIZE


def test_a_missed_round_discards_the_gradients_computed_against_stale_weights():
    """A lost reduce signal means the group stepped without us: drop the batch.

    Nothing here shares an epoch counter, so the round id the trainer sends is
    the only way to notice. Without this check a worker that missed a signal
    would average gradients computed at weights its peers have already left
    behind - and would join the round weighted as though they were fresh.
    """
    module, backend = make_backend()
    backend.last_round = 0

    inputs, labels = batch(32)
    backend.backward(inputs, grad_wrt_logits(module, inputs, labels))
    assert backend.samples_since_step == 32  # accumulated, not stepped

    result = backend.reduce_now(round_id=5)  # rounds 1-4 never arrived

    assert result["reduced"] is False
    assert backend.missed_rounds == 1
    assert backend.samples_since_step == 0
    for accumulator in backend._accumulators:
        assert torch.count_nonzero(accumulator) == 0
    assert backend.last_round == 5, "must resynchronise, not get stuck behind"
