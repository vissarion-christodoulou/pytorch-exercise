"""StageBackend's accumulate-then-step policy, driven without any network.

``ModuleBackend.backward`` is the entry point hivemind's Runtime uses, and it
can be called directly: constructing a backend creates its task pools but does
not start them, so these tests spawn no processes.

Each test feeds gradients of exactly the shape the trainer sends - the gradient
of a ``reduction="mean"`` cross-entropy with respect to the logits - because the
sample weighting in ``on_backward`` is only correct for that convention.
"""

from __future__ import annotations

import multiprocessing as mp
import random

import pytest
import torch
import torch.nn as nn

from hivemind.utils import BatchTensorDescriptor

from swarm_mlp.worker import StageBackend

IN_FEATURES, OUT_FEATURES, TARGET = 4, 3, 64


def make_backend(
    seed: int = 0,
    optimizer_cls=torch.optim.Adam,
    lr: float = 1e-3,
    target: int = TARGET,
    max_batch_size: int = 4096,
):
    torch.manual_seed(seed)
    module = nn.Sequential(nn.Flatten(), nn.Linear(IN_FEATURES, OUT_FEATURES))
    backend = StageBackend(
        "test.0",
        module,
        optimizer=optimizer_cls(module.parameters(), lr=lr),
        target_batch_size=target,
        args_schema=(BatchTensorDescriptor(IN_FEATURES),),
        outputs_schema=BatchTensorDescriptor(OUT_FEATURES),
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


def test_accumulates_until_target():
    module, backend = make_backend()
    inputs, labels = batch(TARGET)
    before = [p.detach().clone() for p in module.parameters()]

    backend.backward(inputs[:16], grad_wrt_logits(module, inputs[:16], labels[:16]))
    assert backend.steps == 0
    assert backend.samples_since_step == 16
    assert all(torch.equal(p, q) for p, q in zip(module.parameters(), before))

    backend.backward(inputs[16:], grad_wrt_logits(module, inputs[16:], labels[16:]))
    assert backend.steps == 1
    assert backend.samples_since_step == 0
    assert backend.last_effective_batch == TARGET
    assert not any(torch.equal(p, q) for p, q in zip(module.parameters(), before))
    assert all(p.grad is None for p in module.parameters())
    assert all(not accumulator.any() for accumulator in backend._accumulators)


@pytest.mark.parametrize("optimizer_cls,lr", [(torch.optim.Adam, 1e-3), (torch.optim.SGD, 0.1)])
def test_weighted_mean_equals_full_batch(optimizer_cls, lr):
    """Micro-batches of 16 + 48 must land exactly where one batch of 64 does.

    This is the test that pins the sample weighting: replacing
    ``acc += batch_size * grad`` / ``grad = acc / samples`` with a plain mean over
    calls weights the 16-sample batch three times too heavily, and this fails.
    """
    split_module, split_backend = make_backend(optimizer_cls=optimizer_cls, lr=lr)
    whole_module, whole_backend = make_backend(optimizer_cls=optimizer_cls, lr=lr)

    inputs, labels = batch(TARGET)
    split_backend.backward(inputs[:16], grad_wrt_logits(split_module, inputs[:16], labels[:16]))
    split_backend.backward(inputs[16:], grad_wrt_logits(split_module, inputs[16:], labels[16:]))
    whole_backend.backward(inputs, grad_wrt_logits(whole_module, inputs, labels))

    assert split_backend.steps == whole_backend.steps == 1
    for split_param, whole_param in zip(split_module.parameters(), whole_module.parameters()):
        assert torch.allclose(split_param, whole_param, atol=1e-6)


def test_oversized_batch_steps_once():
    module, backend = make_backend()
    inputs, labels = batch(100)
    backend.backward(inputs, grad_wrt_logits(module, inputs, labels))

    assert backend.steps == 1
    assert backend.last_effective_batch == 100
    assert backend.samples_since_step == 0


def test_variable_batches_step_on_threshold_not_modulo():
    """Guards against `examples_processed % target == 0`, which skips most steps."""
    module, backend = make_backend(optimizer_cls=torch.optim.SGD, lr=0.1)

    rng = random.Random(0)
    expected_steps, pending = 0, 0
    for _ in range(50):
        size = rng.choice([16, 24, 32, 48])
        inputs, labels = batch(size, seed=size)
        backend.backward(inputs, grad_wrt_logits(module, inputs, labels))
        pending += size
        if pending >= TARGET:
            expected_steps += 1
            pending = 0

    assert backend.steps == expected_steps
    assert backend.backward_calls == 50


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
    _, backend = make_backend(max_batch_size=TARGET)
    pools = list(backend.get_pools())
    assert pools, "expected the backend to expose its task pools"
    for pool in pools:
        assert pool.max_batch_size == TARGET


def test_serve_caps_the_pool_at_the_trainer_batch_size_by_default():
    import inspect

    from swarm_mlp.reference import BATCH_SIZE
    from swarm_mlp.worker import serve

    assert inspect.signature(serve).parameters["max_batch_size"].default == BATCH_SIZE


class _StubAverager:
    """Just enough GradientAverager to reach the missed-round branch."""

    def __init__(self):
        self.local_samples_accumulated = 0  # structurally 0 on this path
        self.resets = 0

    def reset_accumulated_grads_(self):
        self.resets += 1


def test_push_mode_discards_gradients_when_a_round_was_missed():
    """A lost reduce signal means the group stepped without us: drop the batch.

    Nothing here shares an epoch counter, so the round id the trainer sends is
    the only way to notice. Without this check a worker that missed a signal
    would average gradients computed at weights its peers have already left
    behind - and would join the round weighted as though they were fresh.
    """
    module, backend = make_backend()
    backend.grad_averager = _StubAverager()
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
