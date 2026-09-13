"""The two-stage, two-replica pipeline: the split, the scheduler, and the swarm.

Most of this needs no network. The stage split is pure torch, and ``StagePool``
is pure asyncio - only the last test starts real peers.
"""

from __future__ import annotations

import asyncio

import pytest
import torch
import torch.nn as nn

import hivemind

from swarm_mlp.utils.model import PIPELINE, STAGE_SHAPES, build_model, build_stage
from swarm_mlp.distributed_training.trainer import StagePool, train_pipeline
from swarm_mlp.distributed_training.worker import serve

LOCAL_MADDRS = ["/ip4/127.0.0.1/tcp/0"]


# --------------------------------------------------------------- the split


def test_stages_compose_to_the_full_model():
    """stage0 -> stage1 must be the reference model, forward *and* backward.

    The backward half matters as much as the forward: the pipeline detaches the
    activation at the stage boundary and ships a gradient back across it, so this
    reproduces that boundary explicitly. If this drifts, the distributed curve
    cannot track the reference no matter how correct the averaging is.
    """
    model = build_model(0)
    stage0, stage1 = build_stage("stage0", 0), build_stage("stage1", 0)

    # Each build_stage seeds its own SimpleMLP, so these are equal-valued but
    # distinct tensors; in flow order they line up one-for-one with the model's.
    staged_params = list(stage0.parameters()) + list(stage1.parameters())
    assert len(staged_params) == len(list(model.parameters()))

    images = torch.randn(7, 1, 28, 28)
    labels = torch.randint(0, 10, (7,))
    criterion = nn.CrossEntropyLoss()

    assert torch.equal(model(images), stage1(stage0(images)))

    model.zero_grad()
    criterion(model(images), labels).backward()
    reference_grads = [p.grad.clone() for p in model.parameters()]

    stage0.zero_grad()
    stage1.zero_grad()
    activations = stage0(images)
    boundary = activations.detach().requires_grad_(True)  # the RPC boundary
    criterion(stage1(boundary), labels).backward()
    activations.backward(boundary.grad)  # the activation gradient coming back

    for expected, param in zip(reference_grads, staged_params):
        assert param.grad is not None
        assert torch.equal(expected, param.grad)


def test_stage_shapes_match_the_modules():
    for stage in PIPELINE:
        in_shape, out_shape = STAGE_SHAPES[stage]
        module = build_stage(stage, 0)
        assert tuple(module(torch.zeros(3, *in_shape)).shape[1:]) == out_shape


def test_replicas_of_a_stage_start_identical():
    """Averaged gradients only keep replicas in step if they began in step."""
    for stage in PIPELINE:
        a, b = build_stage(stage, 0), build_stage(stage, 0)
        assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


# ---------------------------------------------------------- the scheduler


def test_pool_hands_out_every_replica_before_repeating():
    async def scenario():
        pool = StagePool("stage0", ["a", "b"])
        async with pool.use() as first:
            async with pool.use() as second:
                assert {first, second} == {"a", "b"}
        return pool.handled

    assert asyncio.run(scenario()) == [1, 1]


def test_every_replica_gets_work_without_pinning():
    """No replica can be starved out of an all-reduce, whatever the arrival skew.

    A replica handed no batches has nothing to contribute, so it declines the
    reduce signal - and its peers then wait out the full averaging_timeout on a
    group that can never reach hivemind's min_group_size of 2. An earlier design
    pinned the first ``replicas`` batches one per replica to prevent that.

    It turns out the data structure already guarantees it, which is why the
    pinning could go: ``_free`` starts holding every replica exactly once, and
    ``_claim`` pops from the front before any release can append a duplicate, so
    the first ``replicas`` claims necessarily land on distinct replicas. This
    pins that property, since it is the only thing standing between free-worker
    selection and a stalled round.
    """

    async def scenario(n_batches, n_replicas, delays):
        pool = StagePool("stage0", [f"r{i}" for i in range(n_replicas)])

        async def one(i):
            async with pool.use():
                await asyncio.sleep(delays[i])

        await asyncio.gather(*(one(i) for i in range(n_batches)))
        return pool.handled

    for replicas in (2, 3, 4):
        for batches in range(replicas, replicas + 6):
            for name, delays in (
                ("instant", [0.0] * batches),
                ("uniform", [0.002] * batches),
                # one replica held far longer than the rest, which is the skew
                # that a pinning scheme was meant to protect against
                ("one slow", [0.02 if i == 0 else 0.0 for i in range(batches)]),
            ):
                handled = asyncio.run(scenario(batches, replicas, delays))
                assert all(h > 0 for h in handled), (
                    f"{name}: {batches} batches over {replicas} replicas starved one: {handled}"
                )


def test_pool_blocks_until_a_replica_is_free():
    """The third batch must wait: that wait *is* the stage's queue."""

    async def scenario():
        pool = StagePool("stage0", ["a", "b"])
        order = []

        async def hold(tag, seconds):
            async with pool.use():
                order.append(f"start {tag}")
                await asyncio.sleep(seconds)
            order.append(f"done {tag}")

        await asyncio.gather(hold("x", 0.05), hold("y", 0.05), hold("z", 0.0))
        return order

    order = asyncio.run(scenario())
    # z cannot start until one of x or y has finished and released its slot
    assert order.index("start z") > min(order.index("done x"), order.index("done y"))


# -------------------------------------------------------------- the swarm


@pytest.mark.integration
def test_2x2_pipeline_all_reduces_and_trains():
    """Four workers, two stages, real DHT: do the replicas actually average?

    The assertion that carries the weight is the gradient norm. Each replica of a
    stage sees different data and so computes a different gradient; if they end a
    step with the same ``last_grad_norm`` and the same weights, the only
    explanation is that they all-reduced.
    """
    batches_per_reduce, batch_size = 4, 64
    seed_dht = hivemind.DHT(host_maddrs=LOCAL_MADDRS, start=True)
    servers, backends = [], {}
    try:
        peers = [str(addr) for addr in seed_dht.get_visible_maddrs()]
        for stage in PIPELINE:
            for index in (0, 1):
                _, server, backend = serve(
                    stage=stage,
                    index=index,
                    initial_peers=peers,
                    num_handlers=1,
                    min_matchmaking_time=2.0,
                    request_timeout=1.0,
                    averaging_timeout=60.0,
                )
                servers.append(server)
                backends[f"{stage}.{index}"] = backend

        curve = asyncio.run(
            train_pipeline(
                initial_peers=peers,
                replicas=2,
                epochs=1,
                batch_size=batch_size,
                batches_per_reduce=batches_per_reduce,
                max_steps=24,
                log_every=0,
            )
        )

        assert len(curve) == 24
        assert curve.samples[-1] == 24 * batch_size

        for stage in PIPELINE:
            first, second = backends[f"{stage}.0"], backends[f"{stage}.1"]

            assert first.averaging_rounds > 0, f"{stage}.0 never all-reduced"
            assert first.failed_rounds == 0, f"{stage}.0 had failed rounds"
            assert second.failed_rounds == 0, f"{stage}.1 had failed rounds"
            assert first.last_group_size == 2, f"{stage}.0 averaged alone"
            assert second.last_group_size == 2, f"{stage}.1 averaged alone"

            # Same averaged gradient -> same norm. Not bit-identical: hivemind
            # splits the tensor into parts and each peer reduces a different one,
            # so the two arrive at the same value by different summation orders.
            assert first.last_grad_norm == pytest.approx(second.last_grad_norm, rel=1e-4)

            drift = max(
                (p - q).abs().max().item()
                for p, q in zip(first.module.parameters(), second.module.parameters())
            )
            assert drift < 1e-4, f"{stage} replicas diverged by {drift:.3e}"

        # The whole point: every worker holds one stage, and between them they
        # hold the whole model exactly once.
        assert sum(
            sum(p.numel() for p in backends[f"{s}.0"].module.parameters()) for s in PIPELINE
        ) == sum(p.numel() for p in build_model(0).parameters())
    finally:
        for backend in backends.values():
            backend.shutdown_averaging()  # before the DHT they depend on goes
        for server in servers:
            server.shutdown()
        seed_dht.shutdown()


@pytest.mark.integration
def test_trainer_signalled_rounds_allow_target_group_size():
    """The trainer says when to reduce, which is what makes tight matchmaking safe.

    ``target_group_size`` lets an all-reduce round close the instant every
    replica has joined instead of waiting out its declared expiration - measured
    at 8.6x faster end to end. It is only safe if the replicas actually arrive
    together, and this exact configuration is the proof: when each worker
    decided for itself from a DHT-gossiped sample count, batches_per_reduce=4
    with target_group_size=2 failed outright with ``AllreduceException: could
    not find a group``, because the workers drifted apart on a lagging estimate.

    Having the trainer signal the round removes that skew by construction -
    every replica is told at the same instant - so the same settings become
    safe. That is the property under test here, not merely that the system runs.
    """
    batches_per_reduce, batch_size, replicas = 4, 64, 2
    seed_dht = hivemind.DHT(host_maddrs=LOCAL_MADDRS, start=True)
    servers, backends = [], {}
    try:
        peers = [str(addr) for addr in seed_dht.get_visible_maddrs()]
        for stage in PIPELINE:
            for index in range(replicas):
                _, server, backend = serve(
                    stage=stage,
                    index=index,
                    initial_peers=peers,
                    num_handlers=1,
                    target_group_size=replicas,
                    min_matchmaking_time=2.0,
                    request_timeout=1.0,
                    averaging_timeout=60.0,
                )
                servers.append(server)
                backends[f"{stage}.{index}"] = backend

        groups = 6
        curve = asyncio.run(
            train_pipeline(
                initial_peers=peers,
                replicas=replicas,
                epochs=1,
                batch_size=batch_size,
                batches_per_reduce=batches_per_reduce,
                max_steps=groups * batches_per_reduce,
                log_every=0,
            )
        )

        assert len(curve) == groups * batches_per_reduce

        for stage in PIPELINE:
            for index in range(replicas):
                backend = backends[f"{stage}.{index}"]
                # Exactly one round per group: the trainer counted the batches
                # it dealt, so there is no overshoot and no round fired by a
                # worker guessing at what the group had collectively seen.
                assert backend.averaging_rounds == groups, (
                    f"{stage}.{index} did {backend.averaging_rounds} rounds, expected {groups}"
                )
                assert backend.failed_rounds == 0, f"{stage}.{index} had failed rounds"
                assert backend.last_group_size == replicas

            first, second = backends[f"{stage}.0"], backends[f"{stage}.1"]
            assert first.last_grad_norm == pytest.approx(second.last_grad_norm, rel=1e-4)
            drift = max(
                (p - q).abs().max().item()
                for p, q in zip(first.module.parameters(), second.module.parameters())
            )
            assert drift < 1e-4, f"{stage} replicas diverged by {drift:.3e}"

        # Every round this swarm ran was one the trainer asked for: each
        # worker stepped once per signal, none stepped on its own initiative,
        # none averaged alone, and none missed a signal. This is the assertion
        # that would catch a worker quietly reintroducing a local threshold.
        for name, backend in backends.items():
            assert backend.steps == groups, f"{name} took {backend.steps} steps"
            assert backend.solo_rounds == 0, f"{name} averaged with nobody"
            assert backend.missed_rounds == 0, f"{name} missed a reduce signal"
    finally:
        for backend in backends.values():
            backend.shutdown_averaging()
        for server in servers:
            server.shutdown()
        seed_dht.shutdown()
