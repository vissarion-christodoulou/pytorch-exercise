"""End to end: a real worker, a real trainer, and the reference they must match.

Marked integration because it starts a DHT, a p2pd daemon and a hivemind Server,
which together fork about a dozen processes. It is the test that proves the
whole point of step 2: with one worker hosting the full model and a target batch
size equal to the trainer's batch size, the distributed system performs exactly
the computation the reference does.
"""

from __future__ import annotations

import pytest
import torch

import hivemind

from swarm_mlp.reference import BATCH_SIZE, train_reference
from swarm_mlp.trainer import train
from swarm_mlp.worker import serve

pytestmark = pytest.mark.integration

STEPS = 50
LOCAL_MADDRS = ["/ip4/127.0.0.1/tcp/0"]


def test_distributed_run_reproduces_the_reference():
    seed_dht = hivemind.DHT(host_maddrs=LOCAL_MADDRS, start=True)
    server = None
    try:
        peers = [str(addr) for addr in seed_dht.get_visible_maddrs()]
        _, server, backend = serve(
            stage="full",
            index=0,
            initial_peers=peers,
            num_handlers=1,
        )

        distributed = train(
            expert_uid="full.0",
            initial_peers=peers,
            max_steps=STEPS,
            log_every=0,
        )
        reference = train_reference(epochs=1, log_every=0, max_steps=STEPS)

        assert len(distributed) == STEPS
        assert distributed.samples == reference.samples

        deltas = [abs(d - r) for d, r in zip(distributed.loss, reference.loss)]
        assert max(deltas) <= 1e-5, f"max |delta loss| {max(deltas):.3e}"

        assert distributed.loss[-1] < distributed.loss[0]

        # The worker stepped once per backward because target_batch_size equals
        # the trainer's batch size; loss.backward() blocks on the RPC, so the
        # last step has completed by the time train() returns.
        assert backend.steps == STEPS
        assert backend.samples_total == STEPS * BATCH_SIZE
        assert backend.samples_since_step == 0
    finally:
        # Both, always: the seed DHT owns a p2pd daemon that outlives the test
        # otherwise. Nested so that a throw from the Server's shutdown - which
        # hivemind's does complain during - cannot strand the daemon.
        try:
            if server is not None:
                server.shutdown()
        finally:
            seed_dht.shutdown()


def test_worker_starts_from_the_reference_weights():
    """A worker's initial weights must be the reference's, or curves cannot match."""
    from swarm_mlp.model import build_model, build_stage

    reference_model = build_model(0)
    stage_module = build_stage("full", 0)
    assert all(
        torch.equal(a, b)
        for a, b in zip(reference_model.parameters(), stage_module.parameters())
    )
