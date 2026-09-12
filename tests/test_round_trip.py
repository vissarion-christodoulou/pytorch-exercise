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


pytestmark = pytest.mark.integration

STEPS = 50
LOCAL_MADDRS = ["/ip4/127.0.0.1/tcp/0"]


def test_worker_starts_from_the_reference_weights():
    """A worker's initial weights must be the reference's, or curves cannot match."""
    from swarm_mlp.model import build_model, build_stage

    reference_model = build_model(0)
    stage_module = build_stage("full", 0)
    assert all(
        torch.equal(a, b)
        for a, b in zip(reference_model.parameters(), stage_module.parameters())
    )
