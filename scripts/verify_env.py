#!/usr/bin/env python
"""Verify that hivemind actually works in this environment.

Three checks, cheapest first:

1. ``imports``  - hivemind/torch import, and the libp2p ``p2pd`` daemon binary is
   present and executable. Without it every later check fails obscurely.
2. ``dht``      - two DHT peers discover each other over real libp2p and exchange
   a value. This is peer discovery, which the trainer uses to find workers.
3. ``expert``   - a server declares a module to the DHT; a client in the same
   process but over a separate DHT node resolves it by UID and runs a forward
   *and* a backward through it. This round trip is precisely the mechanism the
   trainer/worker split is built on, so it is the check that matters most.

Run directly::

    python scripts/verify_env.py
"""

from __future__ import annotations

import logging
import os
import stat
import sys
import time
import traceback

import torch

import hivemind
from hivemind.moe import Server, get_experts
from hivemind.utils import get_dht_time

# Bind to loopback only: these are local checks and we do not want to announce
# ourselves to any real network, nor wait on external NAT traversal.
LOCAL_MADDRS = ["/ip4/127.0.0.1/tcp/0"]

HIDDEN_DIM = 64
BATCH = 4


def silence_teardown_noise() -> None:
    """Suppress hivemind's cosmetic shutdown tracebacks.

    During interpreter teardown hivemind's p2p ``Client.__del__`` calls
    ``asyncio.get_event_loop()``, which uvloop refuses once the loop is gone. The
    resulting "Exception ignored in ... no current event loop" tracebacks, and the
    matching "Task was destroyed but it is pending" errors, are emitted *after*
    all work has completed and every DHT/Server has been shut down explicitly.

    They are harmless but they print below this script's PASS/FAIL summary, which
    makes a successful run look like a failure. We filter these two specific
    messages and nothing else, so a genuine teardown problem still surfaces.
    """
    default_unraisable_hook = sys.unraisablehook

    def unraisable_hook(args) -> None:
        exc = args.exc_value
        if isinstance(exc, RuntimeError) and "no current event loop" in str(exc):
            return
        default_unraisable_hook(args)

    sys.unraisablehook = unraisable_hook

    class DropPendingTaskErrors(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "Task was destroyed but it is pending" not in record.getMessage()

    logging.getLogger("asyncio").addFilter(DropPendingTaskErrors())


def check_imports() -> None:
    print(f"    hivemind {hivemind.__version__}, torch {torch.__version__}")

    p2pd = os.path.join(os.path.dirname(hivemind.__file__), "hivemind_cli", "p2pd")
    if not os.path.exists(p2pd):
        raise RuntimeError(f"p2pd daemon not found at {p2pd}")
    if not os.stat(p2pd).st_mode & stat.S_IXUSR:
        raise RuntimeError(f"p2pd at {p2pd} is not executable")
    print(f"    p2pd present ({os.path.getsize(p2pd) // 1024} KiB) and executable")


def check_dht() -> None:
    """Two peers, one writes, the other reads."""
    peer_a = hivemind.DHT(host_maddrs=LOCAL_MADDRS, start=True)
    peer_b = None
    try:
        bootstrap = peer_a.get_visible_maddrs()
        print(f"    peer A listening on {bootstrap[0]}")

        peer_b = hivemind.DHT(host_maddrs=LOCAL_MADDRS, initial_peers=bootstrap, start=True)

        key, value = "verify/handshake", {"stage": 0, "worker": "w0"}
        if not peer_b.store(key, value, expiration_time=get_dht_time() + 60):
            raise RuntimeError("peer B failed to store a value")

        # Propagation through the DHT is not instantaneous.
        deadline, found = time.monotonic() + 15.0, None
        while time.monotonic() < deadline:
            found = peer_a.get(key, latest=True)
            if found is not None:
                break
            time.sleep(0.25)

        if found is None:
            raise RuntimeError("peer A never observed the value peer B stored")
        if found.value != value:
            raise RuntimeError(f"value mismatch: stored {value}, read back {found.value}")
        print(f"    peer B stored and peer A read back {found.value}")
    finally:
        peer_a.shutdown()
        if peer_b is not None:
            peer_b.shutdown()


def check_remote_expert() -> None:
    """Forward and backward through a module hosted by another process."""
    bootstrap = hivemind.DHT(host_maddrs=LOCAL_MADDRS, start=True)
    server = None
    client_dht = None
    try:
        peers = bootstrap.get_visible_maddrs()
        server = Server.create(
            expert_uids=["verify.0"],
            expert_cls="ffn",
            hidden_dim=HIDDEN_DIM,
            num_handlers=1,
            initial_peers=peers,
            host_maddrs=LOCAL_MADDRS,
            start=True,
        )
        client_dht = hivemind.DHT(host_maddrs=LOCAL_MADDRS, initial_peers=peers, start=True)

        experts = get_experts(client_dht, ["verify.0"])
        if not experts or experts[0] is None:
            raise RuntimeError("client could not resolve expert 'verify.0' via the DHT")
        expert = experts[0]
        print(f"    client resolved {expert}")

        inputs = torch.randn(BATCH, HIDDEN_DIM, requires_grad=True)
        outputs = expert(inputs)
        if tuple(outputs.shape) != (BATCH, HIDDEN_DIM):
            raise RuntimeError(f"unexpected output shape {tuple(outputs.shape)}")
        print(f"    forward returned {tuple(outputs.shape)}")

        # Probe with a random linear functional rather than something like
        # out.sum() or out.square().mean(). The hivemind 'ffn' block ends in a
        # LayerNorm, so those losses are (near) invariant to the input and would
        # produce a ~0 gradient -- which would look identical to a backward pass
        # that silently returns nothing.
        probe = torch.randn(BATCH, HIDDEN_DIM)
        (outputs * probe).sum().backward()

        if inputs.grad is None:
            raise RuntimeError("backward produced no gradient on the client input")
        if not torch.isfinite(inputs.grad).all():
            raise RuntimeError("backward produced non-finite gradients")
        grad_norm = float(inputs.grad.norm())
        if grad_norm == 0.0:
            raise RuntimeError("backward produced an all-zero gradient")
        print(f"    backward returned gradients to the client (norm {grad_norm:.4f})")
    finally:
        if server is not None:
            server.shutdown()
        if client_dht is not None:
            client_dht.shutdown()
        bootstrap.shutdown()


CHECKS = [
    ("imports and p2pd binary", check_imports),
    ("DHT peer discovery", check_dht),
    ("remote expert forward/backward", check_remote_expert),
]


def main() -> int:
    silence_teardown_noise()

    failures = []
    for name, fn in CHECKS:
        print(f"\n[ .. ] {name}")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            failures.append(name)
            print(f"[FAIL] {name}: {exc}")
            traceback.print_exc()
        else:
            print(f"[ OK ] {name}")

    print()
    if failures:
        print(f"{len(failures)}/{len(CHECKS)} checks FAILED: {', '.join(failures)}")
        return 1
    print(f"All {len(CHECKS)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
