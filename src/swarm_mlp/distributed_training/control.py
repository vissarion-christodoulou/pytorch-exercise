"""The trainer's control channel to the workers.

hivemind's MoE surface gives a client exactly three RPCs - ``rpc_info``,
``rpc_forward`` and ``rpc_backward`` - and none of them says "the group is
complete, reduce now". Without that, a worker can only *infer* the moment. The
design this replaced inferred it from a ``ProgressTracker``: a DHT-gossiped sum
that lags, overshoots the target, and only gets consulted when a backward
happens to arrive.

The trainer does not have to infer anything. It dealt the batches, so it knows
exactly when a group is done. This module is the wire that lets it say so.

**Why a whole new servicer rather than an extra method on hivemind's.**
``ConnectionHandler`` is ``mp.context.ForkProcess`` - a separate process - while
``Runtime`` is a thread in the worker's main process, and the weights, the
accumulators and the ``GradientAverager`` all live with the Runtime. A signal
delivered to a connection handler would set a flag in a fork's copy of memory
that the Runtime thread can never see: an RPC that returns success and does
nothing. Subclassing it is worse than useless for a second reason -
``_get_handle_name`` is ``f"{cls.__name__}.{method_name}"``, so a subclass would
re-register *every* inherited handler under a new protocol name and every
existing ``RemoteExpert`` client would stop finding ``ConnectionHandler.rpc_forward``.

So ``StageControl`` is registered independently, on a P2P replica taken *in the
worker's main process*, and its handler runs on a thread that shares memory with
the Runtime thread.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.p2p import P2PContext, ServicerBase
from hivemind.proto import runtime_pb2
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)

#: Seconds the trainer waits for a worker to finish a round it asked for. It has
#: to exceed the worker's own averaging timeout, or the trainer gives up on a
#: round that is still perfectly healthy and reports a failure that did not happen.
DEFAULT_SIGNAL_TIMEOUT = 180.0


class StageControl(ServicerBase):
    """Serves ``rpc_reduce_now`` for exactly one worker.

    Do not subclass this. ``ServicerBase._collect_rpc_handlers`` caches its
    handler list on the class and returns early when the attribute is already
    set - which a subclass inherits - so a subclass silently serves the parent's
    handlers under its own protocol name.
    """

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    async def rpc_reduce_now(
        self, request: runtime_pb2.ExpertRequest, context: P2PContext
    ) -> runtime_pb2.ExpertResponse:
        """The trainer says its group is complete. All-reduce and step.

        ``request.metadata`` carries a small JSON blob rather than a bespoke
        protobuf, so this needs no additions to hivemind's ``.proto`` files.
        """
        payload = json.loads(request.metadata) if request.metadata else {}
        round_id = int(payload.get("round", -1))

        # The all-reduce blocks for as long as matchmaking plus the reduction
        # takes. Running it inline would block this event loop, and with it the
        # ack for every other round - so it goes to a thread, and this coroutine
        # simply waits for the result.
        result = await asyncio.to_thread(self.backend.reduce_now, round_id)
        return runtime_pb2.ExpertResponse(metadata=json.dumps(result).encode())


class ControlServer:
    """Runs a ``StageControl`` on its own event loop, in the caller's process.

    The loop lives on a daemon thread of the *worker's main process*, which is
    the whole point: the handler mutates the same ``StageBackend`` object the
    Runtime thread is feeding, so they must share an address space.
    """

    def __init__(self, dht: Any, backend: Any, uid: str) -> None:
        self._dht = dht
        self._uid = uid
        self._servicer = StageControl(backend)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    def start(self, timeout: float = 60.0) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"control[{self._uid}]", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError(f"control channel for {self._uid} did not come up in {timeout:.0f}s")
        if self._error is not None:
            raise self._error
        logger.info("control channel listening for %s", self._uid)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._register())
        except BaseException as error:  # surfaced to start()
            self._error = error
            self._ready.set()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            # Cancel what hivemind's p2p client still has in flight before the
            # loop goes away, or every pending coroutine reports "Event loop is
            # closed" from __del__ during interpreter teardown.
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _register(self) -> None:
        # replicate_p2p caches per-process, so calling it here binds the replica
        # to THIS process - the one holding the weights - rather than to the DHT
        # child or a connection-handler fork.
        p2p = await self._dht.replicate_p2p()
        await self._servicer.add_p2p_handlers(p2p)

    def shutdown(self) -> None:
        """Stop the loop. Deliberately does not remove the p2p handlers.

        ``remove_p2p_handlers`` awaits a daemon reply with no timeout, so calling
        it once the DHT is on its way down hangs forever. The daemon is going
        away regardless; dropping the loop is enough.
        """
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def signal_reduce(dht: Any, peer_id: Any, uid: str, round_id: int, timeout: float):
    """Tell one worker to all-reduce now. Returns a concurrent Future.

    **The call must be made on RemoteExpertWorker's loop.** A P2P replica is
    bound to the loop that created it, and awaiting a stub from any other loop
    does not raise - it silently times out. Since the trainer reaches its
    workers through ``RemoteExpert`` objects, which already live on that loop,
    the call is submitted there and bridged back with ``asyncio.wrap_future``.
    """

    async def _call():
        p2p = await dht.replicate_p2p()
        stub = StageControl.get_stub(p2p, peer_id)
        return await stub.rpc_reduce_now(
            runtime_pb2.ExpertRequest(
                uid=uid, metadata=json.dumps({"round": round_id}).encode()
            ),
            timeout=timeout,
        )

    return RemoteExpertWorker.run_coroutine(_call(), return_future=True)
