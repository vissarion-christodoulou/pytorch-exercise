"""Logging setup shared by every service.

A running system is roughly twenty OS processes: per worker, a main process, a
DHT fork, a ``p2pd`` daemon, N connection handlers, two task-pool forks and
torch's shared-memory manager - plus a trainer with its own DHT. Nothing in a
traceback tells you which of them produced a line, and no exception crosses an
RPC boundary intact. So every log line has to carry its own provenance.
"""

from __future__ import annotations

import logging
import os
import sys

from hivemind.utils.logging import get_logger, use_hivemind_log_handler

DEFAULT_LEVEL = "INFO"


class _RoleFilter(logging.Filter):
    """Prefix every record with the role and the pid that emitted it.

    Attached to the *handler*, not to a logger: a filter on a logger only sees
    records logged directly to it, while a filter on a handler sees everything
    that reaches the handler - including records propagated up from hivemind's
    own loggers, which is most of what a worker prints.

    ``os.getpid()`` is read per record rather than captured once, so a forked
    child (a task pool, a connection handler) reports its own pid without any
    extra setup: ``fork`` inherits the configured handler as-is.
    """

    def __init__(self, role: str) -> None:
        super().__init__()
        self.role = role

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = f"[{self.role} pid={os.getpid()}] {record.msg}"
        return True


def configure_logging(role: str, level: str | None = None) -> logging.Logger:
    """Route our logs and hivemind's through one tagged handler.

    Idempotent: calling it twice replaces the tag rather than prefixing every
    line twice. Entry points call it; library code calls ``get_logger`` and
    inherits whatever the entry point configured.

    :param role: short provenance tag, e.g. ``"worker[stage0.0]"`` or ``"trainer"``.
    :param level: overrides ``HIVEMIND_LOGLEVEL``; defaults to INFO.
    """
    # hivemind's handler is installed on the `hivemind` logger by default and
    # does not propagate. Moving it to the root logger is what lets our own
    # loggers share the formatter - and what puts every record in front of the
    # filter below.
    use_hivemind_log_handler("in_root_logger")

    root = logging.getLogger()
    for handler in root.handlers:
        for existing in [f for f in handler.filters if isinstance(f, _RoleFilter)]:
            handler.removeFilter(existing)
        handler.addFilter(_RoleFilter(role))

    # After use_hivemind_log_handler, which sets the level from HIVEMIND_LOGLEVEL
    # itself - so an explicit argument has to win by being applied second.
    root.setLevel(level or os.environ.get("HIVEMIND_LOGLEVEL", DEFAULT_LEVEL))

    # hivemind logs to stderr unbuffered while our progress goes to stdout, which
    # Python block-buffers when it is a pipe rather than a terminal. Without this
    # the two streams interleave mid-line and the output is unreadable.
    sys.stdout.reconfigure(line_buffering=True)

    return get_logger(f"swarm_mlp.{role}")


def silence_teardown_noise() -> None:
    """Suppress hivemind's cosmetic shutdown tracebacks.

    During interpreter teardown hivemind's p2p ``Client.__del__`` calls
    ``asyncio.get_event_loop()``, which uvloop refuses once the loop is gone. The
    resulting "Exception ignored in ... no current event loop" tracebacks, and the
    matching "Task was destroyed but it is pending" errors, are emitted *after*
    all work has completed and every DHT/Server has been shut down explicitly.

    They are harmless but they print below a service's final summary, which makes
    a successful run look like a failure. We filter these two specific messages
    and nothing else, so a genuine teardown problem still surfaces.
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
