"""The shape of the swarm, fixed rather than configured.
"""
from __future__ import annotations

REPLICAS_PER_STAGE = 2
NUM_WORKER_HANDLERS = 4
TRAINER_LOG_EVERY = 50 # batches
POLL_INTERVAL_FOR_FREE_WORKER = 0.002 # seconds
DEFAULT_BATCHES_PER_REDUCE = 10
RESOLVE_EXPERT_TIMEOUT=60
HOST_MADDRS = ["/ip4/127.0.0.1/tcp/0"]
WORKER_SERVER_UPDATE_PERIOD = 5.0
# All-reduce timings, in seconds. The ordering REQUEST_TIMEOUT <
# MIN_MATCHMAKING_TIME < AVERAGING_TIMEOUT is enforced
MIN_MATCHMAKING_TIME = 2 # seconds
REQUEST_TIMEOUT = 1 # seconds
# Generous because it bounds the whole round, and a round that hits it drops a
# group of gradients.
AVERAGING_TIMEOUT = 120.0 # seconds
