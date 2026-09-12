# Step 2 spec — one worker, one stage, one trainer

**Status:** draft for implementation. **Depends on:** step 1 (reference run, merged in `main`).
**Produces:** a remote worker that hosts a model stage over hivemind, a trainer that
drives it, and a comparison showing the distributed loss curve matching the reference.

This document is written so that it can be implemented without further design
decisions. Where a choice was made, the reason is given so the implementer can
recognise when a situation the spec did not anticipate calls for the same reasoning.

---

## 0. Purpose and scope

### What step 2 proves

1. **Our module survives the wire.** `scripts/verify_env.py` only ever proved that
   hivemind's demo `ffn` block can be called remotely. Step 2 proves that *our*
   `SimpleMLP`, with *our* input schema, can be hosted, resolved by UID, and driven
   through forward and backward from another process.
2. **The accumulate-don't-step seam works.** hivemind's `ModuleBackend` steps its
   optimizer after *every* backward call. The assignment needs the opposite:
   accumulate gradients locally and step only when a target batch size is reached.
   Step 2 introduces the subclass that does this and proves it numerically.
3. **The distributed loop reproduces the reference.** With one worker hosting the
   *whole* model, one trainer, and a target batch size equal to the reference batch
   size, the distributed system performs exactly the same computation as
   `train_reference()`. The two loss curves must therefore agree to floating-point
   tolerance — not "look similar". This is the sharpest test available and it is
   free, so we take it.

### Why the whole model is hosted as one stage

Hosting only stage 0 (fc1) would leave nowhere to compute a loss: the trainer holds
no weights by design, so it cannot run fc2/fc3 locally. Hosting the *full* model as
a single stage keeps the trainer weight-free, exercises the identical RPC path that
the two-stage version will use, and gives an exact comparison target. Step 3 splits
the model; nothing in the worker or trainer written here has to change for that
beyond the stage name.

### Explicitly out of scope

- The two-stage split (`stage0` / `stage1`) — step 3.
- More than one worker per stage, gradient averaging, `GradientAverager`,
  `ProgressTracker` — step 4.
- More than one trainer — step 5. (The worker written here already tolerates it.)
- Compression — step 6. (The schema plumbing written here is where it will plug in.)
- Worker selection, timeouts, fault tolerance — step 7.
- `config.py` — deferred by decision; constants stay in `reference.py` for now.

---

## 1. Background the implementer needs

Everything below was verified against the pinned hivemind source
(`~/src/hivemind`, commit `4d5c414`). Line references are to that tree.

### 1.1 The process model

`hivemind.moe.Server` is not one process. For a worker hosting one expert with
`num_connection_handlers=2`, expect **8 OS processes**:

| process | count | what it does |
|---|---|---|
| main Python process | 1 | runs the `Server` thread, and *inside that thread* the `Runtime` loop, which is where the module's `forward` and `backward` actually execute |
| `hivemind.DHT` | 1 | a forked child (`multiprocessing` `ForkProcess`) running the Kademlia node |
| `p2pd` | 1 | the Go libp2p daemon, spawned by the DHT child; everything else attaches to it over a unix socket |
| `ConnectionHandler` | 2 | forked children that accept RPCs and submit them to task pools |
| `TaskPool` | 2 | forked children named `<uid>_forward` and `<uid>_backward`; they only **batch** requests and dispatch futures — they never run the module |
| `torch_shm_manager` | 1 | torch's shared-memory helper; hivemind forces the `file_system` sharing strategy |

Consequences for this spec:

- **`StageBackend` state lives in the main process and is touched by one thread** (the
  Runtime loop). Counters and accumulators need no locks. Reading integer counters
  from the main thread for logging is fine.
- `ps` shows every child as `python`; the `<uid>_forward` names come from hivemind's
  own log lines, not from the OS.
- Every log line must say which process it came from (§8).

### 1.2 The RPC path

`hivemind/moe/client/expert.py`, `hivemind/moe/server/connection_handler.py`,
`hivemind/moe/server/module_backend.py`.

- `RemoteExpert` is an `nn.Module`. Its `forward` is a custom `autograd.Function`
  (`_RemoteModuleCall`). It sends inputs with `rpc_forward`; on `.backward()` it sends
  the *saved inputs plus the output gradients* with `rpc_backward` and receives input
  gradients. A dummy `requires_grad=True` tensor is threaded through so the backward
  RPC fires even when the real inputs do not require grad (ours will not: they are
  images).
- Payloads under 2 MiB (`MAX_UNARY_PAYLOAD_SIZE`) go as one message; larger ones
  stream. A batch of 64 MNIST images is ~200 KB. Not a concern at this step.
- Server side, `ModuleBackend.forward` runs the module under `torch.no_grad()`.
  `ModuleBackend.backward` **re-runs the forward** with grad enabled on detached
  inputs, calls `torch.autograd.backward(outputs, grad_tensors=grad_outputs)`, then
  calls `self.on_backward(batch_size)`, and returns `input.grad` for every input.
  Nothing is stashed between forward and backward — this is why forward and backward
  may arrive in any order and on any replica.
- `torch.autograd.backward` **accumulates** into `param.grad`. It does not overwrite.
  The default `ModuleBackend.on_backward` is:

  ```python
  def on_backward(self, batch_size: int) -> None:
      if self.optimizer is not None:
          self.optimizer.step()
          self.optimizer.zero_grad()
  ```

  That is the single method step 2 overrides.

### 1.3 Schemas

`ModuleBackend` requires `args_schema` — a tuple of `BatchTensorDescriptor` describing
each positional input with the batch dimension omitted — and derives
`outputs_schema` by running the module on a dummy batch if it is not given. Give it
explicitly; do not rely on the dummy run.

The client validates *structure* only (`nested_compare`: tuple lengths, dict keys),
**not shapes or dtypes**. A wrong shape reaches the server and fails or hangs there.
The trainer therefore asserts shapes itself before every call (§3.6).

`ModuleBackend` forwards unknown kwargs to `TaskPool`, which **requires
`max_batch_size`** and accepts `min_batch_size` (default 1) and `timeout`. Omitting
`max_batch_size` raises `TypeError` at construction.

### 1.4 Discovery

- `Server` starts a `DHTHandlerThread` that calls `declare_experts` immediately and
  then every `update_period` seconds, with expiration
  `max(2 * update_period, 3)` seconds. Defaults: 30 s / 60 s.
- `declare_experts` stores `uid -> peer_id`. For a flat uid such as `full.0` it does
  **not** store any listing under the prefix `full`. There is no way to enumerate
  experts under a prefix from the DHT; `get_experts(dht, [uid, ...])` with known
  uids is the discovery mechanism. Step 2 knows its one uid. (Later steps use the
  naming convention `<stage>.<index>` with a known replica count.)
- A freshly started trainer may resolve `None` for a few seconds until the
  declaration propagates. The trainer retries (§3.6).
- Expert uids must match `^[^.]+(\.\d+)+$` — a prefix, then one or more `.integer`
  coordinates. `full.0` is valid; `full` alone, or `full.a`, is not.

### 1.5 Peers and bootstrap

- A peer's identity is a libp2p `PeerID`, random per launch unless `identity_path`
  (a private-key file) is given. `host_maddrs=["/ip4/127.0.0.1/tcp/0"]` binds
  loopback on an OS-chosen port; two processes on one machine never collide.
- `initial_peers` is the only wiring. A DHT started with none becomes the seed of a
  new one-node network; others join by passing any live peer's
  `get_visible_maddrs()`. Joining is transitive. The seed is needed only while
  peers are joining, not afterwards.
- `Server.shutdown()` **also shuts down the DHT it was given**. Do not shut the
  worker's DHT down separately.

### 1.6 Logging

`hivemind.utils.logging.get_logger(name)` returns a standard `logging.Logger` and
installs hivemind's coloured handler. That handler applies only to the `hivemind.*`
loggers by default; `use_hivemind_log_handler("in_root_logger")` extends it to ours.
Its format has no process or role field:

```
{asctime}.{msecs} [{levelname}]{caller_block} {message}
```

Role and pid must therefore be injected into the message (§3.4). The env var
`HIVEMIND_LOGLEVEL` sets the level.

`Runtime` accepts `stats_report_interval=N`; when set, a `StatsReporter` thread logs
per-pool batches/s and examples/s every N seconds. Use it.

---

## 2. Target layout

```
src/swarm_mlp/
    __init__.py
    __main__.py        # reference CLI — behaviour unchanged, plotting moved out
    model.py           # + seed_everything, build_model (moved in), STAGE_SHAPES, build_stage
    data.py            # unchanged
    reference.py       # train_reference; LossCurve/seed/build moved out; + max_steps
    curves.py          # NEW  LossCurve + save/load
    plotting.py        # NEW  rolling_mean, plot_curve, plot_comparison (moved + one new)
    observability.py   # NEW  configure_logging, silence_teardown_noise
    worker.py          # NEW  StageBackend, serve(), CLI
    trainer.py         # NEW  train(), CLI
    compare.py         # NEW  overlay CLI
tests/
    __init__.py
    test_curves.py
    test_stage_backend.py
    test_round_trip.py        # marked integration
docs/
    step2-spec.md             # this file
results/
    reference_loss.png        # unchanged
    distributed_full.json     # NEW  produced by the runbook
    compare_step2.png         # NEW  produced by the runbook
```

`python -m swarm_mlp` remains the reference run. New entry points:
`python -m swarm_mlp.worker`, `python -m swarm_mlp.trainer`, `python -m swarm_mlp.compare`.

---

## 3. Component specifications

### 3.1 `curves.py` — `LossCurve`

Move `LossCurve` here from `reference.py` unchanged, then extend:

```python
@dataclass
class LossCurve:
    samples: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def record(self, samples: int, loss: float, accuracy: float) -> None: ...
    def __len__(self) -> int: ...

    def save(self, path: Path | str) -> Path:
        """Write as JSON: {"samples": [...], "loss": [...], "accuracy": [...], "meta": {...}}."""

    @classmethod
    def load(cls, path: Path | str) -> "LossCurve": ...
```

**Why persistence now.** The reference curve is obtained by calling a function in
process. The distributed curve is produced *inside the trainer process* and has to
cross a process boundary to reach the comparison script. `save`/`load` is that
boundary and nothing more: `train()` and `train_reference()` still return curves and
write nothing; only CLIs persist.

`meta` is free-form but the producers below populate at least:
`source` (`"reference"` | `"distributed"`), `epochs`, `batch_size`, `seed`,
`learning_rate`, and for distributed runs `expert_uid`, `target_batch_size`.

`save` creates parent directories. `load` raises `FileNotFoundError` naturally.
Round-trip must be exact for floats (JSON `float` repr is; do not round).

### 3.2 `plotting.py`

Move `rolling_mean` and `plot_curve` here from `__main__.py` **verbatim** (the
`window % 2` parity handling included). Add:

```python
def plot_comparison(
    reference: LossCurve,
    distributed: LossCurve,
    output: Path,
    *,
    smooth: int = 50,
    title: str = "",
) -> Path:
```

Three stacked panels sharing the x-axis (samples consumed):

1. **Loss** — reference raw (alpha 0.15) and smoothed in one colour; distributed raw
   and smoothed in another. Legend shows the source names.
2. **Accuracy** — same treatment.
3. **|Δ loss|** — absolute per-step difference over the common prefix
   (`min(len(reference), len(distributed))` steps), log-scale y-axis, with a
   horizontal line at the tolerance passed by the caller (default `1e-3`). This panel
   is the one a reviewer reads; the other two are context.

Use `matplotlib.use("Agg")` before importing pyplot, exactly as `__main__.py` does.
Move the `matplotlib.use` call into `plotting.py` so it happens once, at the top of
that module, before `import matplotlib.pyplot`.

### 3.3 `model.py` additions

Move `seed_everything` and `build_model` here from `reference.py` unchanged
(`reference.py` imports them from here). Add:

```python
# (input shape without batch dim, output shape without batch dim)
STAGE_SHAPES: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "full": ((1, 28, 28), (10,)),
}

def build_stage(stage: str, seed: int) -> nn.Module:
    """The module a worker for `stage` hosts, deterministically initialised.

    "full" is the entire SimpleMLP. Step 3 adds "stage0" and "stage1", carved
    from ONE seeded SimpleMLP so that replicas and the reference share initial
    weights. Unknown names raise ValueError listing the known stages.
    """
```

`build_stage("full", seed)` **must** return exactly `build_model(seed)` — same seed
handling, same construction order — so that the worker's initial weights are
bit-identical to the reference's. Do not construct anything that consumes torch RNG
between `seed_everything` and the module constructor.

`model.py` stays free of hivemind imports. Shapes are plain tuples; `worker.py`
turns them into `BatchTensorDescriptor`s.

### 3.4 `observability.py`

```python
def configure_logging(role: str, level: str | None = None) -> logging.Logger:
```

- Calls `use_hivemind_log_handler("in_root_logger")`.
- Level: `level` argument if given, else `HIVEMIND_LOGLEVEL` env var, else `INFO`.
  Applied to the root logger.
- Installs a `logging.Filter` that prepends `[<role> pid=<os.getpid()>] ` to
  `record.msg`. It must go on the **handler**, not the logger: logger-level filters
  do not see records propagated from child loggers, handler-level filters do.
  After `use_hivemind_log_handler("in_root_logger")` hivemind's handler is the
  `StreamHandler` attached to the root logger, so:

  ```python
  for handler in logging.getLogger().handlers:
      handler.addFilter(RoleFilter(role))
  ```

  `os.getpid()` is evaluated per record, so forked children (task pools, handlers)
  report their own pid without any extra configuration — fork copies the logging
  setup. Verified: with this in place, hivemind's own `demo.0_forward starting`
  line arrives as `[worker[demo.0] pid=41911]` from the pool child, and
  `Server started with 1 modules:` as `[worker[demo.0] pid=41814]` from the main
  process. WARN/ERROR records additionally carry hivemind's `[module.func:line]`
  caller block before our prefix; that is hivemind's formatter and is fine.
- `use_hivemind_log_handler` sets the root level from `HIVEMIND_LOGLEVEL` itself;
  set ours *after* calling it so an explicit `level` argument wins.
- Calls `sys.stdout.reconfigure(line_buffering=True)` (see the comment in
  `scripts/verify_env.py` for why: hivemind logs to stderr unbuffered, and stdout is
  block-buffered when piped).
- Returns `get_logger(f"swarm_mlp.{role}")`.

`role` is a short tag like `worker[full.0]`, `trainer`, `compare`.

```python
def silence_teardown_noise() -> None:
```

Copy from `scripts/verify_env.py` verbatim, with its docstring. Leave the copy in
`verify_env.py` alone — scripts stay standalone.

### 3.5 `worker.py`

#### `StageBackend(ModuleBackend)`

```python
class StageBackend(ModuleBackend):
    """A pipeline stage that accumulates gradients and steps at a target batch size.

    hivemind's ModuleBackend steps its optimizer after every backward call. This
    subclass instead keeps a sample-weighted running sum of gradients and steps
    only once `target_batch_size` samples have been seen since the last step.
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
        **pool_kwargs,          # forwarded to ModuleBackend -> TaskPool; must include max_batch_size
    ) -> None
```

State, all plain attributes:

| attribute | type | meaning |
|---|---|---|
| `target_batch_size` | int | step threshold |
| `samples_since_step` | int | samples accumulated toward the next step |
| `samples_total` | int | lifetime samples seen in backward |
| `backward_calls` | int | lifetime backward invocations |
| `steps` | int | lifetime optimizer steps |
| `last_effective_batch` | int | samples consumed by the most recent step |
| `_accumulators` | list[Tensor] | one zero tensor per parameter, same shape/dtype |

`on_backward(self, batch_size: int) -> None` — the only overridden method:

```
1. for each (param, acc) pair:
       if param.grad is not None:
           acc.add_(param.grad, alpha=float(batch_size))
           param.grad = None
2. samples_since_step += batch_size
   samples_total      += batch_size
   backward_calls     += 1
3. if samples_since_step >= target_batch_size:
       for each (param, acc):
           param.grad = acc / samples_since_step        # sample-weighted mean
       optimizer.step()
       for each (param, acc): param.grad = None; acc.zero_()
       steps += 1
       last_effective_batch = samples_since_step
       samples_since_step = 0
       log (INFO): step, effective batch, samples_total, grad norm (computed
           from the means before zeroing — cheap for 235k params)
```

**Why weight by `batch_size`.** The trainer's loss uses `reduction="mean"`, so the
gradient arriving from each backward call is already divided by *that call's* batch
size. Summing two such gradients from batches of 16 and 48 and dividing by 2 gives
the wrong answer; weighting each by its batch size and dividing by 64 gives exactly
the gradient of the mean loss over all 64 samples. This is the same normalisation
`hivemind.optim.GradientAverager.accumulate_grads_` performs (it scales by
`batch_size / anchor_batch_size`), so step 4 can swap the manual accumulator for the
averager without changing semantics.

**Why `>=` and reset-to-zero, not modulo and not carry-over.** SWARM's fallback used
`examples_processed % target == 0`, which never fires when batch sizes do not divide
the target (verified: with variable batches it stepped 6 times where 23 were due).
Carrying a remainder over is also wrong: the gradient for those samples has already
been applied by the step. So: step whenever the threshold is met, consume everything
accumulated, start again from zero. The effective batch may exceed the target when a
large request lands; it is logged and recorded in `last_effective_batch`.

`get_stats(self) -> dict[str, int]` returns the five counters by name. It exists for
logging and tests; nothing else reads it at this step.

Do **not** pass `optimizer=` to `ModuleBackend.__init__`. Keep the optimizer as
`self.optimizer` on the subclass instead. Reason: `ModuleBackend.__init__` asserts
nothing about it, but keeping the base class unaware of the optimizer guarantees no
base-class code path can ever call `step()`.

Thread-safety: `on_backward` runs in the Runtime loop only. No locks.

#### `serve()`

```python
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
    max_batch_size: int = 4096,
) -> tuple[hivemind.DHT, Server, StageBackend]:
```

Constants come from `swarm_mlp.reference` (`BATCH_SIZE`, `LEARNING_RATE`, `SEED`).

Steps, in order:

1. `uid = f"{stage}.{index}"`; validate with `hivemind.moe.expert_uid.is_valid_uid`,
   raise `ValueError` otherwise.
2. `module = build_stage(stage, seed)`.
3. `dht = hivemind.DHT(initial_peers=list(initial_peers) or None, host_maddrs=list(host_maddrs), identity_path=identity_path, start=True)`.
   (`identity_path` is forwarded through `DHTNode.create` to `P2P.create`; pass it
   only when not `None`.)
4. `in_shape, out_shape = STAGE_SHAPES[stage]`;
   `args_schema = (BatchTensorDescriptor(*in_shape),)`,
   `outputs_schema = BatchTensorDescriptor(*out_shape)`.
5. `backend = StageBackend(uid, module, optimizer=torch.optim.Adam(module.parameters(), lr=learning_rate), target_batch_size=..., args_schema=..., outputs_schema=..., min_batch_size=1, max_batch_size=max_batch_size)`.
6. `server = Server(dht, {uid: backend}, num_connection_handlers=num_handlers, update_period=update_period, device=torch.device("cpu"), stats_report_interval=stats_interval, start=True)`.
   `device` and `stats_report_interval` are Runtime kwargs, forwarded via `**kwargs`.
7. Log, at INFO, one line each: uid; PeerID; and the exact string
   `--initial-peers <maddr>` using `dht.get_visible_maddrs()[0]`, so it can be pasted
   into the trainer command.
8. Return `(dht, server, backend)`.

`update_period=5` rather than hivemind's 30 so a dead worker's declaration expires in
10 s instead of 60. The DHT traffic is negligible.

#### CLI

`python -m swarm_mlp.worker --stage full --index 0 [--initial-peers M ...] [--host-maddrs M ...] [--identity-path P] [--target-batch-size N] [--learning-rate F] [--seed N] [--num-handlers N] [--update-period S] [--stats-interval S] [--log-level L]`

- `configure_logging(f"worker[{stage}.{index}]", level)` first thing.
- `silence_teardown_noise()`.
- Call `serve(...)`.
- Block on a `threading.Event` that `SIGINT`/`SIGTERM` handlers set. On wake, call
  `server.shutdown()` only (it shuts the DHT down), log `"worker stopped"`, exit 0.
  Expect `Server shutdown successfully` to appear **twice** and one
  `ConnectionHandler shutdown had no effect, the process is already dead` warning:
  our call makes `Runtime.run` return, and `Server.run`'s own `finally` then calls
  `shutdown()` a second time. Harmless; do not try to suppress it.
- Log the DHT child's pid at startup (`dht.pid`) so its log lines can be matched to
  a process in `ps`.
- Every `--stats-interval` seconds (if set), also log `backend.get_stats()` from the
  main thread — this is *our* view (steps, samples) alongside hivemind's
  `StatsReporter` view (batches/s).

### 3.6 `trainer.py`

#### `train()`

```python
def train(
    *,
    expert_uid: str,
    initial_peers: Sequence[str],
    epochs: int = 1,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    max_steps: int | None = None,
    log_every: int = 100,
    resolve_timeout: float = 60.0,
    dht: hivemind.DHT | None = None,
) -> LossCurve:
```

The trainer holds no weights and no optimizer. Steps:

1. `dht = dht or hivemind.DHT(initial_peers=list(initial_peers), start=True)`. Remember
   whether we created it; shut it down in a `finally` only if we did.
2. Resolve: loop until `get_experts(dht, [expert_uid])[0]` is not `None`, sleeping
   0.5 s between attempts, up to `resolve_timeout`; raise `TimeoutError` naming the
   uid. Log which PeerID it resolved to.
3. Look up `expert_uid.rsplit(".", 1)[0]` in `STAGE_SHAPES` to get the expected input
   shape; raise `ValueError` if the stage is unknown.
4. `loader = mnist_train_loader(batch_size=batch_size, seed=seed)` — the same call the
   reference makes, so the batch order is identical.
5. `criterion = nn.CrossEntropyLoss()`.
6. Loop `for epoch in range(epochs): for images, labels in loader:`
   - `assert tuple(images.shape[1:]) == in_shape` (see §1.3 for why).
   - `logits = expert(images)`; `loss = criterion(logits, labels)`; `loss.backward()`.
     No `zero_grad`, no `step`: there is nothing local to zero or step.
   - `samples_seen += labels.size(0)`; record `(samples_seen, loss.item(), batch_accuracy)`
     exactly as `train_reference` does.
   - Log at INFO every `log_every` steps: epoch, step, samples, loss, acc, and
     steps/s over the interval.
   - Stop when `max_steps` recorded points exist.
7. Populate `curve.meta` (§3.1) and return it.

**What `loss.backward()` does here.** `logits` came out of `_RemoteModuleCall`, so
autograd calls its `backward`, which sends `(images, dlogits)` to the worker's
`rpc_backward`. The worker recomputes the forward, backpropagates into its
parameters, calls `StageBackend.on_backward`, and returns `dimages` — which the
trainer discards. The remote step happens on the worker's schedule, not the
trainer's.

**Determinism.** Weight initialisation happens on the worker (`build_stage`), data
order comes from the loader's explicit generator. With `seed=0` on both sides, the
trainer sees exactly the batches the reference saw, in order, against exactly the
weights the reference had.

#### CLI

`python -m swarm_mlp.trainer --expert full.0 --initial-peers M [...] [--epochs N] [--batch-size N] [--seed N] [--max-steps N] [--log-every N] [--output results/distributed_full.json]`

- `configure_logging("trainer", level)`, `silence_teardown_noise()`.
- Call `train(...)`, then `curve.save(args.output)`, then log the path and the final
  50-step mean loss/accuracy in the same format `__main__.py` prints for the
  reference.

### 3.7 `compare.py`

`python -m swarm_mlp.compare results/distributed_full.json [--reference results/reference.json] [--output results/compare_step2.png] [--tolerance 1e-3] [--smooth 50]`

1. `distributed = LossCurve.load(path)`.
2. If `--reference` is given, load it. Otherwise call
   `train_reference(epochs=meta["epochs"], batch_size=meta["batch_size"], seed=meta["seed"], learning_rate=meta["learning_rate"], max_steps=len(distributed), log_every=0)`
   — reading every parameter from the distributed curve's `meta` so the comparison
   is against the matching configuration by construction, not by convention.
3. Over the common prefix `n = min(len(reference), len(distributed))`:
   - assert `reference.samples[:n] == distributed.samples[:n]` — if the sample axes
     differ, the runs are not comparable and the script exits 2 with a message
     saying so.
   - `max_abs = max(|Δloss|)`, `mean_abs = mean(|Δloss|)`, and the index of the
     first step where `|Δloss| > tolerance` (or `None`).
4. Print those three numbers plus the final 50-step mean loss of each curve.
5. `plot_comparison(...)`, print the output path.
6. Exit 0 if `max_abs <= tolerance`, else 1. Print which.

The reference run is ~20 s per epoch, so recomputing it is acceptable; `--reference`
exists so a long comparison need not repeat it.

### 3.8 `__main__.py` and `reference.py` changes

Refactor only. After this change:

- `reference.py` imports `LossCurve` from `curves`, `seed_everything`/`build_model`
  from `model`, and gains `max_steps: int | None = None` (stop after that many
  recorded steps; `None` means run all epochs). It also populates `curve.meta`.
- `__main__.py` imports `rolling_mean`/`plot_curve` from `plotting`; its argparse,
  output and printed summary are unchanged.
- `python -m swarm_mlp` with default arguments prints the same summary line as
  before the refactor. On the reference machine that is:

  ```
  2814 steps, 180000 samples consumed. Final 50-step mean: loss 0.0753, accuracy 0.9772
  ```

  and the first three recorded losses are `2.293397, 2.30167, 2.258826`. If either
  differs, the refactor changed behaviour; stop and find out why before continuing.

---

## 4. The contract between trainer and worker

| item | value | where enforced |
|---|---|---|
| expert uid | `<stage>.<index>`, e.g. `full.0` | `serve()` validates; trainer parses |
| input | `float32`, shape `(B, 1, 28, 28)`, values in `[0, 1]` | `STAGE_SHAPES["full"]`; trainer asserts; worker schema |
| output | `float32`, shape `(B, 10)` logits — no softmax | `STAGE_SHAPES["full"]`; worker schema |
| gradient sent back | `dlogits`, shape `(B, 10)`, already divided by `B` | consequence of `CrossEntropyLoss(reduction="mean")` |
| batch-size contract | `target_batch_size == trainer batch_size == reference BATCH_SIZE == 64` | both default from `swarm_mlp.reference`; the runbook does not override either |
| optimizer | Adam, `lr = LEARNING_RATE`, default betas/eps | `serve()`; matches `train_reference` |
| seed | `SEED = 0` on both sides | defaults |

The batch-size row is the load-bearing one. If a future change makes
`target_batch_size` differ from the trainer's batch size, the distributed curve will
legitimately differ from the reference and the comparison in §6 stops meaning what
it means here. Any such change must re-run and re-commit the reference alongside.

---

## 5. Runbook

Two terminals, both with `source ~/.venvs/pluralis/bin/activate` and the repo root
as cwd.

**Terminal 1 — worker**

```bash
python -m swarm_mlp.worker --stage full --index 0 --stats-interval 10
```

Expected within a few seconds (pids and addresses will differ):

```
Sep 12 18:02:11.412 [INFO] [worker[full.0] pid=41022] hosting full.0 as peer 12D3KooW...
Sep 12 18:02:11.413 [INFO] [worker[full.0] pid=41022] trainers join with: --initial-peers /ip4/127.0.0.1/tcp/44155/p2p/12D3KooW...
Sep 12 18:02:11.502 [INFO] [worker[full.0] pid=41022] Server started with 1 modules:
Sep 12 18:02:11.502 [INFO] [worker[full.0] pid=41022] full.0: SimpleMLP, 235146 parameters
```

**Terminal 2 — trainer, then comparison**

```bash
python -m swarm_mlp.trainer --expert full.0 --initial-peers /ip4/127.0.0.1/tcp/44155/p2p/12D3KooW... --epochs 1
python -m swarm_mlp.compare results/distributed_full.json
```

Expected trainer output shape:

```
... [trainer pid=41310] resolved full.0 -> 12D3KooW... after 0.2s
... [trainer pid=41310] epoch 1/1  step    0  samples     64  loss 2.2934  acc 0.109
... [trainer pid=41310] epoch 1/1  step  100  samples   6464  loss 0.3640  acc 0.875  (36.2 steps/s)
...
... [trainer pid=41310] 938 steps, 60000 samples. Final 50-step mean: loss 0.1528, accuracy 0.9553
... [trainer pid=41310] curve written to results/distributed_full.json
```

The first-step loss `2.2934` and the epoch-1 final mean `0.1528 / 0.9553` are the
reference's own numbers (see the step-1 smoke run); with the contract in §4 they
should reappear here to the printed precision.

Meanwhile terminal 1 shows one line per optimizer step and, every 10 s, both
hivemind's pool throughput and our counters:

```
... [worker[full.0] pid=41022] step 100  effective_batch 64  samples_total 6400  grad_norm 0.4127
... [worker[full.0] pid=41022] Processed 730 batches in last 10 seconds:
... [worker[full.0] pid=41022] full.0_forward: 365 batches (36.50 batches/s), 23360 examples (2336.00 examples/s), avg batch size 64.00
... [worker[full.0] pid=41022] full.0_backward: 365 batches ...
... [worker[full.0] pid=41022] stats: steps=100 samples_total=6400 backward_calls=100 samples_since_step=0
```

Expected comparison output:

```
common prefix: 938 steps
max |Δloss| = 0.000e+00   mean |Δloss| = 0.000e+00   first step over 1e-3: none
reference   final 50-step mean loss 0.1528
distributed final 50-step mean loss 0.1528
plot written to results/compare_step2.png
PASS (tolerance 1e-3)
```

One epoch takes about 30 s at ~36 steps/s (measured with the prototype on the
reference machine; each step is one forward RPC, one backward RPC, and a recompute).

Stop the worker with Ctrl-C. The `RuntimeError: There is no current event loop`
lines on exit are the cosmetic teardown noise documented in `docs/setup.md`;
`silence_teardown_noise()` removes them but hivemind's child processes may still
print a few.

---

## 6. Acceptance criteria

All of the following, on the reference machine:

1. **Refactor is behaviour-preserving.** `python -m swarm_mlp` prints the summary
   line in §3.8 and the first three losses match.
2. **Unit tests pass.** `pytest` (which excludes integration tests by default, §7)
   is green.
3. **Integration test passes.** `pytest -m integration` is green.
4. **The runbook reproduces the reference.** After §5, `compare` reports
   `max |Δloss| <= 1e-3` over all 938 steps of epoch 1 and exits 0. The expected
   value is **exactly `0.0`**: a prototype of this exact configuration measured
   `max |Δloss| = 0.000e+00` over 200 steps, with the first three and last three
   losses matching the reference to all printed digits. The threshold is generous so
   the check does not flake on a future torch upgrade; but anything above `1e-6`
   should be investigated even though it passes, because it means the two
   computations are no longer the same computation.
5. **Artifacts are committed.** `results/distributed_full.json` and
   `results/compare_step2.png` from the run in (4).
6. **Every log line from every process carries a role and a pid** (§8). Verify by
   grepping the worker's output for lines that lack `pid=`: there should be none
   from our loggers. (hivemind's own child-process log lines that predate
   `configure_logging` — there are a handful at DHT start — are acceptable.)
7. **`torch.distributed` is not imported anywhere.** `grep -rn "torch.distributed" src/ tests/` returns nothing.

---

## 7. Tests

`pyproject.toml` gains:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not integration'"
markers = ["integration: starts real DHT/p2pd processes; slow (~1-2 min)"]
```

### `tests/test_curves.py`

- `save` then `load` round-trips a curve with non-trivial floats exactly
  (`==` on the lists, not `allclose`).
- `save` creates a missing parent directory.
- `meta` round-trips.

### `tests/test_stage_backend.py` — no network, no DHT

Helper: build a tiny deterministic module (e.g. `nn.Sequential(nn.Flatten(), nn.Linear(4, 3))`
seeded), wrap it in a `StageBackend` with `target_batch_size=64`, `max_batch_size=4096`,
and drive `backend.backward(x, grad_out)` **directly** — the same entry point the
Runtime uses — where `grad_out` is obtained by running the module locally with grad,
computing `CrossEntropyLoss(reduction="mean")`, and taking `torch.autograd.grad(loss, logits)`.
This emulates precisely what the trainer sends.

1. **`test_accumulates_until_target`** — micro-batches of 16 then 48. After the first:
   `steps == 0`, parameters unchanged, `samples_since_step == 16`. After the second:
   `steps == 1`, parameters changed, `samples_since_step == 0`,
   `last_effective_batch == 64`, every `param.grad is None`, every accumulator is
   all-zeros.
2. **`test_weighted_mean_equals_full_batch`** — two backends with identical initial
   weights. Backend A receives micro-batches `(16, 48)` of some data; backend B
   receives the same 64 samples as one batch. After both step,
   `torch.allclose(a_param, b_param, atol=1e-6)` for every parameter. This is the
   test that proves the sample weighting; if someone replaces it with `/ k`, it fails.
   (Measured with the prototype: Adam gives a difference of exactly `0.0`, plain SGD
   `3.7e-9`, so `1e-6` has ample slack.)
3. **`test_oversized_batch_steps_once`** — one backward of 100 samples with target 64:
   `steps == 1`, `last_effective_batch == 100`, `samples_since_step == 0`.
4. **`test_variable_batches_step_count`** — feed 50 micro-batches drawn from
   `{16, 24, 32, 48}` (fixed seed); assert `steps == floor(cumulative / 64)` computed
   with the same threshold rule the spec defines. (Guards against the modulo bug
   coming back.)
5. **`test_get_stats_keys`** — the five counter names are present.

### `tests/test_round_trip.py` — `@pytest.mark.integration`

In one test, in-process (no subprocesses to manage; the DHT and Server still fork
their own children, which is what makes it an integration test):

1. `dht_seed = hivemind.DHT(host_maddrs=["/ip4/127.0.0.1/tcp/0"], start=True)`.
2. `_, server, backend = serve(stage="full", index=0, initial_peers=dht_seed.get_visible_maddrs(), num_handlers=1)`.
3. `distributed = train(expert_uid="full.0", initial_peers=..., max_steps=50, log_every=0)`.
4. `reference = train_reference(max_steps=50, log_every=0)`.
5. Assert: `distributed.samples == reference.samples`;
   `max(abs(d - r)) <= 1e-5` over the 50 losses (measured: exactly `0.0`);
   `distributed.loss[-1] < distributed.loss[0]`; `backend.steps == 50`;
   `backend.samples_total == 3200`.
6. `finally`: `server.shutdown()`, `dht_seed.shutdown()`.

Budget: about 15 s (DHT/p2pd startup dominates; 50 steps take ~1.5 s). If it takes
over a minute, the expert-resolution loop or a hung shutdown is the first place to
look.

---

## 8. Observability requirements

- Every line our code logs goes through `configure_logging` and therefore carries
  `[<role> pid=N]`. The role for a worker includes its uid.
- The worker logs **one line per optimizer step** with: step number, effective batch,
  lifetime samples, gradient norm. This is the signal that will later show whether
  averaging changed anything.
- The worker logs the paste-able `--initial-peers` line at startup.
- The trainer logs resolution (uid → PeerID, and how long it took), periodic progress
  with steps/s, and the output path.
- `--stats-interval` on the worker turns on hivemind's per-pool throughput report and
  our counter dump at the same cadence, so the two can be read side by side.
- `HIVEMIND_LOGLEVEL=DEBUG` should make hivemind's own `Processing batch N from pool
  X` lines appear without any code change.

Nothing else at this step: no metrics files, no dashboards. The requirement is that
when something goes wrong across eight processes, the logs say which process and
what it was doing.

---

## 9. Known limitations, deliberately accepted

- **No timeouts on RPCs.** If the worker dies mid-run, the trainer's next call hangs.
  Fault handling is step 7's concern; do not add retries here.
- **Input gradients are computed and shipped back for nothing.** hivemind always
  returns `dinputs`; for image inputs that is `B × 784` floats per backward that
  nobody reads. Roughly doubles backward payload. Acceptable now; note it.
- **Recompute.** Every backward re-runs the forward on the worker. Expected and by
  design (§1.2); it means the worker does ~3 forward-equivalents per sample.
- **The worker is the bootstrap peer.** Fine for one worker. A dedicated bootstrap
  process is a later step.
- **`identity_path` is optional and unused in the runbook.** It exists so a scripted
  launcher can have a stable `--initial-peers` string; nothing depends on it yet.
- **`ModuleBackend.backward` and `on_backward` come from hivemind's `Runtime` loop
  in the Server thread.** If anything ever calls them from another thread, the
  no-locks assumption in §3.5 breaks. Nothing in this spec does.

---

## 10. Implementation order

Each numbered item is a commit. Do not start the next until the previous one's
check passes.

1. **Refactor.** `curves.py`, `plotting.py`, moves into `model.py`, `max_steps` on
   `train_reference`, `meta` population. Check: §6.1 regression numbers.
   Add `tests/test_curves.py`; check: `pytest` green.
2. **`observability.py`.** Check: a three-line script that calls
   `configure_logging("demo")` and logs once prints `[demo pid=N]`.
3. **`worker.py` — `StageBackend` only**, plus `tests/test_stage_backend.py`.
   Check: `pytest` green. Do not write `serve()` yet; the backend is testable alone
   and that is the point.
4. **`worker.py` — `serve()` and the CLI.** Check: start it, see the four expected
   log lines from §5, Ctrl-C exits cleanly.
5. **`trainer.py`**, plus `tests/test_round_trip.py`. Check: `pytest -m integration`
   green.
6. **`compare.py`**, run the §5 runbook end to end, commit the two artifacts.
   Check: §6.4 and §6.5.
7. **Docs.** README gains a "Running the distributed system" section with the
   runbook; `docs/` gains a short `step2.md` recording what was measured (the actual
   `max |Δloss|`, steps/s, process count) — numbers, not prose. Update the README
   layout block.

Commit messages follow the existing style in `git log`: an imperative subject
line, a body that explains *why*, not what.

---

## 11. Definition of done

Steps 1–7 of §10 committed on `distributed-pipeline`; §6 fully satisfied; the
branch pushed. Open questions discovered along the way go in `docs/step2.md` under
"Open questions", with enough context that they can be picked up cold.
