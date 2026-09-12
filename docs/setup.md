# Environment setup

Everything in this project runs inside **WSL2 Ubuntu**, driven from a Windows host.

## Quick start

From a WSL2 Ubuntu shell:

```bash
cd /mnt/c/Users/<you>/Desktop/pytorch-exercise
./scripts/setup.sh
```

The script is idempotent. It installs `uv`, provisions CPython 3.11, creates a
virtualenv at `~/.venvs/pluralis`, installs the pinned dependency set, builds
hivemind from the mandated git hash, and then verifies the whole thing.

Afterwards:

```bash
source ~/.venvs/pluralis/bin/activate
python scripts/verify_env.py     # re-run the checks at any time
```

To drive it from a Windows terminal without opening a WSL shell:

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/c/Users/<you>/Desktop/pytorch-exercise && ./scripts/setup.sh'
```

## Verified configuration

| Component | Version |
|---|---|
| OS | Ubuntu 26.04 LTS (WSL2), 12 cores / 9.6 GB RAM |
| Python | 3.11.16 (provisioned by `uv`) |
| torch | 2.14.0+cpu |
| torchvision | 0.29.0+cpu |
| hivemind | 1.2.0.dev0 @ `4d5c41495be082490ea44cce4e9dd58f9926bb4e` |
| p2pd | `p2pd-linux-amd64` v0.5.0.hivemind1 (prebuilt, sha256-verified) |

Exact pins for everything else live in [`requirements.lock.txt`](../requirements.lock.txt).

## Why the setup looks like this

Each of these was a forced move, not a preference.

**Why WSL2 and not native Windows.** hivemind's peer-to-peer layer is the
`p2pd` libp2p daemon, which ships prebuilt for Linux and macOS only. There is no
Windows binary and no supported Windows build path. `scripts/setup.sh` refuses to
run anywhere but Linux so this fails loudly rather than three steps later.

**Why Python 3.11 from `uv` instead of apt.** hivemind's `setup.py` declares
support for 3.9 through 3.12. Ubuntu 26.04's archive contains exactly one Python,
3.14, which is outside that range; there is no 3.10/3.11 package to install. `uv`
fetches a standalone 3.11 build, which sidesteps third-party PPAs entirely.

**Why the CPU torch wheel.** The default PyPI `torch` bundles the CUDA runtime,
roughly 2.5 GB of libraries that are dead weight on a CPU-only laptop. The
lockfile carries `--extra-index-url https://download.pytorch.org/whl/cpu` so the
CPU build is selected automatically.

**Why `--no-build-isolation`.** hivemind's `setup.py` imports `grpc_tools.protoc`
(to compile its protobufs) and `pkg_resources` at build time, but the project
declares no `[build-system]` requires. A PEP 517 isolated build environment
therefore lacks both and the build fails. We install `grpcio-tools`, `wheel` and
`setuptools<81` into the venv and build without isolation. The `<81` bound
matters: `pkg_resources` was removed in setuptools 81.

**Why no Go toolchain.** hivemind's `setup.py` defaults to
`download_p2p_daemon()`, which fetches a prebuilt, sha256-checked p2pd binary. Go
is only needed for the `--buildgo` path, which we do not use.

**Why the venv is not in the repo.** The repo sits on `/mnt/c` so it stays
editable from Windows tooling, but `/mnt/c` is a DrvFs mount: slow for the many
small files a venv contains, and prone to file-locking errors mid-install. The
venv lives on the Linux filesystem at `~/.venvs/pluralis` instead. `setup.sh`
rejects a `VENV_DIR` under `/mnt/`.

## pyproject.toml vs requirements.lock.txt vs setup.sh

Three files that all look like "dependencies" but answer different questions:

| File | Question it answers |
|---|---|
| `pyproject.toml` | *What is this package?* Its name, its importable layout, its entry points, and what it depends on in the abstract. |
| `requirements.lock.txt` | *Which exact versions are known to work?* Pins for the whole transitive tree, generated from a verified venv. |
| `scripts/setup.sh` | *How do I construct the environment?* The ordering and flags needed to get from a bare Ubuntu to a working install. |

`setup.sh` builds the environment; `pyproject.toml` is what makes `src/pytorch_exercise/` importable at all, since a `src/` layout is not on `sys.path` by default. The last step of `setup.sh` installs this project **editable** (`pip install -e`), so source edits take effect immediately and any process can import the package regardless of its working directory — which matters because workers and trainers run as separate processes, with hivemind forking more beneath them.

Both the hivemind and project installs pass `--no-deps`, because `requirements.lock.txt` is the single source of truth for versions. The dependency list in `pyproject.toml` is therefore declarative: it documents what the package needs, but does not drive what gets installed.

`pyproject.toml` deliberately does **not** list hivemind. Naming it would make a normal resolve fetch an unrelated release from PyPI, whereas the assignment mandates one specific git hash built with `--no-build-isolation`.

## What `verify_env.py` proves

Three checks, cheapest first, so a failure points at the right layer:

1. **imports and p2pd binary** — hivemind and torch import, the daemon binary
   exists and is executable, and this project's package is importable by name
   (proving the editable install landed, not just that the repo is the cwd).
2. **DHT peer discovery** — two DHT peers find each other over real libp2p; one
   stores a value, the other reads it back. This is the discovery path a trainer
   uses to locate workers.
3. **remote expert forward/backward** — a server declares a module to the DHT, a
   client resolves it by UID and runs a forward *and* a backward through it, with
   gradients arriving back at the client's input tensor. This round trip is the
   exact mechanism the trainer/worker split depends on.

The gradient check deliberately uses a random linear probe, `(out * probe).sum()`,
rather than `out.sum()` or `out.square().mean()`. hivemind's `ffn` block ends in a
`LayerNorm`, whose output is forced to zero mean and unit variance — so those
simpler losses are essentially invariant to the input and yield a ~0 gradient.
That is indistinguishable from a backward pass that silently returns nothing, so
the check would pass vacuously.

## Known quirks

**Noisy shutdown.** On teardown hivemind raises
`RuntimeError: There is no current event loop in thread 'MainThread'` from
`ControlClient.__del__`, via uvloop. It is cosmetic and fires after the work is
done, but it will clutter logs, so services should shut their DHT and servers
down explicitly rather than relying on interpreter exit.

**Separate handler processes per expert.** Starting a server logs something like:

```
verify.0: FeedforwardBlock, 33216 parameters
verify.0_forward starting, pid=1674
verify.0_backward starting, pid=1675
```

hivemind forks a distinct process for each direction of each expert. Any logging
or metrics strategy has to account for that — records come from several processes
per worker, not one.

**torch is newer than this hivemind expects.** The pinned hivemind commit is from
May 2025; torch 2.14 is considerably newer and already emits a
`torch.jit.script is deprecated` warning on import. Nothing is broken today, but
if something inexplicable turns up later, pinning torch to a 2.7-era release is
the first thing to try.

## Regenerating the lockfile

```bash
uv pip freeze --python ~/.venvs/pluralis/bin/python | grep -v '^hivemind' | sort
```

hivemind is excluded on purpose: it is built from the pinned git hash by
`setup.sh`, not installed from an index, so a `file:///` path in the lockfile
would only be valid on one machine.
