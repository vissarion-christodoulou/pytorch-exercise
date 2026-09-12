# Distributed MLP training with hivemind

A peer-to-peer system that trains a 3-layer MLP on MNIST across two pipeline
parallel stages, with two data-parallel workers per stage, built on
[hivemind](https://github.com/learning-at-home/hivemind). `torch.distributed` is
not used anywhere.

The system has two services:

- **worker** — owns one pipeline stage's weights, serves `forward` and `backward`
  for that stage, and accumulates gradients locally. When the workers in a stage
  collectively reach the target batch size they all-reduce gradients and step.
- **trainer** — holds no weights. It samples data batches and routes
  activations and gradients between workers.

## Setup

Everything runs inside WSL2 Ubuntu. From a WSL shell:

```bash
./scripts/setup.sh
source ~/.venvs/pluralis/bin/activate
```

See [docs/setup.md](docs/setup.md) for the verified configuration, the reasoning
behind each choice, and known quirks.

## Verifying the environment

```bash
python scripts/verify_env.py
```

Checks that hivemind imports, that two DHT peers can discover each other over
libp2p, and that a forward/backward round trip through a remotely hosted module
returns gradients to the caller.

## Reference baseline

Before the distributed system can be judged there has to be something to judge it
against. The single-process run trains the same `SimpleMLP` on the same MNIST
pipeline the workers and trainer will use:

```bash
python -m swarm_mlp
```

It prints a summary and writes [results/reference_loss.png](results/reference_loss.png).
Defaults are 3 epochs, batch 64, Adam at 1e-3, seed 0 — all overridable as flags.

Three properties make it usable as a control rather than just a nice graph:

- **Deterministic.** Same seed, same initial weights, same batch order; two runs
  produce bit-identical curves. The distributed system starts from the same
  seeded weights and consumes the same data, so the curves should very nearly
  coincide — a far sharper test than "both trend downwards".
- **Densely sampled.** Loss is recorded after every optimiser step, not once per
  epoch. Five points per run cannot separate a subtly wrong all-reduce from
  noise; a few thousand can.
- **Plotted against samples consumed**, not step count. Four workers stepping
  once per group of batches have no step counter corresponding to this one.

`train_reference()` returns the curve in memory and writes nothing, so the
eventual comparison script can call it directly.

## Layout

```
scripts/
    setup.sh            # idempotent environment bootstrap
    verify_env.py       # hivemind installation checks
src/swarm_mlp/
    model.py            # SimpleMLP, mandated verbatim by the assignment
    data.py             # MNIST pipeline, shared by reference and trainer
    reference.py        # single-process baseline -> LossCurve
    __main__.py         # CLI: run the baseline, plot the curve
docs/
    setup.md
results/                # committed plots
requirements.lock.txt   # exact pins for the verified environment
```
