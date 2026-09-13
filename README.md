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
  activations and gradients between workers. It orchestrates load balancing
  within a stage and triggers all-reduce when the target bach size is reached

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
  produce bit-identical curves.
- **Densely sampled.** Loss is recorded after every optimiser step, not once per
  epoch.
- **Plotted against samples consumed**, not step count. Four workers stepping
  once per group of batches have no step counter corresponding to this one.

`train_reference()` returns the curve in memory and writes nothing, so the
eventual comparison script can call it directly.

## TRAINER
For every stage in the model, it holds a pool of corresponding workers.
For simplicity, harming generalization, assuming same number of workers (replicas) per stage.
Again for simplicity, harming speed, the all-reduce is triggered once all stages are done propagating.


## WORKER

## Layout

```
scripts/
    setup.sh            # idempotent environment bootstrap
    verify_env.py       # hivemind installation checks
src/swarm_mlp/
    __main__.py                 # CLI: run the baseline, plot the curve
    baseline_reference/
        reference.py            # single-process baseline -> LossCurve
    distributed_training/
        trainer.py              # the trainer class
        worker.py               # the worker class
        control.py              # built on hivermind library to control the communication between a worker and a trainer
    utils/
        constants.py            # values shared by the baseline and the distributed system
        model.py                # SimpleMLP, mandated verbatim by the assignment and functions for building the distributed model
        data.py                 # MNIST pipeline, shared by reference and trainer
        observability.py        # logging-specific settings
        curves.py               # helper for building sample to training error curves
        plotting.py             # helper for plotting curves
        compare.py              # used to compare distributed curves to the reference curve

docs/
    setup.md
results/                # committed plots
requirements.lock.txt   # exact pins for the verified environment
```
