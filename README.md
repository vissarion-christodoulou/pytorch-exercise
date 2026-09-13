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
  within a stage and triggers all-reduce when the target bach size is reached. For simplicity, harming generalization, assuming same number of workers (replicas) per stage. Again for simplicity, harming speed, the all-reduce is triggered once all stages are done propagating.

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
python -m swarm_mlp.baseline_reference
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

## Train distributed
To train the distributed model, start up 4 workers, 2 for each stage:
```bash
python src/swarm_mlp/distributed_training/worker.py --stage stage0 --index 0
python src/swarm_mlp/distributed_training/worker.py --stage stage0 --index 1
python src/swarm_mlp/distributed_training/worker.py --stage stage1 --index 0
python src/swarm_mlp/distributed_training/worker.py --stage stage1 --index 1
```

On startup, workers print their peer id and tcp port, allowing us to start up a
trainer with known peers (example peers given below):
```bash
python src/swarm_mlp/distributed_training/trainer.py --initial-peers /ip4/127.0.0.1/tcp/41527/p2p/12D3KooWRbeoKaUfFurHKZJ8CBdeW5Q1GSeuT5nFimYJcwJ3i7mR /ip4/127.0.0.1/tcp/42475/p2p/12D3KooWCjSLeZooFcPRFXBByEZaDKgYG496KJV9Hou3i6ZwTMJE /ip4/127.0.0.1/tcp/37423/p2p/12D3KooWJtHozKtnzA2vT6X3rJdWbyGQE4xH1pciv5hWhSRQAfX4 /ip4/127.0.0.1/tcp/32975/p2p/12D3KooWCCf266PgMvrL99tfhrrJu7ZAiEUSuo39eMBa8B7WhQZG
```

Run without arguments to see all optional arguments that can be passed to the trainer and workers.

## Results
### Reference curve standalone
reference_loss.png is the loss curve is the cross entropy loss and the accuracy of the reference 
model - SimpleMLP non-distributed, 64 samples per batch. It can be acquired by running from the venv:
```bash
python -m swarm_mlp.baseline_reference
```
### Compare distributed to reference
First, I trained the distributed model with default parameters (notably batch_size of 64, batches per group of 10, 3 epochs on the 60k sample dataset). The trainer, upon completion, saves the results under results/distributed_pipeline_{ts}.json where ts is the timestamp of the run. I then run:
```bash
python src/swarm_mlp/utils/compare.py --distributed results/distributed_pipeline_{ts}.json  
```
The compare runs the reference model with the exact same parameters, saved as metadata in the json file of the distributed run. It uses a batch size of batch_size * batches_per_group, since that is when a step occurs in the distributed model. The compare has two outputs:
```
results/
    compare_pipeline_{ts}.png
    comparison_{ts}.txt
```
ts here is the timestamp of the corresponding distributed run curve. The txt file shows that the mean difference in loss is 5.3e-0.5 and the max difference is 4.071e-04, hence accepting the models as equivalent at the 0.001 precision level. In the png, the two curves overlay one another. By using the same seeds, the models have achieved exactly the same behaviour!

## Layout

```
scripts/
    setup.sh            # idempotent environment bootstrap
    verify_env.py       # hivemind installation checks
src/swarm_mlp/
    baseline_reference/
        __main__.py             # CLI: run the baseline, plot the curve
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
