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
  within a stage and triggers all-reduce when the target batch size is reached.




## Docs
The following docs cover the scope of the project:
```
docs/
  setup.md              # set up running environment. Prerequisite for other runs
  referenceBaseline.md  # run SimpleMLP and plot its loss curve
  trainDistributed.md   # instructions on running a trainer and workers
  distributedProof.md   # distributed training proof of correctness and comparison of loss curve between distributed and reference
  ablation.md           # experiment with gradient and activation compression
  tradeoffs.md          # tradeoffs of the project, generally aiming at simplicity. Gotchas and possible extensions.

```


## Rest of Layout

```
scripts/
    setup.sh            # idempotent environment bootstrap
    verify_env.py       # hivemind installation checks
    ablation_report.py  # builds the wire-precision figure from the five curves
src/swarm_mlp/
    baseline_reference/
        __main__.py             # CLI: run the baseline, plot the curve
        reference.py            # single-process baseline -> LossCurve
    distributed_training/
        __main__.py             # Compares saved weights that are saved under results from latest dying workers of the same stage
        constants.py            # constants not needed by the reference model
        precision.py            # wire precisions for both channels, and a NumPy 2 fix for 8-bit
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
