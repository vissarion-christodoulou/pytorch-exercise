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
All the results are saved under results/, which is not gitignored. Files are produced when training by both trainers and workers so take care when commiting after running.
### Reference curve standalone
reference_loss.png is the is the cross entropy loss curve and the accuracy curve of the reference 
model - SimpleMLP non-distributed, 64 samples per batch. It can be acquired by running from the venv:
```bash
python -m swarm_mlp.baseline_reference
```
### Distributed training proof of correctness and comparison of loss curve between distributed and reference
First, I trained the distributed model with default parameters (notably batch_size of 64, batches per group of 10, 3 epochs on the 60k sample dataset). The trainer, upon completion, saves the results under `results/distributed_pipeline_{ts}.json` where ts is the timestamp of the run. Each worker, when killed, dumps its final parameters (the weights of the Neural Network) in a `results/worker_{stage}_{index}_{ts}.json`. Note the timestamp of all these files could be different.

If the workers are killed within two minutes of each other, we can then run:
```bash
python src/swarm_mlp/distributed_training  
```
This compares the weights of worker 0.0 and worker 0.1 (the two replicas in the first stage) and the weights of worker 1.0 and 1.1 and checks if they are equal (within floating-precision distance). The result is saved inside distributed_{ts}.txt and it shows a maximum weight difference of 1e-7 across all weights and biases in both stages. The workers within a stage are in sync!  

I then run:
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

### Ablation: activation and gradient transmission precision
Tried out five different models with different levels of compression in either activation or gradient transmission to experiment. The initial configurations with epochs=3, batch_size=64, batches_per_group=10 produced counterintuitive results in terms of time taken for training, because in that case time is dominated by RPC call overheads, rather than data transferred by the RPC.
Hence, switched to these configurations: epochs=8, batch_size=512, batches_per_group=10.

For each configuration, I span up workers with additional arguments `--activation-precision`, `--gradient-precision` and a trainer with `--output results/ablation_{model_name}.json`. The values for the arguments can be seen in the table below a few lines below. 

The results are saved under `results/ablation_{model_name}.json`
To visualize losses:
```bash
python scripts/ablation_report.py
```
This produces the comparative loss curve in `results/ablation_precision.png`

| config | activations | gradients | batches/s | final loss | final acc | replica max \|dw\| | replicas |
|---|---|---|---|---|---|---|---|
| baseline | fp32 | fp32 | 8.2 | 0.2392 | 0.9313 | 2.2e-07 | PASS |
| grad_fp16 | fp32 | fp16 | 9.5 | 0.2393 | 0.9312 | 7.6e-03 | FAIL |
| grad_int8 | fp32 | int8 | 8.8 | 0.2381 | 0.9318 | 3.8e-02 | FAIL |
| act_fp16 | fp16 | fp32 | 10.4 | 0.2387 | 0.9315 | 3.4e-08 | PASS |
| act_int8 | int8 | fp32 | 9.9 | 0.2312 | 0.9335 | 1.5e-07 | PASS |


CONCLUSIONS
<ol>
  <li>Throughput increases with compression, especially with compression on activations which are shared for every batch, rather than per group of batches for gradients (see batches/s from table).</li>
  <li>Models converge on a very similar model - they track loss very well regardless of compression (see png first two figures).</li>
  <li>Activation compression makes the loss deviate more to the baseline model compared to gradient compression, but perhaps surprisingly, higher level of compression on gradients has a bigger influence than lower level of conversion on activations (see png third figure)</li>
  <li>Gradient sharing compression threatens the consistency of weights across layers. The replicas column from the table shows failures, the delta between weights in the same stage is order of magnitude ~0.01 (fairly large).</li>
</ol>

## Tradeoffs
I have opted for simplicity in a number of cases to focus on producing a Proof of Concept rather than having to worry about how the implementation generalizes or handles adverse situations:


<ol>
  <li>Assuming static 2x2 model throughout (harming generalization)
</li>
  <li>The all-reduce is triggered once all stages are done propagating (harming speed)</li>
  <li>A worker crashing or losing synchronization with peers causes the trainer to do and lives other workers "orphaned" (harming reliability)</li>
  <li><code>src/swarm_mlp/distributed_training/constants.py</code> holds some constants such as polling timeouts, not configurable by the trainers/workers to avoid passing around parameters. Selected values should be fine for all cases, but feel free to change on local run if they cause an issue.</li>
  <li>Trainer is informed about all workers. No DHT discovery</li>
  <li>RPC failures/hanging are not detected. This means some of the experiments described might have sporadic failures.</li>
</ol>


## Layout

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
