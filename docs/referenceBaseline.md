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