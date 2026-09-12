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

## Layout

```
scripts/
    setup.sh            # idempotent environment bootstrap
    verify_env.py       # hivemind installation checks
src/swarm_mlp/
docs/
    setup.md
requirements.lock.txt   # exact pins for the verified environment
```
