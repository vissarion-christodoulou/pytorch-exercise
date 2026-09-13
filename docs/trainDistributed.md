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