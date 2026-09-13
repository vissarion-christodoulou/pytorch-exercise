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