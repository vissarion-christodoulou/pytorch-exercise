## Tradeoffs/Extentions
I have opted for simplicity in a number of cases to focus on producing a Proof of Concept rather than having to worry about how the implementation generalizes or handles adverse situations:


<ol>
  <li>Assuming static 2x2 model throughout (harming generalization)</li>
  <li>The all-reduce is triggered once all stages are done propagating (harming speed)</li>
  <li>A worker crashing or losing synchronization with peers causes the trainer to die and lives other workers "orphaned" (harming reliability)</li>
  <li>Support for single trainer only</li>
  <li><code>src/swarm_mlp/distributed_training/constants.py</code> holds some constants such as polling timeouts, not configurable by the trainers/workers to avoid passing around parameters. Selected values should be fine for all cases, but feel free to change on local run if they cause an issue.</li>
  <li>Trainer is informed about all workers. No DHT discovery</li>
  <li>RPC failures/hanging are not detected. This means some of the experiments described might have sporadic failures.</li>
  <li>The experimenting on ablation is not reproducible through a single python command. Data collection was done largely by hand/with the help of ai tools, but should be a fairly straightforward implementation.</li>
  <li>All the results are saved under results/, which is not gitignored. Files are produced when training by both trainers and workers so take care when commiting after running.</li>
</ol>