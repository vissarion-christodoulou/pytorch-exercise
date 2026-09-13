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