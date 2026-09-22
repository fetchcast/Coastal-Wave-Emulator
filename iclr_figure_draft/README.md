# Architecture-search cost and validation accuracy figure

English | [Korean](README_KO.md)

`plot_cost_accuracy.py` plots 29 stage-A configurations and three pilot anchors, all seed 42. B/C runs are excluded. Stars mark the lowest validation Hs MAE in each family among these A/pilot runs, not the final campaign selections.

## Generate

Supply the original stage-A source CSV outside Git:

```bash
python3 plot_cost_accuracy.py /absolute/path/results_A_source.csv --output /absolute/path/figure_output
```

The PDF, SVG, and PNG outputs represent the same figure. Generated figures and the result CSV are not included in this repository snapshot.

## Draft caption

Validation accuracy across the architecture search. Each point represents one configuration trained with seed 42 for 76,950 successful optimizer updates. The panels relate the best validation Hs MAE to parameter count (a) and recorded training wall time (b). Stars mark the lowest validation MAE for each model family among the 29 stage-A configurations and three pilot anchors. Stage-B learning-rate trials and stage-C seed repetitions are not included. Training wall times reflect the recorded runs and are not controlled measurements of inference latency.

## Interpretation limits

Equal updates do not imply equal computation. Single-seed search does not establish statistical superiority. Parameter count and recorded wall time are distinct cost measures. This figure does not establish cyclone/2021 accuracy or inference acceleration. Because B results are absent, it does not include FFNO's later selected max_lr 5e-5 result.
