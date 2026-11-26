| Experiment | What | Variant | IoU test | IoU Bolivia | IoU South Asia | F1 test | Precision test | Recall test | Seeds |
|---|---|---|---|---|---|---|---|---|---|
| E1 | Otsu threshold | shipped | 0.546 | 0.697 | 0.430 | 0.706 | 0.681 | 0.733 | 1 |
| E1 | Otsu threshold | vh | 0.230 | 0.403 | 0.208 | 0.374 | 0.253 | 0.714 | 1 |
| E1 | Otsu threshold | vh_capped | 0.296 | 0.577 | 0.288 | 0.456 | 0.338 | 0.705 | 1 |
| E1 | Otsu threshold | vh_event | 0.429 | 0.643 | 0.368 | 0.600 | 0.489 | 0.777 | 1 |
| E1 | Otsu threshold | vh_event_terrain | 0.444 | 0.644 | 0.386 | 0.615 | 0.511 | 0.773 | 1 |
| E1 | Otsu threshold | vh_terrain | 0.311 | 0.578 | 0.309 | 0.474 | 0.358 | 0.700 | 1 |
| E1 | Otsu threshold | vv | 0.210 | 0.349 | 0.183 | 0.347 | 0.230 | 0.713 | 1 |
| E1 | Otsu threshold | vv_event | 0.319 | 0.607 | 0.272 | 0.483 | 0.362 | 0.724 | 1 |
| E2 | Random forest |  | 0.635 ± 0.000 | 0.622 ± 0.003 | n/a | 0.776 ± 0.000 | 0.825 ± 0.002 | 0.733 ± 0.002 | 3 |
| E2sa | Random forest (South Asia hold-out training) |  | n/a | n/a | 0.546 ± 0.000 | n/a | n/a | n/a | 3 |
| E3 | ETCI U-Net, no retraining |  | 0.607 | 0.672 | 0.512 | 0.756 | 0.729 | 0.785 | 1 |
| E4 | Fine-tuned on hand labels |  | 0.658 ± 0.008 | 0.655 ± 0.028 | n/a | 0.794 ± 0.006 | 0.832 ± 0.011 | 0.759 ± 0.007 | 3 |
| E4_patience10 | E4, patience 10, superseded (final E4 trains all 60 epochs) |  | 0.649 ± 0.014 | 0.566 ± 0.149 | n/a | 0.787 ± 0.010 | 0.843 ± 0.012 | 0.739 ± 0.027 | 3 |
| E5 | E4 + pseudo-labels |  | 0.663 ± 0.009 | 0.607 ± 0.073 | n/a | 0.797 ± 0.007 | 0.864 ± 0.009 | 0.740 ± 0.005 | 3 |
| E6 | E5 + HAND channel |  | 0.667 ± 0.004 | 0.681 ± 0.006 | n/a | 0.801 ± 0.003 | 0.864 ± 0.003 | 0.746 ± 0.004 | 3 |
| E6ens | Ensemble of 5 E6 models (4 seeds + ResNet-34) |  | 0.669 | 0.684 | n/a | 0.802 | 0.866 | 0.747 | 1 |
| E7 | Fusion of E1, E2, E6 |  | 0.667 ± 0.001 | 0.687 ± 0.003 | n/a | 0.800 ± 0.001 | 0.850 ± 0.000 | 0.756 ± 0.001 | 3 |
