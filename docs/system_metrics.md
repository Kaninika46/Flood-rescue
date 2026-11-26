# System metrics

Scene `S1_20240927.tif`, GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8.0 GB, compute 8.9.

| Step | Minutes |
|---|---|
| level1 | 2.58 |
| mask | 0.12 |
| depth | 0.30 |
| level2 | 0.00 |
| roads | 1.24 |
| priority | 0.21 |
| alert | 0.00 |
| **image to alert draft** | **4.46** |

HAND and other terrain rasters are prepared before an event (they do not change between floods) and were passed with `--hand`; their preparation is not included in the time above.

Routes (minutes; None = no passable route): car: 55.42, truck: 69.27, foot: 366.16

Route length through water deeper than the vehicle limit (target 0): car: 0 m, truck: 0 m, foot: 0 m
