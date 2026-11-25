# E3 notes: the ETCI U-Net on Sen1Floods11 without retraining (prompt A6)

Model: the released ETCI round-2 pseudo-label U-Net (MobileNetV2), the best single member
(`configs/E3.yaml`), with D4 test-time augmentation. No Sen1Floods11 training.

## The input problem

ETCI 2021 tiles are **8-bit PNG renderings** of Sentinel-1 VV and VH (0 to 255), produced by
NASA IMPACT with a scaling they did not publish. Sen1Floods11 chips are **float dB** (about -50
to +20). The model was trained on `PNG / 255`, so dB values must be mapped onto that scale first,
or the model sees nonsense.

## The mapping

Per band, a linear map with clipping, then rounding to 8-bit steps as a PNG would store it
(`db_to_unit` and `quantize_like_png` in `src/level1_radar/etci_reproduce.py`):

```
x01 = round(255 * clip((dB - lo) / (hi - lo), 0, 1)) / 255        NaN (no data) -> 0, a black pixel
```

The ETCI third channel is then built exactly as in training: `1 - clip(VH01 / VV01, 0, 1)`.

`lo` and `hi` per band are fitted by `python -m src.level1_radar.etci_reproduce fit-scaling`:

- `--method val-search` (used): coordinate search over the four numbers, steps 5, 2.5 and 1 dB,
  maximising flood IoU on the first 60 chips of `s11_val` (no test chips), without TTA.
  Start (-30, 5) / (-35, 0) dB gave IoU 0.276; the search ended at:

  | Band | lo (dB) | hi (dB) |
  |---|---|---|
  | VV | -17 | 11 |
  | VH | -29 | 5 |

  validation IoU 0.564 (`configs/etci_scaling.yaml`). The upper limits were still moving at the
  last 1 dB step, so a wider search might gain a little more.
- `--method quantile` (alternative, needs the ETCI images): match 7 quantiles of ETCI training PNG
  values to Sen1Floods11 training dB values per band and fit the line.

Reading: the fitted VV range (-17 to 11 dB) is narrow and high. Open water (VV below about -17 dB)
is all mapped to black (0), which is how the model learned water.

## Results (`docs/results/level1/E3.json`, seed 0, CPU)

| Split | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| s11_test | **0.607** | 0.756 | 0.729 | 0.785 |
| s11_bolivia | **0.672** | 0.804 | 0.724 | 0.903 |
| s11_southasia_holdout_test | 0.512 (see caveat) | 0.678 | 0.675 | 0.680 |

Per event on `s11_test`: Nigeria 0.856, Sri Lanka 0.844, Mekong 0.839, India 0.696, Spain 0.664,
Ghana 0.559, USA 0.417, Somalia 0.413, Paraguay 0.374, Pakistan 0.187.

Comparison: without seeing a single Sen1Floods11 label, E3 is close to the random forest trained on
them (E2 test 0.635, Bolivia 0.622) and **better on Bolivia**. It beats our computed Otsu
(0.444). E4 (fine-tuning) should beat both; that is the gate for weeks 3 and 4.

**Caveat on the South Asia number:** `s11_val` is an official random-within-event split, so 20 of
the 60 chips used to fit the four scaling numbers are also in `s11_southasia_holdout_test`. Four
numbers fitted on 60 chips leak little, but report the South Asia IoU of E3 with this note, or
refit on `s11_val` chips outside India, Pakistan and Sri Lanka before quoting it. `s11_test` and
`s11_bolivia` share no chips with the fitting set.
