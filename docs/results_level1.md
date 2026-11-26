# Level 1 results: radar flood map

Flood-class IoU with TP, FP and FN summed over the whole split (`src/common/metrics.py`).
Numbers come only from `scripts/evaluate.py` (files in `docs/results/level1/`). The full table,
mean ± standard deviation over seeds, is rebuilt by `python scripts/make_report_tables.py`
(`docs/results_tables.md`).

Splits (committed CSVs, split by event): `s11_test` (90 chips, official), `s11_bolivia` (15, the
official held-out country), `s11_southasia_holdout_test` (138 chips of India, Pakistan and Sri
Lanka, for models trained without them: the closest test to Nepal).

## Done (CPU)

| ID | Method | IoU test | IoU Bolivia | IoU South Asia | Seeds |
|---|---|---|---|---|---|
| E1 | Otsu: Sen1Floods11's shipped Otsu labels | 0.546 | 0.697 | 0.430 | 1 |
| E1 | Otsu: our best (VH, per event, HAND/slope exclusion) | 0.444 | 0.644 | 0.386 | 1 |
| E2 | Random forest on engineered features (threshold tuned on s11_val) | 0.635 ± 0.000 | 0.622 ± 0.003 | n/a | 3 |
| E2sa | Random forest trained without South Asia | n/a | n/a | 0.546 ± 0.000 | 3 |
| E3 | ETCI round-2 U-Net, no retraining, fitted dB -> PNG mapping | 0.607 | 0.672 | 0.512* | 1 |

\* 20 of the 60 chips used to fit E3's four scaling numbers are in the South Asia test split; see
`docs/E3_notes.md`.

Per event on `s11_test` (E2): Sri Lanka 0.865, Nigeria 0.819, Mekong 0.809, Spain 0.723,
India 0.687, Paraguay 0.536, Ghana 0.507, USA 0.504, Somalia 0.319, **Pakistan 0.194**.
South Asia hold-out (E2sa): Sri Lanka 0.698, India 0.533, **Pakistan 0.300**.

Findings so far:

- Sen1Floods11's own Otsu labels beat every per-chip/per-event Otsu we computed: the shipped
  labels used a cleaner processing chain, so E1 reports both.
- The random forest gains most where Otsu fails on mixed land cover (Paraguay 0.34 -> 0.54).
- Pakistan is the hardest event for every method (dry, bright desert and thin water bodies). It
  is the event most like the Koshi plains in the hold-out split, so watch it for E4 to E6.

- E3, the ETCI model with no Sen1Floods11 training, is already close to the random forest and
  better on Bolivia (0.672 vs 0.622): the dB -> PNG mapping (`docs/E3_notes.md`) matters most.

## Training protocol change for the U-Nets (E4 onwards)

The first E4 run used early stopping with `train.patience: 10`. On `s11_val` the IoU of every
seed rose to about 0.62 within 2 to 5 epochs and then went up and down between 0.60 and 0.65
from epoch to epoch, without a clear trend. With that much noise, 10 epochs without a new best
happen by chance: seeds 1 and 2 stopped after 14 epochs, both keeping the checkpoint of their
4th epoch, while seed 0 ran 33 epochs and kept its 23rd.

From now on every U-Net trains all 60 epochs (`train.patience: 60` in `configs/E4.yaml`, which
E5, E6 and the ensemble members inherit through `base:`) and keeps the checkpoint with the best
`s11_val` IoU. Nothing else changed. The decision was made from validation curves only; no test
or Bolivia number was used to choose it, and the first run stays on record as
`docs/results/level1/E4_patience10.json` (checkpoints in `$DATA_ROOT/checkpoints/E4_patience10-s*`).

Outcome, measured after the change (IoU, mean ± std over seeds 0, 1, 2; batch 16 x 1):

| E4 run | Best val epoch (s0, s1, s2) | IoU test | IoU Bolivia |
|---|---|---|---|
| patience 10 | 22, 3, 3 | 0.649 ± 0.014 | 0.566 ± 0.149 |
| all 60 epochs | 19, 12, 7 | 0.658 ± 0.008 | 0.655 ± 0.028 |

The seeds now agree far better on Bolivia; E4 still stays below E1 and E3 there.

### Batch size

Every E4, E5 and E6 training run used batch 16 × 1 (`batch_size` 16, `accum_steps` 1), now the
value in `configs/E4.yaml`; the runs before this change passed it as a `--set` override.
One exception: E6-s3 stopped after epoch 51 and was resumed without the override, so epochs
52 to 59 (the last 8 of 60) ran at 8 × 2 (same effective batch 16). Its `best.pt` is from
before the stop (epoch 44, val IoU 0.6701) and is the checkpoint used for evaluation and the
ensemble; no epoch after the resume beat it. The resume overwrote
`$DATA_ROOT/checkpoints/E6-s3/config.yaml`, which now shows 8 × 2; the original settings are
in the MLflow params of run E6-s3.

### E2 retrained on this laptop

The original E2 forests (`forest.joblib`) were not saved, and E7 needs them as an input. E2 was
retrained on this laptop on 26 November 2026 with the same config (`configs/E2.yaml`) and seeds
(0, 1, 2), then re-scored with `scripts/evaluate.py`. Every overall IoU (3 seeds × s11_test and
s11_bolivia) matches the earlier numbers to four decimals (largest difference 0.0000; the
acceptance limit was 0.005), so `docs/results/level1/E2.json` was replaced by the new run; only
its `git_commit` field changed (`unknown` before). E7 uses these retrained forests.

## To run on the laptop GPU

E0 (needs the ETCI images), E4 to E6 (3 seeds each), the E6 ensemble (5 members) and E7 fusion:

```bash
python scripts/evaluate.py --config configs/E4.yaml
```

(the training commands per experiment are in README.md). Fill the rest of the table with
`python scripts/make_report_tables.py`.
