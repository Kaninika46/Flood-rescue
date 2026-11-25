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

## To run on the laptop GPU

E0 (needs the ETCI images), E4 to E6 (3 seeds each), the E6 ensemble (5 members) and E7 fusion:

```bash
python scripts/evaluate.py --config configs/E4.yaml
```

(the training commands per experiment are in README.md). Fill the rest of the table with
`python scripts/make_report_tables.py`.
