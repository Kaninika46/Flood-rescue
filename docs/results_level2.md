# Level 2 results: road-level flood map (SpaceNet 8)

Scores from `python -m src.level2_roads.evaluate --config configs/<S>.yaml --split <split>`
(files in `docs/results/level2/`): APLS (SpaceNet 8 scoring code, `src/level2_roads/apls/`),
pixel IoU for buildings and roads, and length-weighted precision and recall of **flooded road**
segments (`src/level2_roads/scoring.py`). Flooded-road recall is the key safety number: a missed
flooded road can send a vehicle into water. `python scripts/make_report_tables.py` puts them in
`docs/results_tables.md`.

## Pipeline check (done here, no model)

The full chain (vectorize_roads -> wkt_to_G -> infer_speed -> create_submission ->
building_postprocessing -> APLS and length scores) run on ground-truth masks as if they were
predictions, on `sn8_val` tiles:

| Check | Value | Expected |
|---|---|---|
| APLS | 0.896 | close to 1 (raster -> graph -> raster loses a little) |
| Flooded road length precision | 0.982 | close to 1 |
| Flooded road length recall | 0.997 | close to 1 |

So the scoring code works; the numbers below come from trained models.

## Experiments (run on the laptop GPU)

| ID | What | Split | APLS | Building IoU | Flooded road recall (length) |
|---|---|---|---|---|---|
| S0 | baseline, 85/15 split | sn8_val | | | |
| S1 | train Louisiana-East, test Germany | sn8_cross_test | | | |
| S2 | S0 + SpaceNet 5 road weights (`docs/S2_notes.md`) | sn8_val | | | |
| S3 | S2 + cloud mask + ensemble confidence | sn8_val | | | |
| S4 | best model on labelled Nepal 2024 tiles (Kathmandu Valley) | nepal_maxar_labelled | | | |

Note for S4: Maxar's post-event images of the September 2024 Nepal floods cover only the
Kathmandu Valley (GeoEye-1, 6 October 2024); the Koshi-area images are all pre-event. So the
Level 2 Nepal test is in Kathmandu, and the Koshi replay uses Level 1 road status.
