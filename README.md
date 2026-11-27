# Flood Rescue AI

Student ANN project, built by **one developer**. It covers:
- radar and road-level flood maps
- water depth
- safe routes per vehicle
- a priority list of areas
- officer-approved CAP alert drafts
- a control-room dashboard and an offline rescuer app

The test case is the 27 September 2024 Koshi floods in Nepal.

The rules every part of the code follows are in [Project rules](#project-rules). The work is
split into four sections: A radar and depth, B roads and routing, C alerts and priority, D
platform. The step IDs in the Status table (A1, B5, ...) refer to them; they are sections of the
work, not people.

Safety, in one line: routes are advice and always say so, low-confidence roads are shown as
"verify", alerts are drafts until an officer approves them, and they go only to your own test
phones.

## Status

"Done" means the code is written, tested (`pytest`, smoke runs) and, where the data could be
reached, run. GPU training and the Earth Engine download happen on your laptop.

| Prompt | What | State |
|---|---|---|
| P1 to P3 | Skeleton, `src/common`, submodules, both conda envs, `check_env.py`, `find_batch_size.py` | done |
| A1, A2 | Sen1Floods11 download, event splits (incl. South Asia hold-out), HAND and slope per chip | done |
| A3 | E0: ETCI ensemble ported (`docs/E0_notes.md`) | code done; needs the ETCI images (Hugging Face) on the laptop |
| A4 | E1 Otsu, E2 random forest | **run**: see results |
| A5 | Sen1Floods11 dataset + `train.py` (AMP, accumulation, crops, `--resume`, `--smoke`) | done; smoke-tested |
| A6 | E3 (ETCI model, dB -> PNG mapping) / E4 fine-tuning | **E3 run** (`docs/E3_notes.md`); E4 on the laptop GPU |
| A7 | E5 pseudo-labels, E6 HAND channel | code done; GPU |
| A8 | Ensemble prediction (probability + confidence GeoTIFFs), E7 fusion | code done; GPU |
| A9 | FwDET depth (NumPy port, GPL-2.0) | done; PeeDee check needs `git lfs pull` |
| A10 | Nepal radar replay vs UNOSAT | code done for both post dates (27 Sep, 1 Oct), each against UNOSAT FL20240928NPL of the same date; S1 exported, UNOSAT saved; run after the E6 ensemble |
| B1 to B4 | SpaceNet 8 env, data, S0 training, evaluation chain + APLS, S1 | done; scoring checked on ground truth (APLS 0.896); training on GPU |
| B5 | S2: SpaceNet 5 road weights (`docs/S2_notes.md`) | code done; 98 MB download + GPU |
| B6 | S3: cloud mask + ensemble confidence | code done; GPU |
| B7, B8 | Maxar Nepal tiles + labelling, S4 | done up to labelling (`docs/labelling_guide.md`) |
| B9 | Safe-route engine | done |
| C1 to C5 | Extra ensemble members, exposure, priority, CAP alerts with approval, timing | done |
| D1, D2 | FastAPI + PostGIS + RQ backend, `run_event.py` | done; API tests pass on PostGIS |
| D3 | React + MapLibre dashboard | done; checked in a browser |
| D4 | Flutter rescuer app (offline map, route, priorities, report queue) | done; `flutter test` passes; build the APK on the laptop |
| D5 | Overnight training jobs | the commands below, with `--resume` |
| D6 | Demo script, report tables | done (`docs/demo_script.md`, `scripts/make_report_tables.py`) |

## Results so far (CPU-only runs)

Flood-class IoU (TP, FP and FN summed per split). Full tables: `docs/results_tables.md`
(`python scripts/make_report_tables.py`).

| ID | Method | Test | Bolivia | South Asia hold-out |
|---|---|---|---|---|
| E1 | Otsu (Sen1Floods11's shipped labels / our best) | 0.546 / 0.444 | 0.697 / 0.644 | 0.430 / 0.386 |
| E2 | Random forest (3 seeds) | 0.635 ± 0.000 | 0.622 ± 0.003 | 0.546 ± 0.000 (E2sa) |
| E3 | ETCI U-Net, no retraining | 0.607 | 0.672 | 0.512 (see E3 notes) |

Findings: Pakistan is the hardest event for every method (0.19 to 0.30). The ETCI model with
only a fitted dB -> PNG mapping already beats the random forest on Bolivia. For Maxar's Nepal
2024 collection, post-event images cover only the Kathmandu Valley, so Level 2 is tested there.

## Setup

Two conda environments:
- **`flood`** (`environment.yml`): everything except Level 2.
- **`sn8`** (`environment-sn8.yml`): SpaceNet 8 only.

Never mix them. You need an NVIDIA GPU driver. You do **not** need a separate CUDA toolkit:
the CUDA libraries come with PyTorch.

### Linux (Ubuntu)

1. Check the driver: `nvidia-smi` should list your GPU.
2. Install Miniforge (conda with the conda-forge channel and the fast solver):

   ```bash
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   ```
   ```bash
   bash Miniforge3-$(uname)-$(uname -m).sh -b && ~/miniforge3/bin/conda init bash
   ```
   Open a new terminal afterwards.
3. Clone and create the environments (several GB of downloads each):

   ```bash
   git clone --recurse-submodules https://github.com/Kaninika46/Flood-rescue.git flood-rescue && cd flood-rescue
   ```
   ```bash
   conda env create -f environment.yml
   ```
   ```bash
   conda env create -f environment-sn8.yml
   ```
   ```bash
   conda activate flood
   ```
4. Continue with [Data folder](#data-folder), [Experiment tracking](#experiment-tracking),
   [Earth Engine](#earth-engine-sentinel-1-for-nepal) and [Check the laptop](#check-the-laptop).

### Windows (WSL2)

All work happens inside WSL2 (Ubuntu). Do not install GDAL or the project with pip on Windows.

1. Install the normal **Windows** NVIDIA driver (Game Ready or Studio, recent version). It
   provides the GPU to WSL2. **Never install a Linux NVIDIA driver inside WSL.**
2. In PowerShell as administrator:

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
   Restart, open "Ubuntu" from the Start menu and create your Linux user.
3. Inside Ubuntu, `nvidia-smi` must list your GPU. If it does not, update the Windows driver
   and run `wsl --update` in PowerShell.
4. Keep the repository **and the data** in the Linux home folder (`~/...`), never under
   `/mnt/c/...`: Windows drives are 10 to 50 times slower from WSL.
5. Follow steps 2 and 3 of the Linux instructions inside Ubuntu.
6. Optional: WSL2 gets half of the laptop's RAM by default. If training runs out of memory,
   create `C:\Users\<you>\.wslconfig` with:

   ```ini
   [wsl2]
   memory=12GB
   swap=16GB
   ```
   then run `wsl --shutdown` in PowerShell and reopen Ubuntu.

### Updating the environment

After you change `environment.yml` (write down why in the commit message):

```bash
conda env update -f environment.yml --prune
```

### Data folder

All data lives outside the repository, in the folder named by `DATA_ROOT`. Code never contains
absolute paths; configs and split files hold paths relative to `$DATA_ROOT`.

```bash
mkdir -p ~/flood-data && echo 'export DATA_ROOT=$HOME/flood-data' >> ~/.bashrc && source ~/.bashrc
```

Plan for about 120 GB free space:
- Sen1Floods11: 34 GiB for the whole bucket, less without the weak labels.
- HAND and DEM tiles.
- SpaceNet 8: 4.7 GB.
- Maxar tiles, Nepal scenes and checkpoints.

### Experiment tracking

Runs log to Weights & Biases, project `flood-rescue`. Log in once:

```bash
wandb login
```

Without internet (or without a W&B login), runs go to MLflow in `mlruns/` at the repo root
(an SQLite database plus an artifacts folder; MLflow 3 no longer supports the plain-folder
store). View them with:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Force a tracker with `export FLOOD_TRACKER=wandb|mlflow|console`. `--smoke` runs print to the
console only.

### Earth Engine (Sentinel-1 for Nepal)

One-time, with your Google account and your Cloud project (the project id stays on your laptop;
never put it in the repository):

```bash
earthengine authenticate
```
```bash
echo 'export EE_PROJECT=<your-cloud-project-id>' >> ~/.bashrc && source ~/.bashrc
```

### Check the laptop

```bash
python scripts/check_env.py
```

It prints Python, PyTorch, CUDA, the GPU name and memory, every package version, the state of
`DATA_ROOT`, and runs one mixed-precision training step of a small U-Net. It ends with
`OK: this laptop is ready` (exit code 0) or a list of problems (exit code 1). On a machine
without a GPU, add `--cpu-ok`. Then find the largest batch size that fits your GPU (and a
`batch_size x accum_steps` for an effective batch of 16):

```bash
python scripts/find_batch_size.py --effective 16
```

Tests and linter:

```bash
pytest
```
```bash
ruff check . && ruff format --check .
```

`tests/test_backend_api.py` runs only when `TEST_DATABASE_URL` points to an empty PostGIS
database (see `backend/README.md`).

## Run it, part by part

Commands run from the repo root in the `flood` env unless marked **sn8**. Long training runs
go overnight; if the laptop stops, run the same command with `--resume`.

### A. Radar flood map and depth (Level 1)

```bash
python scripts/download_sen1floods11.py
```
```bash
python scripts/download_sen1floods11.py --part weak
```
```bash
python scripts/prepare_hand.py chips
```
```bash
python scripts/compute_s11_norm.py
```

E0 and E3 (ETCI weights 108 MB, ETCI images about 5 GB):

```bash
python -m src.level1_radar.etci_reproduce download-weights
```
```bash
python -m src.level1_radar.etci_reproduce download-data
```
```bash
python scripts/evaluate.py --config configs/E0.yaml
```
```bash
python scripts/evaluate.py --config configs/E3.yaml
```

E1, E2 and the U-Nets. Train every seed (0, 1, 2) of E4, E5 and E6, then evaluate:

```bash
python scripts/evaluate.py --config configs/E1.yaml
```
```bash
python -m src.level1_radar.random_forest --config configs/E2.yaml
```
```bash
python -m src.level1_radar.train --config configs/E4.yaml --seed 0
```
```bash
python -m src.level1_radar.pseudo_label --config configs/E5.yaml
```
```bash
python -m src.level1_radar.train --config configs/E6.yaml --seed 0
```
```bash
python scripts/evaluate.py --config configs/E4.yaml
```

Ensemble (`configs/E6ens.yaml`: E6 seeds 0 to 3 plus one ResNet-34 member, `configs/E6r34.yaml`)
and fusion:

```bash
python -m src.level1_radar.fusion --config configs/E7.yaml --seed 0
```
```bash
python scripts/evaluate.py --config configs/E7.yaml
```

Depth and the Nepal replay (Earth Engine export first, then the UNOSAT product FL20240928NPL
unzipped by hand into `$DATA_ROOT/nepal/unosat/`; credit "Analysis: UNOSAT"):

```bash
python scripts/compare_fwdet_peedee.py
```
```bash
python scripts/nepal_s1_download.py export --dry-run
```
```bash
python scripts/nepal_s1_download.py export
```

`--dry-run` lists each export (post date, its pre-flood scene on the same orbit, JRC) with its
estimated size. `--date 2024-09-27` exports one post date and its pre scene; `--no-jrc` skips
JRC. Each post date's pre scene is `S1_<post date>_pre`. The area is the Koshi and Madhesh
outline clipped to `aoi.clip_bbox` (Terai plains): about 11 GB uncompressed for both dates.
```bash
python scripts/nepal_s1_download.py mosaic
```
```bash
python scripts/nepal_level1.py run
```

`run` handles every post date in `dates.post` (`--date 2024-10-01` for one). Each scene is
compared with the UNOSAT flood layer whose file name has the same date, inside that date's UNOSAT
analysis extent, with JRC permanent water left out of both maps (the product has no
permanent-water layer). Results are given for the whole extent inside our scene and for its Nepal
part (the extents cover India too): `docs/results/nepal_level1_<YYYYMMDD>.json` and
`docs/nepal_level1.md`. The 1 October scene reuses the 27 September HAND and slope (same grid).
Three model maps are scored: (a) all water minus JRC >= 50 %; (b) new water, the same ensemble
run on the same-orbit pre scene, flood = water in post and not in pre; (c) all water minus
JRC >= 10 % (sensitivity only). (a) feeds exposure, priority, alerts and `run_event.py` (step
`mask`) on a safety principle: under-counting flooded people is worse than over-counting. It also
counts the normal Koshi channel, so those numbers are an upper estimate. (b) recalled only 12 to
20 % of UNOSAT's flood because the mid-September pre scenes were already wet.

### B. Roads (Level 2) and safe routes

**sn8** env:

```bash
python scripts/download_sn8.py download
```
```bash
python scripts/download_sn8.py prepare
```
```bash
python -m src.level2_roads.train --config configs/S0.yaml --task foundation --seed 0
```
```bash
python -m src.level2_roads.train --config configs/S0.yaml --task flood --seed 0
```
```bash
python -m src.level2_roads.evaluate --config configs/S0.yaml --split sn8_val
```

S1 uses `configs/S1.yaml` with `--split sn8_cross_test`; S2 is in `docs/S2_notes.md`; S3 is
`configs/S3.yaml`. Nepal (Maxar, Kathmandu Valley), in the `flood` env unless marked:

```bash
python scripts/maxar_nepal.py pairs
```
```bash
python scripts/maxar_nepal.py tiles --max-pairs 60
```
```bash
python scripts/maxar_nepal.py osm
```

Label the tiles (`docs/labelling_guide.md`), then **sn8**: `python scripts/nepal_level2.py prepare`
and `python -m src.level2_roads.evaluate --config configs/S4.yaml --split nepal_maxar_labelled`.

Routes (OSM road graph for the Koshi plains, then statuses, routes and a figure):

```bash
python scripts/route_demo.py graph
```
```bash
python scripts/route_demo.py run
```

### C. Priority and alerts

```bash
python scripts/priority_nepal.py download
```
```bash
python scripts/priority_nepal.py run
```
```bash
cp config/test_recipients.example.yaml config/test_recipients.yaml
```

Put **your own** phone number in `config/test_recipients.yaml` (git-ignored), then:

```bash
python scripts/alert_demo.py --officer "Your Name"
```

### D. Platform

One command for a whole event, stored in the database:

```bash
python scripts/run_event.py --scene $DATA_ROOT/nepal/s1/S1_20240927.tif
```
```bash
python scripts/time_pipeline.py --scene $DATA_ROOT/nepal/s1/S1_20240927.tif
```

Backend: `backend/README.md`. Dashboard: `dashboard/README.md`. Rescuer app: `app/README.md`.
Offline map for the app:

```bash
python scripts/make_mbtiles.py
```

No real data yet? `scripts/demo_seed.py` fills a **new, empty** `DATA_ROOT` and the database
with a clearly labelled synthetic event, so the dashboard and the app can be tried.

Report tables and the demo: `python scripts/make_report_tables.py`, `docs/demo_script.md`.

## Project rules

### Safety: alerts and routes

* Alerts are drafts until an officer approves them in the dashboard. Every alert record needs
  `approved_by` and `approved_at` before it can be sent.
* `src/alerts/gateway.py` sends only to the numbers listed in `config/test_recipients.yaml`
  (your own test phones; the file is git-ignored) and refuses every other recipient. The
  default gateway is a console mock.
* Nothing in this project broadcasts to the public or impersonates an agency.
* Routes are advice, and every route response says so. Low-confidence road segments are never
  hidden: they are marked "verify".

### Never commit

* Data, checkpoints, `.env` files, API keys, credentials or phone numbers. `.gitignore` covers
  them; keep it up to date.
* All data lives in `$DATA_ROOT`, outside the repository. Code and configs never contain
  absolute paths: data paths are relative to `$DATA_ROOT`.

### Metrics

* Flood-class IoU, F1, precision and recall: positive class 1, label -1 ignored. TP, FP and FN
  are summed over the whole split (never the mean of per-chip IoU), and every split is also
  reported per flood event (`src/common/metrics.py`, unit-tested).
* Level 2: APLS and building IoU with the SpaceNet 8 scoring code, plus precision and recall
  for flooded road segments. Flooded-road recall is the key safety number.
* System: minutes from image to alert draft, and the length of each route that runs through
  water deeper than its vehicle's limit (target 0).

### Reproducibility

* Every run calls `seed_everything(seed)` from `src/common/seed.py` first. Standard seeds: 0,
  1, 2.
* Splits are read only from the committed `splits/*.csv` files and are split by flood event,
  never by random chip. No training script creates a random split.
* One config file per experiment ID; run name `<experiment>-s<seed>` (for example `E4-s1`).
* Every run logs its config, git commit, metrics, GPU name and runtime to Weights & Biases
  (project `flood-rescue`), or offline to MLflow in `mlruns/`.
* Final test numbers come only from `scripts/evaluate.py`, run with saved checkpoints.

### Hardware (one laptop with a 6 to 8 GB GPU)

* Every training script runs on a 6 GB GPU: mixed precision on by default, `batch_size` and
  `accum_steps` set in the config, random crops for large tiles, and a `--smoke` flag that runs
  2 batches end to end.
* The effective batch size (`batch_size x accum_steps`) is fixed per experiment, so runs stay
  comparable when the batch size has to change.
* A checkpoint is saved every epoch and `--resume` continues a stopped run. Every run logs the
  GPU name and peak GPU memory.

## Workflow (one developer)

* Do one step (A1, A2, ...) at a time, check that it works, then commit:
  `git add -A && git commit -m "<step id>: <what>"`.
* Before committing: `ruff check .`, `ruff format .`, `pytest`, and the relevant `--smoke` run.
* Never commit data, checkpoints, `.env`, keys or phone numbers (`.gitignore` covers them).

### First push to GitHub

From the unpacked project folder (it adds the three submodules at the pinned commits and makes
the first commit under your git name):

```bash
bash scripts/bootstrap_git.sh
```
```bash
git remote add origin https://github.com/Kaninika46/Flood-rescue.git
```
```bash
git push -u origin main
```

## Shared helpers (`src/common`)

| Module | Use it for |
|---|---|
| `seed.py` | `seed_everything(seed)` at the start of every run (seeds 0, 1, 2); `seed_worker` and `torch_generator` for DataLoaders. |
| `config.py` | `load_config("configs/E4.yaml", overrides)`, `data_path("sen1floods11", ...)`, `run_name("E4", 1)` (gives `E4-s1`), `effective_batch_size(cfg)`. |
| `logging.py` | `with start_run(cfg, seed=seed) as run:` logs config, git commit, GPU, effective batch size, metrics, runtime and peak GPU memory. `run.resume_info()` goes in checkpoints for `--resume`. |
| `gpu.py` | `gpu_info()`, `peak_memory_gb()`, `describe()`. |
| `metrics.py` | `SplitScorer`: flood IoU, F1, precision and recall summed over a whole split, plus per event; ignores label -1. `binarize(prob, 0.5)` first. |
| `raster.py` | `read_raster` / `write_raster` / `write_like` keep CRS, transform and nodata; `describe_raster(path)` before writing any new loader. |
| `geo.py` | `pixel_area_km2`, `rasterize_geometries`, `read_vectors`. |

### Configs

* `configs/` holds one YAML per experiment ID (`E0.yaml` ... `E7.yaml`, `S0.yaml` ... `S4.yaml`,
  plus `depth`, `nepal`, `routing`, `priority`). `config/` (singular) holds runtime settings:
  `vehicles.yaml` and your git-ignored `test_recipients.yaml`.
* `experiment` defaults to the file name; run names are `<experiment>-s<seed>`.
* `base: E4.yaml` starts a config from another one; nested sections are merged, the child wins.
* Training settings go under `train:`. `train.batch_size x train.accum_steps` is the effective
  batch size and stays fixed for an experiment; on a smaller GPU, halve `batch_size` and double
  `accum_steps`.
* Data paths in configs are relative to `$DATA_ROOT`.
* Command-line overrides: `--set key.sub=value` (values parsed as YAML), for example
  `--set train.batch_size=4 --set train.accum_steps=4`. A misspelt key is an error; use
  `+key=value` to add a new key on purpose.

### Metric conventions

* Positive class 1, ignore label -1. TP, FP and FN are summed over the whole split, never
  averaged per chip.
* Predictions must be binary (`bool` or 0/1). A prediction of -1 counts as "not water", so all
  methods are scored on the same pixels. Masks saved as 0/255 are refused.
* A metric with a zero denominator is `NaN` (for example IoU on a split with no water at all);
  report it as "n/a".

## Security

Details: `backend/README.md` (tokens, rate limits, secrets) and `app/README.md` (phone demo,
firewall).

* **Backend.** Listens on 127.0.0.1 unless started with `--phone-demo`, which refuses to start
  without `OFFICER_TOKEN` and `TEAM_TOKEN`. Every action that changes something needs a bearer
  token (401 otherwise); approvals record the token's role. `POST /routes` and
  `POST /field_reports` are rate-limited per IP (429 with `Retry-After`). Text lengths and
  lon/lat ranges are checked, and `GET /layers` returns no server file paths. Database and Redis
  passwords and the tokens live only in the git-ignored `backend/.env`; Redis needs its password
  and every Docker port is bound to 127.0.0.1.
* **Rescuer app.** The team token is set in Settings; reports refused for a missing or wrong
  token stay queued. It talks plain HTTP on the local Wi-Fi, so open port 8000 in the Windows
  firewall to the phone's IP only, and remove the rule after the demo. The release APK is signed
  with the debug key: for your own test phones only, never publish it.
* **Dashboard.** Keeps the officer token in memory for the tab only; `npm audit` is clean.
* **Checkpoints** are loaded with `weights_only=True`.
* **Python packages.** `urllib3` (2.8.0) and `setuptools` (84.2.1) were upgraded in the `flood`
  env with pip, not conda, because the conda solve ran out of memory in WSL. `environment.yml`
  sets the same floors (`urllib3>=2.8`, `setuptools>=84`) for new envs. `awscli` was removed from
  `environment.yml` (no project code uses it, and it requires `urllib3<=2.6.3`); in an existing
  env it stays installed until `conda env update --prune`, and `pip check` reports that conflict.
* **Known, low exposure: GDAL 3.12.** It stays at 3.12 from conda-forge. GDAL reads only rasters
  we downloaded into `$DATA_ROOT`; the API registers scenes by path under `$DATA_ROOT` and
  accepts no raster uploads.

## Troubleshooting

| Problem | Fix |
|---|---|
| `nothing provides __cuda` when creating the env | The solver could not see the driver. Check `nvidia-smi`, or run `CONDA_OVERRIDE_CUDA=12.9 conda env create -f environment.yml`. |
| `check_env.py` says no CUDA GPU | Linux: driver not loaded (`nvidia-smi`). WSL2: update the Windows driver, `wsl --update`, reopen Ubuntu. |
| `no kernel image is available for execution on the device` | The GPU is older than the PyTorch build supports. Compare the GPU's `arch` line in the `check_env.py` output with the architectures it says PyTorch was built for. |
| `No module named 'simsimd'` when importing albumentations | `simsimd` 7.x is an empty renamed package. `environment.yml` pins `simsimd=6`; run `conda env update -f environment.yml --prune`. |
| `DATA_ROOT is not set` | See [Data folder](#data-folder), then open a new terminal. |
| `PROJ: proj_create_from_database` in the sn8 env | The env is not activated (`conda activate sn8`), so PROJ cannot find its database. |
| `prepare_sn5_weights.py convert` stops | It needs `--trust-pickle` (see `docs/S2_notes.md`). |
| Everything is very slow on WSL | Repo or data under `/mnt/c`. Move both into `~`. |
| Alert sending says "no registered test phones" | Create `config/test_recipients.yaml` from the example file with your own number. |

## Licences and data credits

Code in `third_party/` keeps its own licences (see `THIRD_PARTY.md`); `src/depth/fwdet.py` is a
port of FwDET and stays GPL-2.0. Maxar Open Data imagery is CC BY-NC 4.0: academic use only,
credit Maxar. HAND (GLO-30 HAND) is CC0. OpenStreetMap data © OpenStreetMap contributors (ODbL).
WorldPop data CC BY 4.0.
