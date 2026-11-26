# Third-party code

Everything under `third_party/` is a git submodule and is **read-only**. To change third-party code,
copy the file into `src/`, add a header naming the source repository, commit and licence, then edit
the copy.

| Folder | Repository | Licence | Pinned commit | Used for |
|---|---|---|---|---|
| `third_party/etci2021` | https://github.com/sidgan/ETCI-2021-Competition-on-Flood-Detection | Apache-2.0 | `90d47ca217e240395fcf16287bd85e3cf79afe88` (2022-06-19, v1.0.0 + 36) | Level 1: Paul and Ganju (2021) U-Net / U-Net++ ensemble, input encoding, pseudo-labelling recipe, released weights (release `v1.0.0`). Reproduced in E0 and reused in E3 to E6. |
| `third_party/spacenet8` | https://github.com/SpaceNetChallenge/SpaceNet8 | Apache-2.0 | `30ca30c2530c6a234d040f60a99a7366ff62d756` (2023-06-14) | Level 2: SpaceNet 8 baseline (foundation and Siamese flood networks, data preparation, road and building post-processing) and the APLS scoring code in `03-sianalytics/code/apls/`. Adapted copies live in `src/level2_roads/`. |
| `third_party/fwdet` | https://github.com/csdms-contrib/fwdet | **GPL-2.0** | `b47aec828bee313d373f37899e4d98d75937eb98` (2026-04-28, v2.1.0 + 26) | Water depth: FwDET v2.1 algorithm (Cohen et al.). Ported to NumPy/SciPy in `src/depth/fwdet.py`; the PeeDee test case is used for comparison. |

## Licence obligations

* **Apache-2.0 (etci2021, spacenet8).** Copies in `src/` keep the original copyright and licence notice
  in their header and say what we changed.
* **GPL-2.0 (fwdet).** `src/depth/fwdet.py` is a port of FwDET, so that file is distributed under
  GPL-2.0 (its header says so). If this repository is ever published under another licence, keep
  `src/depth/` under GPL-2.0 or rewrite it from the paper alone.

## Data and weights (not code, not in git)

| Item | Source | Licence / terms |
|---|---|---|
| ETCI 2021 pretrained weights | release `v1.0.0` of etci2021, `pretrained_weights.tar.gz` (an uncompressed tar despite the name; 4 models) | Apache-2.0 (repository licence) |
| Sen1Floods11 v1.1 | `gs://sen1floods11` | see the Sen1Floods11 repository (Cloud to Street) |
| ETCI 2021 data | Hugging Face `blanchon/ETCI-2021-Flood-Detection` | competition terms |
| GLO-30 HAND | `s3://glo-30-hand/v1/2021/` | CC0 |
| Copernicus GLO-30 DEM | `s3://copernicus-dem-30m/` | Copernicus DEM licence (free, attribution) |
| SpaceNet 8 | `s3://spacenet-dataset/spacenet/SN8_floods/` | CC BY-SA 4.0 |
| Maxar Open Data | `https://maxar-opendata.s3.amazonaws.com/events/catalog.json` | CC BY-NC 4.0: academic use only, credit Maxar |
| OpenStreetMap | Overpass via OSMnx | ODbL, credit "© OpenStreetMap contributors" |
| WorldPop | worldpop.org (Nepal, 100 m) | CC BY 4.0 |
| UNOSAT FL20240928NPL | UNOSAT flood product, Koshi and Madhesh (Sentinel-1 water extent 27 Sep and 1 Oct 2024, analysis extents, population-exposure table, A3 map); saved by hand in `$DATA_ROOT/nepal/unosat/` | Reference only, not redistributed. Credit **"Analysis: UNOSAT"** wherever it is shown or compared |
| Sentinel-1 | Google Earth Engine `COPERNICUS/S1_GRD` | Copernicus Sentinel data terms |
| OASIS CAP 1.2 schema | `tests/data/CAP-v1.2.xsd` | OASIS (copied verbatim for validation) |
