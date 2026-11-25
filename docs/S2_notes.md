# S2 notes: pre-training from earlier SpaceNet data (prompt B5)

## What the winning SpaceNet 8 solutions did

- **2nd place (`02_number13`, `code/download_data.sh`)** and **4th place (`04-ZABURO`,
  `code/download_external_data.sh`)**: downloaded the SpaceNet 2 building and SpaceNet 3 and 5 road
  **imagery** tarballs for several cities and pre-trained on them before SpaceNet 8.
- **3rd place (`03-sianalytics`)**: road and building models pre-trained on SpaceNet 3
  (`mmsegmentation/configs/best_model_sn3.py`, `road_pretrained.py`).
- **5th place (`05-motokimura`, `docs/prepare_sn5_XD_XD_models.md`)**: no old imagery; it
  downloads XD_XD's published SpaceNet 5 models from
  `s3://spacenet-dataset/spacenet-model-weights/spacenet-5/xd_xd`, converts them to state_dicts and
  initialises its road network with them. This is the cheap route: weights only.

## Our choice (cheapest that fits a 6 GB GPU)

Follow the 5th place idea (weights, not imagery), with the model whose architecture matches ours:
initialise the S0 foundation network (ResNet-34 U-Net, `pytorch_zoo` family, the SN8 baseline)
from the **SpaceNet 5 baseline (CRESI) road model**, an XD_XD-style ResNet-34 U-Net:
`s3://spacenet-dataset/spacenet-model-weights/spacenet-5/baseline/weights/fold0_best.pth`,
**98 MB** (one file; no imagery). Encoder and decoder tensors map one to one; only the output
layer differs (SN5 predicts road speed only, SN8 roads + buildings), so it is left random.

Security: the file is a 2019 whole-model pickle, which PyTorch's safe loader cannot read. A static
scan (`pickletools`, nothing executed) found only PyTorch layers and `pytorch_zoo` classes.
`scripts/prepare_sn5_weights.py convert` refuses the full load unless you pass `--trust-pickle`
(your decision), then saves a plain tensor `state_dict` that training loads with
`weights_only=True`.

Config: `configs/S2.yaml` (= S0 + `model.init_weights.foundation`).

## Commands (overnight on the laptop, resumable)

```bash
conda activate sn8
```
```bash
python scripts/prepare_sn5_weights.py download
```
```bash
python scripts/prepare_sn5_weights.py convert --trust-pickle
```
```bash
python -m src.level2_roads.train --config configs/S2.yaml --task foundation --seed 0 --resume
```
```bash
python -m src.level2_roads.train --config configs/S2.yaml --task flood --seed 0 --resume
```
```bash
python -m src.level2_roads.evaluate --config configs/S2.yaml --split sn8_val
```

If the laptop stops, run the same command again: `--resume` continues from the last epoch's
checkpoint. Compare with S0 in `docs/results_level2.md`.
