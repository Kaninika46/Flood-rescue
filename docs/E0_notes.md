# E0 notes: the released ETCI 2021 ensemble in our environment (prompt A3)

Source: `third_party/etci2021` (Paul and Ganju 2021, Apache-2.0, commit 90d47ca),
`src/etci_dataset.py` and `notebooks/Ensemble_Inference.ipynb`. Our port:
`src/level1_radar/etci_reproduce.py`, config `configs/E0.yaml`.

## Changes made to run it on Python 3.11 / PyTorch 2.x

| Original (Python 3.8, PyTorch 1.9) | Ours | Why |
|---|---|---|
| `ttach` test-time augmentation wrapper | the 8 D4 transforms (4 rotations x flip) done in `predict.py` (`_d4`, `_d4_inverse`), logits averaged | `ttach` is unmaintained; same transforms, same averaging |
| `nn.DataParallel` and `module.` key prefixes | single device, plain `state_dict` | one laptop GPU; the released files load with `weights_only=True` |
| CRF post-processing (`pydensecrf`) | **skipped** | `pydensecrf` does not build for Python 3.11. The paper reports the CRF as a small gain |
| `argmax` over the 2 classes | `softmax(...)[:, 1]` = p(flood) | our metrics, ensemble confidence and E7 fusion need probabilities; `p > 0.5` equals the argmax |
| `segmentation_models_pytorch` 0.2 | current release | same `Unet` / `UnetPlusPlus` with `mobilenet_v2`; the released state_dicts load unchanged |
| Notebook paths and Kaggle data layout | `etci_rows()` reads the original folder layout or the Hugging Face parquet shards | the dataset is now distributed through Hugging Face |
| Input encoding `s1_to_rgb` | **unchanged**: VV/255, VH/255, `1 - clip(nan_to_num(VH/VV), 0, 1)` | kept bit-for-bit, including the `np.nan_to_num(x, 0)` quirk (0 is the `copy` flag: NaN -> 0, +inf -> huge -> clipped to 1) |
| `filter_df` drops tiles whose VV is only 0/255 | same rule (`is_blank`, `eval.skip_blank: true`) | the original validation numbers exclude those tiles |

Members (release v1.0.0, 108 MB): U-Net and U-Net++ (MobileNetV2, ImageNet-initialised) and the
round-2 pseudo-label U-Net. `python -m src.level1_radar.etci_reproduce download-weights` fetches
them (the `.tar.gz` is an uncompressed tar).

## Expected gap from the paper's 0.7654

- 0.7654 is the **competition test set** IoU (labels only on the competition server). We score the
  public **validation** tiles, a different set of regions.
- No CRF.
- The paper's final submission used more pseudo-labelling rounds and post-processing tuned on
  the leaderboard.
- Our IoU sums TP, FP and FN over the whole split (the competition did the same), and blank tiles are
  dropped as in `filter_df`.

## Status

Weights downloaded and loading checked here. The ETCI images (about 5 GB on Hugging Face) could not
be downloaded in the build environment, so run on the laptop:

```bash
python -m src.level1_radar.etci_reproduce download-data
```
```bash
python -m src.level1_radar.etci_reproduce inspect
```
```bash
python scripts/evaluate.py --config configs/E0.yaml
```

and add the IoU here.
