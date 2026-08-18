# APOR revision (E01–E08) reproduction package

This directory reproduces the eight controlled experiments behind the
ablation table and the chronological-holdout stress test in the revised
paper. The eight runs share one inference engine
(`inference_ablation_v5_3.py`) and are driven by a single batch script
(`run_all_inference.py`).

The typhoon-case analysis of Section 4.6 (Typhoons Lingling, Bavi,
Maysak, Haishen) uses a separate standalone script
(`inference_typhoons.py`). It enforces a fixed −90° boundary rotation
to match the training setup and is described under
[Typhoon analysis](#typhoon-analysis) below.

---

## Contents

```
apor_revision/
├── README_revision.md
├── requirements.txt
├── CITATION.cff
│
├── inference_ablation_v5_3.py    ← inference engine (driven by --variant, --split, --use_bnd)
├── model_architectures.py        ← model class definitions and build_model()
├── revision_patches.py           ← train-only rotation, chronological split, test metrics
├── bnd_features.py               ← boundary-descriptor construction
├── boundspec_segments.py         ← open-boundary segment definitions
│
├── run_all_inference.py          ← batch runner: E01–E08
├── inference_typhoons.py         ← standalone typhoon analysis (Section 4.6)
│
└── weights/
    ├── ckpt_E01_main_full_block_seed42_bndtrainonly_usebndon_best_ema.pth
    ├── ckpt_E02_convlstm_only_..._best_ema.pth
    ├── ckpt_E03_unetpp_stack_..._best_ema.pth
    ├── ckpt_E04_chrono_2019tr_full_chrono_2019_train_2020_test_..._best_ema.pth
    ├── ckpt_E05_bnd_off_full_block_seed42_bndtrainonly_usebndoff_best_ema.pth
    ├── ckpt_E06_seed7_full_block_seed7_..._best_ema.pth
    ├── ckpt_E07_seed1337_full_block_seed1337_..._best_ema.pth
    └── ckpt_E08_seed2024_full_block_seed2024_..._best_ema.pth
```

---

## Experiment table

| Tag                 | Variant         | Split                          | BND | Purpose                                     |
| ------------------- | --------------- | ------------------------------ | --- | ------------------------------------------- |
| `E01_main`          | full            | block                          | on  | Main run; basis for all reported numbers    |
| `E02_convlstm_only` | convlstm_only   | block                          | on  | Drop UNet++ encoder–decoder                 |
| `E03_unetpp_stack`  | unetpp_stack    | block                          | on  | Drop ConvLSTM recurrent module              |
| `E04_chrono_2019tr` | full            | chrono_2019_train_2020_test    | on  | Train 2019, test 2020 (chronological)       |
| `E05_bnd_off`       | full            | block                          | off | Drop spectra-derived boundary descriptors   |
| `E06_seed7`         | full            | block                          | on  | Seed 7 (variability)                        |
| `E07_seed1337`      | full            | block                          | on  | Seed 1337 (variability)                     |
| `E08_seed2024`      | full            | block                          | on  | Seed 2024 (variability)                     |

**Important.** E04 is the only chronological-split run. All other runs
use the block split. E05 is the only BND-off run (6 input channels);
all others use BND-on (10 input channels). The runner enforces these
matchings automatically based on the experiment table inside
`run_all_inference.py`.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended but not required; the code falls
back to CPU if no CUDA device is found.

`cartopy` requires GEOS and PROJ system libraries. On Ubuntu:
```bash
sudo apt-get install libgeos-dev libproj-dev proj-data proj-bin
```
On macOS with Homebrew:
```bash
brew install geos proj
```

---

## Required external inputs

The batch runner needs two paths set inside `run_all_inference.py`:

```python
DATA_PATH    = "..."   # Delft3D-FM WAVE NetCDF used as the parent-model field
STATION_ROOT = "..."   # folder with per-station observation CSV files
```

`DATA_PATH` should point at the same NetCDF used during training
(`wavm-Waves_2019_2020_final.nc`). The full file is hosted on Zenodo;
see the paper's data-availability statement for the DOI.

`STATION_ROOT` should contain the nine KHOA buoy CSV files in the
exact subdirectory layout used at training time. If the folder is
missing, the ablation table is still produced; only the
station-by-station tables become NaN.

A 10-step sample NetCDF is shipped at the repository root in
`data/sample_0010_with_bnd.zip`. It is enough to confirm that the
inference engine runs end to end and produces a `*_aggregated.csv`,
but it is **not** enough to reproduce manuscript-scale numbers
(those require the full 2-year hindcast).

---

## Reproducing all eight experiments

Default — run all eight in sequence:

```bash
python run_all_inference.py
```

Run only a subset:

```bash
python run_all_inference.py --only E01_main E04_chrono_2019tr
```

Skip specific experiments:

```bash
python run_all_inference.py --skip E05_bnd_off
```

Print the commands without executing:

```bash
python run_all_inference.py --dry-run
```

Continue past a failure instead of stopping:

```bash
python run_all_inference.py --keep-going
```

Each run writes to its own timestamped folder under `./outputs/`
(prefixed by the experiment tag), so runs never overwrite each other.

---

## Reproducing a single experiment manually

Each run is one `inference_ablation_v5_3.py` invocation. For example,
to reproduce E01 (main run) by hand:

```bash
python inference_ablation_v5_3.py \
    --data_path    /path/to/wavm-Waves_2019_2020_final.nc \
    --weights      weights/ckpt_E01_main_full_block_seed42_bndtrainonly_usebndon_best_ema.pth \
    --station_root /path/to/station_csvs \
    --variant      full \
    --split        block \
    --use_bnd      on \
    --feat         32 64 128 256 512 \
    --seq_length   12 \
    --tag          E01_main
```

For E04 (chronological holdout):

```bash
python inference_ablation_v5_3.py \
    --data_path    /path/to/wavm-Waves_2019_2020_final.nc \
    --weights      weights/ckpt_E04_chrono_2019tr_..._best_ema.pth \
    --station_root /path/to/station_csvs \
    --variant      full \
    --split        chrono_2019_train_2020_test \
    --use_bnd      on \
    --feat         32 64 128 256 512 \
    --seq_length   12 \
    --tag          E04_chrono_2019tr
```

The `--split` value must match the way the checkpoint was trained.
For E04 this is what produces the train-only normalization statistics
and the train-only boundary-direction rotation; using the block split
on the E04 checkpoint would silently produce wrong numbers.

---

## Outputs

Each run writes the following into `outputs/<tag>_<timestamp>/`:

- `aggregated_metrics.csv` — area-weighted RMSE, MAE, bias on the test
  set for Hs (m), Tm (s), and directional metrics (degrees).
- `station_metrics_*.csv` — per-station skill against KHOA buoys
  (NaN-filled if `STATION_ROOT` is missing).
- `bnd_rotation_scores.csv` — train-only direction-alignment scores
  for the four candidate rotations {0°, +90°, −90°, +180°}.
- Per-variable figures saved as PNG (and optionally SVG with
  `--save_svg`).

The aggregated CSV is the single file consumed when populating the
ablation table in the paper.

---

## Typhoon analysis

Section 4.6 of the paper uses a separate standalone script,
`inference_typhoons.py`. It reads the same `wavm-Waves_2019_2020_final.nc`,
applies the fixed −90° boundary-direction rotation (matching the
training setup), and produces typhoon-window time-series plots, peak
diagnostics, and the runtime benchmark.

```bash
python inference_typhoons.py \
    --data_path  /path/to/wavm-Waves_2019_2020_final.nc \
    --model_path weights/ckpt_E01_main_full_block_seed42_bndtrainonly_usebndon_best_ema.pth \
    --bnd \
    --bnd_force_deg -90.0 \
    --typhoon_table \
    --speed_benchmark
```

The `--bnd_force_deg -90.0` flag is the default, and the script's
help text describes it as the value matching the training setup.

---

## Notes on numerical reproducibility

Floating-point reductions on GPU are not bit-identical across
hardware. Aggregated metrics should reproduce to the precision
reported in the paper (typically the third decimal); per-pixel
predictions can differ at the level of single-precision noise. Single-
seed runs (E06–E08) are precisely for quantifying this variability;
the seed-42 main run (E01) is the value reported throughout the text.

---

## Citation

See `CITATION.cff` in this directory.
