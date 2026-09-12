# Coastal-Wave-Emulator

Neural-network emulators of a Delft3D-FM / SWAN wave hindcast of Korean
coastal waters (261 x 256 grid, 1.8 km, hourly, 2019-2020). Two papers depend
on this repository and live on different branches.

| Paper | Branch | Tag | Hindcast | Status |
|---|---|---|---|---|
| Applied Ocean Research 2026 (UNet++–ConvLSTM emulator) | `main` | `v1.0-apor` | v1 | published; corrigendum in preparation |
| Ten-architecture benchmark (in preparation) | `v2-benchmark` | `v2.0-run-20260904`, `v2.1` (pending) | v2 | this branch |

`docs/VERSIONS.md` records each tag with its hindcast version, boundary
definition, direction transform, evaluation policy, and known issues.
`CLAUDE.md` holds the standing rules for anyone (or any agent) editing the
repository.

---

## Benchmark (`v2-benchmark` branch)

Ten architectures (ConvLSTM, UNet-LSTM, FNO, F-FNO, TNO, U-FFNO, Swin, ViT,
ConvNeXt-LSTM, Conv-Swin) are trained under one recipe on the v2 hindcast:
block-stratified split (`bh=168, q=5, emb=12`, train/val/test
9770/1980/1980), AdamW with a OneCycle schedule (max_lr 1e-4, weight decay
1e-4), up to 30 epochs with early stopping, Kendall uncertainty-weighted
loss, and peak-curriculum oversampling. Boundary direction channels are
transformed by theta' = 270 - theta.

### Layout

```
train.py                                   launcher + worker; edit the CONFIG dict at the top
UNET_LSTM_V64_..._9input.py                legacy training script called by every worker
UNET_LSTM_V64_..._9input_followup.py       generated copy with the follow-up hooks (see below)
benchmark_inference_full_fixed.py          stage-2 evaluation: physical metrics and figures
bnd_leakage_report.py                      boundary-corruption control test
bench_epoch_report.py                      epochs completed, early stop, updates per run
patch_train_fraction.py                    regenerates the follow-up copy from the legacy script
followup_launcher.py                       R/D/L/B follow-up experiments in an isolated results root
tests/                                     CPU unit tests (no data needed)
archive/                                   v1-config launcher and pre-reflection legacy script (provenance)
swan_repaired_v1/                          server package with the spectral fix and a proposed protocol change
docs/VERSIONS.md                           tag table and open decisions
```

### Requirements

Python 3.10 or later, PyTorch 2.x with CUDA for training, and numpy, pandas,
xarray, netCDF4, scipy, matplotlib, tqdm, wavespectra (boundary spectra).
The server that produced the benchmark ran Python 3.11, torch 2.9.1+cu130,
numpy 1.26 on NVIDIA B200 GPUs. The unit tests need only numpy, pandas,
pytest, and a CPU build of torch.

### Running the benchmark

All paths and the search space are hard-coded in the `CONFIG` dict at the
top of `train.py` (data file, boundary folders, results root, GPUs). Adjust
them, commit, then launch:

```bash
python train.py
```

`train.py` plans the jobs, dispatches one worker per GPU through
`torch.distributed.run`, and writes one folder per run under
`CONFIG["results_root"]`. Each run folder holds `run_manifest.json`
(config, job, resolved paths, `git_commit`, `git_dirty`), the legacy
training outputs, `ckpt_best_raw.pth` / `ckpt_best_ema.pth`, and
`run_summary.json`. Stage 2 (multi-seed and boundary-off ablation) starts
automatically after the sweep when `run_stage2_after_sweep` is on; the
stage-2 configuration is selected by `val_loss_final`.

Tag the commit before the first run that uses it, and pull that commit on
the server so the manifest records it.

### Follow-up experiments

The follow-up (weight decay, data fraction, learning rate, schedule length)
never edits the benchmark scripts in place.

```bash
python patch_train_fraction.py        # regenerates UNET_..._followup.py from the legacy script
python followup_launcher.py --plan    # list the jobs
python followup_launcher.py --run --gpus 0,1,2,3
python followup_launcher.py --analyze
```

`patch_train_fraction.py` applies four edits to a copy of the legacy script
(training-fraction hook, fraction validation, no split fallback under a
fraction, train-only direction scoring with a `SWAN_BND_DIR_TRANSFORM`
override) and records the SHA-256 of both files. The generated copy is
committed so that the follow-up runs are reproducible from the repository.
Both tools carry the server path `/home/jovyan/swan` at the top; change it
for another machine.

### Evaluation

```bash
python benchmark_inference_full_fixed.py --probe                 # inspect the legacy module
python benchmark_inference_full_fixed.py --results-root <root> --out <dir>
python bnd_leakage_report.py --root <root>                       # after the --bnd-corrupt runs
python bench_epoch_report.py --csv epochs.csv
```

The evaluation rebuilds each architecture from `train.py`'s registry, loads
the checkpoint strictly, reuses the legacy data pipeline for the test split,
and checks its recomputed Hs RMSE against `run_summary.json` before any
number is reported.

### Tests

```bash
pip install numpy pandas pytest torch
pytest -q tests
```

`tests/test_spectral_blocks.py` is the regression test for the two-block
spectral convolution: it fails on the code tagged `v2.0-run-20260904` and
passes from the fix onward. The other tests cover the training-fraction
hook, the train-only direction autocorrect, and the manifest fields. The
tests compile functions out of the legacy scripts without importing them,
so they run on CPU without the hindcast.

### Data availability

The v2 hindcast (`wavm-Waves_2019_2020_v2.nc`), the boundary spectra
(`bnd_2019_v2/`, `bnd_2020_v2/`), run folders, checkpoints, and result
tables are not in this repository; `.gitignore` excludes them. They are
archived on Zenodo with each release. The APOR paper's v1 hindcast is
referenced in that paper's data-availability statement.

### Citation

Benchmark paper: in preparation. Until it is published, cite the repository
tag you used (`v2.0-run-20260904` or later) and the APOR paper below for the
emulator family. A machine-readable citation file for the APOR paper is at
`apor_revision/CITATION.cff`.

---

## APOR paper code (`main` branch)

The code below accompanies the published Applied Ocean Research paper and is
frozen at tag `v1.0-apor`. Its hindcast has known open-boundary defects
(segment placement, missing northern boundary, transposed boundary feature
map); a corrigendum is in preparation and will be mirrored in
`docs/CORRIGENDUM.md`. The sections that follow are the original README of
that code.

## Two reproducibility paths

### Path A — Legacy quick-start demo (L = 6)

A minimal, inference-only pipeline that runs one checkpoint on a
10-step sample dataset. Useful as a smoke test or a first look. **The
numbers it produces are not the manuscript numbers.** See
[Legacy demo](#legacy-demo-5-minute-smoke-test) below.

### Path B — APOR revision experiments (L = 12, E01–E08)

The eight experiments behind Tables 5–6 of the revised manuscript:
main run (E01), architecture ablations (E02, E03), chronological
holdout (E04), boundary-off ablation (E05), and three additional
seeds (E06, E07, E08). One batch script runs all of them. See
[`apor_revision/README_revision.md`](apor_revision/README_revision.md).

The two paths use different checkpoint files, different input-sequence
lengths, and different inference scripts; they should not be mixed.

---

## Repository layout

```
Coastal-Wave-Emulator/
├── README.md                       ← this file
├── LICENSE                         ← Apache-2.0
│
├── main.py                         ← legacy demo entry point (L = 6)
├── src/swan_emul/                  ← legacy demo modules
├── assets/norm_params_pctl.json    ← legacy normalization params
├── data/
│   ├── sample_0010.zip             ← legacy 10-step sample
│   └── sample_0010_with_bnd.zip    ← legacy 10-step sample with bnd
├── weights/                        ← legacy single checkpoint
│   └── 20250906_..._seq6_..._bndON.pth
├── figure/                         ← study-region figures
├── maysak_hs.gif                   ← example typhoon animation
├── requirements.txt                ← legacy demo requirements
│
├── docs/
│   ├── VERSIONS.md                 ← paper → branch → tag → data mapping
│   └── CORRIGENDUM.md              ← v1 hindcast defects (added when the journal text is final)
├── CLAUDE.md, TASKS.md             ← repository rules and task list
│
└── apor_revision/                  ← revised-paper reproduction
    ├── README_revision.md
    ├── requirements.txt
    ├── CITATION.cff
    ├── inference_ablation_v5_3.py
    ├── model_architectures.py
    ├── revision_patches.py
    ├── bnd_features.py
    ├── boundspec_segments.py
    ├── run_all_inference.py
    ├── inference_typhoons.py
    └── weights/
        ├── ckpt_E01_main_full_block_seed42_bndtrainonly_usebndon_best_ema.pth
        ├── ckpt_E02_convlstm_only_..._best_ema.pth
        ├── ckpt_E03_unetpp_stack_..._best_ema.pth
        ├── ckpt_E04_chrono_2019tr_..._best_ema.pth
        ├── ckpt_E05_bnd_off_..._best_ema.pth
        ├── ckpt_E06_seed7_..._best_ema.pth
        ├── ckpt_E07_seed1337_..._best_ema.pth
        └── ckpt_E08_seed2024_..._best_ema.pth
```

---

## Legacy demo (5-minute smoke test)

A single L = 6 checkpoint, a 10-step sample dataset, and a thin CLI
that prints predictions. Use this only if you want a fast look at what
the emulator output looks like; the numbers are **not** the manuscript
numbers.

**Install:**

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Unzip the sample:**

```bash
unzip data/sample_0010.zip -d data/
unzip data/sample_0010_with_bnd.zip -d data/
```

**Run inference (with boundary channels in the NetCDF):**

```bash
python main.py \
    --checkpoint weights/20250906_032209_model_weights_17498_seq6_epochs20_hid128_UNET32_bndON.pth \
    --input_nc  data/sample_0010_with_bnd.nc \
    --norm_json assets/norm_params_pctl.json \
    --seq_len 6 \
    --bnd on \
    --device cpu \
    --outdir outputs/demo \
    --denorm off
```

**Run inference (without boundary channels):**

```bash
python main.py \
    --checkpoint weights/20250906_032209_model_weights_17498_seq6_epochs20_hid128_UNET32_bndON.pth \
    --input_nc  data/sample_0010.nc \
    --norm_json assets/norm_params_pctl.json \
    --seq_len 6 \
    --bnd auto \
    --device cpu \
    --outdir outputs/demo \
    --denorm off
```

With a 10-step input and L = 6, the model produces 4 prediction frames.
Outputs are already in physical units: `hs` (m), `tm` (s), `dir`
(degrees, 0–360°).

---

## APOR revision experiments

Everything for the revised paper lives in `apor_revision/`. See
[`apor_revision/README_revision.md`](apor_revision/README_revision.md)
for details. Brief summary:

| Tag                 | Variant         | Split                          | BND | Purpose                              |
| ------------------- | --------------- | ------------------------------ | --- | ------------------------------------ |
| `E01_main`          | full            | block                          | on  | Main run reported throughout text    |
| `E02_convlstm_only` | convlstm_only   | block                          | on  | Architecture ablation (no UNet++)    |
| `E03_unetpp_stack`  | unetpp_stack    | block                          | on  | Architecture ablation (no ConvLSTM)  |
| `E04_chrono_2019tr` | full            | chrono_2019_train_2020_test    | on  | Chronological-holdout stress test    |
| `E05_bnd_off`       | full            | block                          | off | Boundary-descriptor ablation         |
| `E06_seed7`         | full            | block                          | on  | Multi-seed variability               |
| `E07_seed1337`      | full            | block                          | on  | Multi-seed variability               |
| `E08_seed2024`      | full            | block                          | on  | Multi-seed variability               |

To reproduce all eight in sequence on one machine:

```bash
cd apor_revision
pip install -r requirements.txt
python run_all_inference.py
```

---

## License

Apache-2.0. See `LICENSE`.

## Citation

If you use this code or the trained weights, please cite the paper.
A machine-readable citation file is provided in
`apor_revision/CITATION.cff`.
