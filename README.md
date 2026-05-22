# Coastal-Wave-Emulator

UNet++–ConvLSTM emulator of nearshore wave fields, trained on a coupled
Delft3D-FM WAVE hindcast around the Korean Peninsula.

This repository hosts two related but separate things:

1. A **5-minute legacy demo** (`main.py` + `src/swan_emul/`) that loads a
   single pretrained checkpoint and produces a small example
   prediction. This is what was originally posted to accompany the
   first submission.
2. The **APOR revision package** (`apor_revision/`) that reproduces the
   eight controlled experiments (E01–E08) reported in the revised
   manuscript, including the architecture ablations, the boundary-
   descriptor ablation, the chronological-holdout stress test, and the
   three-seed variability run.

Both areas use the same model family but ship different code paths,
different checkpoints, and different reproducibility scopes. **If you
want to reproduce the revised-paper numbers, use `apor_revision/`. If
you only want to see what an emulator output looks like, use the
legacy demo.**

If you use anything in this repository, please cite the paper (see
`apor_revision/CITATION.cff`) and the original UNet++ and ConvLSTM
references.

---

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
└── apor_revision/                  ← revised-paper reproduction (NEW)
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
