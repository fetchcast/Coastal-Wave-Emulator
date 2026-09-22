# CLAUDE.md — Coastal-Wave-Emulator

Repository for the SWAN coastal wave emulator and its benchmark (Korea
University, J. Kim & S. Son). Claude Code reads this file first. Everything
below is a standing rule unless a task file says otherwise.

## What this repository is

Neural-network emulators of a Delft3D-FM / SWAN wave hindcast of Korean
coastal waters (261 x 256 grid, 1.8 km, 2019-2020, hourly). Two papers depend
on this repository, and they live on different branches:

| Paper | Branch | Tag | Hindcast | Boundary | Direction transform | Status |
|---|---|---|---|---|---|---|
| APOR 2026 (UNet++-ConvLSTM emulator) | `main` | `v1.0-apor` | v1 | segments defined in `apor_revision/boundspec_segments.py` (W01-W10, S01-S09, E03-E10; no northern boundary); misaligned against the ERA5 extraction points | rotation -90 deg | published; corrigendum in preparation |
| EM&S benchmark (10 architectures) | `v2-benchmark` | `v2.x` | v2 | 37 segments, complete | reflection 270 - theta | in preparation |

`main` carries the code that accompanies the published APOR paper: the legacy
demo (`main.py`, `src/swan_emul/`) and the revision package
(`apor_revision/`, experiments E01-E08). Never edit that code after the
`v1.0-apor` tag. The only additions allowed on `main` afterwards are
`docs/CORRIGENDUM.md` and version bookkeeping in `docs/VERSIONS.md`.

The v2 benchmark code (`train.py`, the legacy training script, evaluation and
follow-up tools) lives on the `v2-benchmark` branch. It is developed on the
training server; commit and tag here before a run, then pull the branch on
the server so that `run_manifest.json` records the commit hash.

## Hard rules

- Never commit data, checkpoints, or results: `runs/`, `*.nc`, `*.pth`, `*.pt`,
  `*.npy`, `bnd_*/`, `logs/`. These are hundreds of GB. `.gitignore` enforces
  it. The small demo checkpoints already tracked under `weights/` and
  `apor_revision/weights/` are the one historical exception; do not add more.
- Never rewrite history (`rebase`, `amend`, `force-push`) on any pushed branch.
- Never modify the benchmark's legacy training script in place while a run is
  active. Follow-up experiments use the `*_followup.py` copy produced by
  `patch_train_fraction.py`.
- Every code change that affects training or evaluation gets a tag before the
  first run that uses it, and `run_manifest.json` must record the commit hash.
- Do not "improve" scientific text or numbers on your own initiative. Numbers in
  manuscripts come only from CSV files produced by the pipeline.

## Facts that must not drift (verified from code)

- Data split: block-stratified, `bh=168, q=5, emb=12` -> train/val/test
  9770/1980/1980 samples (75 training blocks). Sample index t targets raw time
  t + seq_length.
- Training recipe: AdamW, OneCycleLR max_lr 1e-4, weight decay 1e-4, up to 30
  epochs with early stopping (patience 3 on the EMA validation loss), Kendall
  uncertainty-weighted loss (learned `log_vars`, values can be negative), peak
  curriculum oversampling in the training loader.
- Evaluated weights in the legacy pipeline are the final-epoch raw weights;
  `ckpt_best_raw.pth` / `ckpt_best_ema.pth` exist in every run folder.
  The paper's evaluation policy is decided in TASKS; apply it uniformly.
- Boundary (BND) direction channels are transformed by theta' = 270 - theta
  (reflection). The 8-candidate autocorrect confirmed this (score 0.52 vs 0.16
  for the best rotation). In follow-up runs the transform is forced through
  `SWAN_BND_DIR_TRANSFORM=refl+270` and scored on training times only.
- FNO `SpectralConv2d` and TNO `SpectralConv3d` must process BOTH low-frequency
  blocks of the first (non-rfft) axes: `[:mx, :my]` and `[-mx:, :my]`
  (and `[:mt]`/`[-mt:]` for time). The single-block version is a defect;
  runs made with it are superseded and must be labelled as such.
- Direction loss clips sin/cos components and the dot product without
  normalising the predicted vector; this is a known limitation kept for
  comparability within the v2 benchmark. Do not change it inside v2.x.
- Config search included `modes_x != modes_y` (32x64). Always report both.
- Stage-2 config selection uses `val_loss_final` (Kendall loss), not RMSE.

## Layout

`main` (APOR, tag `v1.0-apor`):

```
main.py                                   legacy demo entry point (L = 6)
src/swan_emul/                            legacy demo modules
assets/                                   legacy normalization parameters
data/sample_0010*.zip                     10-step sample inputs
weights/                                  legacy demo checkpoint(s)
apor_revision/                            E01-E08 reproduction package (see its README)
docs/VERSIONS.md                          paper -> branch -> tag -> data mapping
docs/CORRIGENDUM.md                       v1 defects and v2 corrections (added when the journal text is final)
CLAUDE.md, TASKS.md                       standing rules and the task list
```

`v2-benchmark` (EM&S, tags `v2.x`), in addition to everything on `main`:

```
train.py                                  launcher + worker (CONFIG dict at top)
UNET_LSTM_V64_..._9input.py               legacy training script (benchmark)
UNET_LSTM_V64_..._9input_followup.py      patched copy for follow-up (generated, committed)
UNET_LSTM_V64_..._9input_followup.sha256.json   hashes of the original and the copy
benchmark_inference_full_fixed.py         unified evaluation + figures
bnd_leakage_report.py                     boundary corruption test
patch_train_fraction.py                   builds the follow-up legacy copy
followup_launcher.py                      R/D/L/B follow-up experiments (isolated root)
bench_epoch_report.py                     completed epochs / early stop / updates per run
tests/                                    unit tests (spectral conv blocks, split hook, autocorrect, manifest)
archive/                                  v1-config launcher and pre-reflection legacy script (provenance only)
swan_repaired_v1/                         server package with the spectral fix and a proposed protocol change (see docs/VERSIONS.md)
```

## Style for anything written into manuscripts

American English. No `substantial`, `rather than`, `driven by`, `case-to-case`,
`highlight`, `interim`; no `, which` relative clauses; no `, -ing` participial
clauses (`, indicating`); no `\textbf` in body text; minimal em-dashes and
semicolons. Claims proportional to evidence. Code comments in English.

## Working language

Talk to the user in Korean. Code, commit messages, comments, and docs in English.


## September 22, 2026 campaign scope

See `docs/CAMPAIGN_20260922.md` for the added campaign packages and `docs/ASSISTANT_HANDOFF.md` for cross-assistant handoff. The legacy training/selection facts above describe the earlier workflow, not every newly added campaign. The repaired A/B/C campaign uses a fixed successful-update budget, validation Hs MAE with EMA weights, and saved train-only preprocessing. Its selected models use 2019-2020 training data and 2021 held-out evaluation. These policies must not be silently replaced with the legacy Kendall-loss selection or final-raw-weight policy.

The delivered package source and its original Korean READMEs were imported without translation to preserve hashes. New repository-level documentation is in English. Importing these files does not establish that the running server has this Git commit. Do not backfill historical run manifests with the new commit. The v5 three-year-training package is staged code, not a completed experiment. No robustness or additional HPO package has been implemented from the September 22 discussion.

