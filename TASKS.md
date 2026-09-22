> **Current source version: v2.1.1.** [Version policy](docs/VERSIONS.md) and [experiment mapping](docs/EXPERIMENT_PROTOCOLS.md) define the current protocol scope. Folder labels v3/v4/v41/v5 are retained compatibility aliases. Historical task/status statements below are not evidence of current server completion.

# Historical tasks — GitHub setup for the SWAN emulator benchmark

## Status (2026-09-12)

The tasks below were written for the training server (`/home/jovyan/swan`).
They were carried out in the `fetchcast/Coastal-Wave-Emulator` repository
from a code bundle exported from the server; the server file system was
inspected through that bundle (timestamps, hashes, diffs).

Done:

- Task 0: resolved. The pre-edit v2 `train.py` was never overwritten (the
  two-block edit lives in `swan_repaired_v1/train_repaired.py`). Recorded in
  `docs/VERSIONS.md`.
- Task 1: `.gitignore`; `V2.0.0/` renamed back to `apor_revision/`;
  `CITATION.cff` extension restored; placeholder files removed; the seven
  code files plus `archive/` committed on `v2-benchmark`.
- Task 2: `main` = APOR code, tag `v1.0-apor`; `v2-benchmark` branched from
  it; tag `v2.0-run-20260904` on the imported server code. `v2.1` is not
  tagged: the FNO/TNO re-run has not been launched.
- Task 3: `git_commit` and `git_dirty` in `run_manifest.json`.
- Task 4: `tests/` (spectral blocks, split hook, autocorrect, manifest);
  `pytest -q tests` passes (26 tests) on CPU without data.
- Task 5: README (benchmark usage on `v2-benchmark`), `docs/VERSIONS.md`.
  `docs/CORRIGENDUM.md` is held back until the `[verify]` values in the
  draft are confirmed against the journal text.
- Task 6: `main`, `v2-benchmark`, and both tags pushed to
  `fetchcast/Coastal-Wave-Emulator`.

Remaining:

- Task 2: tag `v2.1` when the FNO/TNO re-run starts from the committed
  `train.py`. Pull the branch onto the server first so that the run uses
  the committed file and the manifest records the hash.
- Task 5: `docs/CORRIGENDUM.md` on `main` after verification.
- Task 6: enable the Zenodo-GitHub integration (browser step).
- Task 7: separate `oe-buoy-validation` repository.
- Decision: scope of `swan_repaired_v1` (see `docs/VERSIONS.md`).

---

# Original task list (as supplied)

Claude Code: work through these in order. Each task has an acceptance check.
Stop and report if a check fails. Do not skip Task 0. Talk to the user in
Korean; write everything committed in English.

Working directory: `/home/jovyan/swan` (server). Remote: the user's GitHub
account, repository name `swan-emulator-benchmark` (create it empty on GitHub
if it does not exist; do not initialise it with a README there).

---

## Task 0 — Preserve the code that produced the current runs (do this first)

The 209-job v2 run started 2026-09-04. FNO/TNO spectral convolutions were
edited afterwards. The exact pre-edit v2 code must be recoverable.

1. Look for any copy of the pre-edit v2 `train.py`: `train.py~`, `*.bak`,
   `train_v1_backup.py` (that one is the v1-CONFIG version, not v2), editor
   backups, or the version inside `runs/v2_focused_all/config_snapshot.json`
   (that file holds CONFIG only, not code).
2. If a pre-edit v2 copy exists, keep it as `archive/train_v2.0_single_block.py`
   and note its origin in `docs/VERSIONS.md`.
3. If none exists, write in `docs/VERSIONS.md` that the v2.0 FNO/TNO runs used a
   single-block `SpectralConv2d`/`SpectralConv3d` (as described in
   `CLAUDE.md`), that the source file was overwritten before it was committed,
   and that those runs are superseded by the v2.1 re-run.

Check: `docs/VERSIONS.md` exists and states one of the two outcomes.

## Task 1 — Repository, .gitignore, snapshot of the current state

```
git init  (if needed)
```
Create `.gitignore`:
```
runs/
logs/
*.nc
*.pth
*.pt
*.npy
*.npz
bnd_*/
__pycache__/
*.pyc
*.log
# results; whitelist small config CSVs explicitly if any are needed
*.csv
!station_meta.csv
.venv/
```
Commit the current code files only (never `runs/`):
`train.py`, the legacy script, `benchmark_inference_full_fixed.py`,
`bnd_leakage_report.py`, `patch_train_fraction.py`, `followup_launcher.py`,
`bench_epoch_report.py`, `CLAUDE.md`, `TASKS.md`.

Check: `git status` shows no `runs/`, `*.nc`, `*.pth` staged;
`git ls-files | wc -l` is small (< 30).

## Task 2 — Branches and tags

- `main` = the APOR published version. If the current working tree is already
  v2, ask the user for the v1 code (they have `train_v1_backup.py` and the
  pre-reflection legacy backup `*_before_reflection_patch.py` /
  `*_v1_original.py`). Put v1 on `main`, tag `v1.0-apor`.
- Branch `v2-benchmark` from `main` with the current v2 code; tag the state
  that corresponds to the running 209 jobs as `v2.0-run-20260904`
  (use the archive copy from Task 0 if the FNO/TNO edit is already in the tree,
  otherwise the tree itself).
- The FNO/TNO two-block fix and everything after it is `v2.1`; tag when the
  re-run of FNO/TNO is launched.

Check: `git tag` lists `v1.0-apor`, `v2.0-run-20260904`; `git log --oneline
--graph --all` shows v2-benchmark branching from main.

## Task 3 — Record the commit hash in every run manifest

In `train.py`, where `run_manifest.json` is written (search `run_manifest`),
add `"git_commit"`: output of `git rev-parse HEAD` (fall back to `"unknown"`
if git is unavailable) and `"git_dirty"`: whether `git status --porcelain` is
non-empty. Do not change anything else in that function.

Check: a dry `python -c "import train"` succeeds; a unit test builds the
manifest dict and finds both keys.

## Task 4 — Unit tests (`tests/`)

Use `pytest`. Tests must run on CPU without data.

1. `test_spectral_blocks.py`: for `SpectralConv2d` and `SpectralConv3d`, pass
   `cos(2*pi*(3x+4y))` and `cos(2*pi*(-3x+4y))` through the spectral path with
   identity weights on the retained modes; both must reconstruct within 1e-4.
   This is the regression test for the two-block fix.
2. `test_split_hook.py`: extract `make_block_stratified_split` from the
   follow-up legacy copy; assert (a) unset env and `1.0` give indices identical
   to the original function, (b) `0.5` keeps val/test identical and a training
   subset, (c) `0.25` is nested in `0.5`, (d) `0`, `-1`, `1.1`, `nan`, `abc`
   raise `ValueError` from `robust_block_split` (no fallback).
3. `test_autocorrect.py`: synthetic BND/dir fields with a different convention
   in the "test" frames; train-only scoring must pick `refl+270`; forced
   `SWAN_BND_DIR_TRANSFORM=refl+270` is applied; `bogus` raises.

Check: `pytest -q tests` passes.

## Task 5 — Documentation

- `README.md`: purpose, the version table from `CLAUDE.md`, how to run the
  benchmark (`python train.py`), the follow-up (`patch_train_fraction.py` then
  `followup_launcher.py`), and evaluation; data availability statement
  (hindcast on Zenodo, not in git); citation placeholders.
- `docs/CORRIGENDUM.md`: from the draft supplied by the user (`CORRIGENDUM_draft.md`).
  Link the journal corrigendum DOI when available.
- `docs/VERSIONS.md`: from Task 0, plus a row per tag with date, hindcast
  version, boundary description, direction transform, evaluation policy,
  known issues.

Check: all three files exist; README links to both docs.

## Task 6 — Push and Zenodo

1. Authentication (user does the browser step):
   `ssh-keygen -t ed25519 -C "swan-server"` -> user adds the public key at
   GitHub Settings > SSH and GPG keys -> `ssh -T git@github.com` succeeds.
   Alternative: `gh auth login` if `gh` is installed.
2. `git remote add origin git@github.com:<user>/swan-emulator-benchmark.git`
3. `git push -u origin main v2-benchmark --tags`
4. Tell the user to enable the Zenodo-GitHub integration for this repository
   (zenodo.org > GitHub) so that each GitHub Release gets a DOI. Releases are
   created at submission time, not now.

Check: `git ls-remote origin` shows both branches and both tags.

## Task 7 — Separate repository for the OE buoy paper (later)

`oe-buoy-validation` with `integrated_loader.py`, `validate_pipeline_v2.py`,
`make_paper_figures_v2.py`, `journal_style_v5.py`, `station_meta.csv`, and
the manuscript. Raw buoy files stay out of git. This repository has its own
`CLAUDE.md` (see `OE_PROJECT_BRIEF.md`). Do not mix it into this repository.

---

## Do not

- Do not commit anything under `runs/`, even "small" summaries; the analysis
  CSVs that the papers cite are archived on Zenodo with the release.
- Do not touch `main` after `v1.0-apor` except to add `docs/CORRIGENDUM.md`.
- Do not run or stop training jobs. Repository work only.


## September 22, 2026 source synchronization

Added the delivered campaign packages through v4.1, the staged v5 workflow, evidence/plotting utilities, and event diagnostics. See `docs/CAMPAIGN_20260922.md` for the authoritative package status. The four `swan_repaired_v1` trainer sources already match the supplied server copies and were not replaced. Existing model code, published APOR files, results, and running jobs were not changed.

Validation at import: 70 package unit tests, 69 passed and one synthetic NetCDF test skipped because its optional backend is unavailable; Python source parsing and package SHA256 verification passed. These are local CPU checks, not new full-GPU training runs. Tag the reviewed source before the next new experiment as required above; previously started runs retain their original provenance.


