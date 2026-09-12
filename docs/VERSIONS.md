# Versions

This file maps each paper to the branch, tag, hindcast, and code state that
produced its numbers. Update it whenever a tag is added.

## Task 0 outcome (pre-edit v2 code): resolved, original intact

The code that launched the 209-job v2 run on 2026-09-04 was never
overwritten. Evidence gathered on the training server on 2026-09-12:

- `~/swan/train.py` last modified 2026-09-04 01:45:30 UTC, sha256
  `530182bd...`. The run root `runs/v2_focused_all/config_snapshot.json`
  was written 2026-09-04 06:59:59 UTC and matches CONFIG in that file.
- `~/swan/UNET_LSTM_V64_..._9input.py` last modified 2026-09-04 06:59:14
  UTC, sha256 `8a36d585...`; the `*_270_autocorrect_v2.py` copy is
  byte-identical. Every `run_manifest.json` names this script.
- `SpectralConv2d` and `SpectralConv3d` in that `train.py` process only the
  `[:mx, :my]` block (single-block). No file under `~/swan` older than the
  run contains a two-block version of these classes; the two-block edit was
  made in a separate file, `swan_repaired_v1/train_repaired.py`
  (2026-09-12), so the original needed no recovery.

Tag `v2.0-run-20260904` pins that state (commit 55b2d07 on
`v2-benchmark`). The FNO and TNO results of that run are superseded.

## Tags

| Tag | Date | Branch | Hindcast | Boundary | Direction transform | Evaluation policy | Known issues |
|---|---|---|---|---|---|---|---|
| `v1.0-apor` | 2026-09-12 (tag); code as posted 2026-05-22 to 2026-08-18 | `main` | v1 | Segments defined in `apor_revision/boundspec_segments.py`: W01-W10 (I = 0), S01-S09 (J = 0), E03-E10 (I = 260); no northern boundary. Misaligned against the ERA5 extraction points (see corrigendum). | Rotation chosen among {0, +90, -90, +180} deg on training times only (E01-E08); `inference_typhoons.py` forces -90 deg. The selected value was -90 deg. | Best-EMA checkpoints (`*_best_ema.pth`), block-stratified test split; E04 uses the chronological 2019-train / 2020-test split. | Hindcast boundary defects described in the corrigendum (in preparation). |
| `v2.0-run-20260904` | 2026-09-04 (run); tagged 2026-09-12 | `v2-benchmark` | v2 | 37 segments covering north, east, south, and west open boundaries | Reflection theta' = 270 - theta, selected by the 8-candidate autocorrect (score 0.52 vs 0.16 for the best rotation) | Legacy pipeline evaluates the final-epoch raw weights; `ckpt_best_raw.pth` and `ckpt_best_ema.pth` are saved in every run folder. The paper's policy is fixed in TASKS before evaluation. | FNO/TNO single-block spectral convolution: FNO and TNO runs superseded. Direction loss does not normalize the predicted vector (kept for comparability within v2.x). |
| `v2.1` | pending: tag when the FNO/TNO re-run is launched | `v2-benchmark` | v2 | 37 segments | Reflection theta' = 270 - theta; follow-up runs force `SWAN_BND_DIR_TRANSFORM=refl+270` and score on training times only | Same as v2.0 | Two-block fix in `train.py` (commit 7ad923f, ported verbatim from `swan_repaired_v1/train_repaired.py`); `run_manifest.json` records `git_commit` and `git_dirty` (commit 6f6f393). |

## Open decision: scope of `swan_repaired_v1`

`swan_repaired_v1/` (archived unchanged from the server) contains more than
the spectral fix. It changes the direction loss (unit-normalized vectors),
the checkpoint policy (best EMA by validation Hs MAE, loaded for the test
evaluation), the training budget (optimizer updates instead of epochs,
early stopping off), and the split hook integration. Running the benchmark
with that package is a protocol change and its results are not comparable
with `v2.0-run-20260904` or with a `v2.1` re-run that changes only the
spectral classes. If that protocol is adopted, it needs its own tag and a
full re-run of every architecture, not only FNO and TNO.

## Notes

- The legacy demo at the repository root (`main.py`, `src/swan_emul/`, the
  `seq6` checkpoint) predates the revision package and is a smoke test only.
  Its outputs are not manuscript numbers.
- `apor_revision/` was briefly named `V2.0.0/` (commit 5690cee, 2026-08-18).
  The rename was reverted so that the path cited in the paper and the README
  resolves. The V2 label belongs to the benchmark branch.
- `archive/train_v1_config.py` is `train.py` with the v1-hindcast paths
  (server name `train_v1_backup.py`); `archive/UNET_LSTM_V64_..._v1_original.py`
  is the legacy script before the reflection candidates were added to the
  autocorrect. Both are kept for provenance and are not run.
