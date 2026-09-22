# Versions

The current source maintenance version is **v2.1.1** (`VERSION`). Version labels
below are distinct from existing Git tags and do not retrospectively identify
code used on the training server. See [experiment mapping](EXPERIMENT_PROTOCOLS.md).

| Version | Meaning | Status / legacy aliases |
|---|---|---|
| v2.0 | Original September 4 benchmark | Existing tag `v2.0-run-20260904`; FNO/TNO results superseded |
| v2.1 | Repaired two-year development protocol and 2021 evaluation | Campaign family documented from source snapshot `d824b92`; folders v1/v2/v3/v4/v41 are historical package names, not release versions |
| v2.1.1 | Documentation, dependency guidance, and test stabilization | Current maintenance revision; no training/evaluation algorithm changes |
| v2.2 | Three-year development span (2019–2021), held-out 2022 | Planned protocol; existing `swan_three_year_v5` is its staged implementation, not proof of completed runs |

Patch versions (`v2.1.2`, etc.) cover compatible maintenance. Changes to training
years, split, normalization, or evaluation protocol require a new minor protocol
version and their own frozen run metadata. Preserve legacy directories, job IDs,
result paths and internal protocol identifiers; never rename live run roots.
New utilities use descriptive directory names rather than another v-number.

Only the historical v1.0/v2.0 tags listed below were verified to exist. v2.1 and
v2.1.1 here are version labels, not claims that corresponding Git tags exist.
The earlier *unissued* v2.1 proposal meant a spectral-only rerun under the legacy
recipe. It is superseded by this explicit naming policy; no completed run is
relabelled as a spectral-only experiment.

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

## Repaired protocol scope

`swan_repaired_v1` changes more than the spectral block: normalized direction
vectors, best EMA selected by validation Hs MAE, successful-update budgets with
early stopping disabled, and time-gap-aware sampling. The repaired campaign
must be reported separately from the legacy run. Reusing old results from
unrepaired architectures does not establish a common-protocol comparison.

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


## September 22, 2026 campaign source import

The delivered campaign and analysis source packages were added to `v2-benchmark` after the September 12 snapshot. This import preserves the existing repaired trainer and published-code files. See [CAMPAIGN_20260922.md](CAMPAIGN_20260922.md) for package order, dependencies, and experiment status. v4.1 is the current supplied controller; v5 is staged and must not be described as a completed three-year experiment. This source import does not retroactively assign a Git commit or tag to earlier server runs.


