# Versions

This file maps each paper to the branch, tag, hindcast, and code state that
produced its numbers. Update it whenever a tag is added.

## Task 0 outcome (pre-edit v2 code)

Not resolved in this repository. The v2 benchmark training code
(`train.py`, the legacy training script) and any editor backups of the
version that launched the 209-job run on 2026-09-04 exist only on the
training server. The check for a pre-edit copy of the FNO/TNO spectral
convolution (`train.py~`, `*.bak`, `archive/`) has to be done there before
the `v2.0-run-20260904` tag can be placed. Record the result in this section
when it is known:

- Pre-edit copy found: keep it as `archive/train_v2.0_single_block.py` on the
  `v2-benchmark` branch and tag that state.
- No copy found: state here that the v2.0 FNO/TNO runs used a single-block
  `SpectralConv2d`/`SpectralConv3d`, that the source file was overwritten
  before it was committed, and that those runs are superseded by the v2.1
  re-run.

## Tags

| Tag | Date | Branch | Hindcast | Boundary | Direction transform | Evaluation policy | Known issues |
|---|---|---|---|---|---|---|---|
| `v1.0-apor` | 2026-09-12 (tag); code as posted 2026-05-22 to 2026-08-18 | `main` | v1 | Segments defined in `apor_revision/boundspec_segments.py`: W01-W10 (I = 0), S01-S09 (J = 0), E03-E10 (I = 260); no northern boundary. Misaligned against the ERA5 extraction points (see corrigendum). | Rotation chosen among {0, +90, -90, +180} deg on training times only (E01-E08); `inference_typhoons.py` forces -90 deg. The selected value was -90 deg. | Best-EMA checkpoints (`*_best_ema.pth`), block-stratified test split; E04 uses the chronological 2019-train / 2020-test split. | Hindcast boundary defects described in the corrigendum (in preparation). |
| `v2.0-run-20260904` | pending | `v2-benchmark` | v2 | 37 segments covering north, east, south, and west open boundaries | Reflection theta' = 270 - theta | To be fixed in TASKS before evaluation | FNO/TNO single-block spectral convolution; runs superseded by v2.1 |
| `v2.1` | pending | `v2-benchmark` | v2 | 37 segments | Reflection theta' = 270 - theta | Same as v2.0 | FNO/TNO two-block fix; EM&S submission |

## Notes

- The legacy demo at the repository root (`main.py`, `src/swan_emul/`, the
  `seq6` checkpoint) predates the revision package and is a smoke test only.
  Its outputs are not manuscript numbers.
- `apor_revision/` was briefly named `V2.0.0/` (commit 5690cee, 2026-08-18).
  The rename was reverted so that the path cited in the paper and the README
  resolves. The V2 label belongs to the benchmark branch.
