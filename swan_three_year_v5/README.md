# SWAN three-year development and 2022 evaluation v5

English | [Korean](README_KO.md)

This staged package trains fresh FNO/FFNO models using 2019-2021 development data in a separate root, then compares the existing two-year models and new models on the same 2022 inputs. It does not edit the original trainer or existing results. Inclusion in this repository does not mean the experiment has completed. See the [campaign guide](../docs/CAMPAIGN_20260922.md) for outstanding data-quality checks.

## Experiment contract

| Item | Setting |
|---|---|
| Controls | Completed v3-selected FNO/FFNO, seeds 42/43/44 |
| New models | Same selected architecture, learning rate, and batch settings, seeds 42/43/44, trained from scratch |
| Development years | Original 2019-2020 data plus 2021 |
| Split | Apply the existing weekly/block and embargo/time-gap rules to the extended data |
| Budget | Same successful-update budget and validation interval as selected controls, not 30 passes through the larger dataset |
| External evaluation | Calendar 2022 and frozen JMA cyclone windows; 12 two-year/three-year model runs |
| Search | None; TNO and the seven other architectures are not newly trained here |
| Default GPUs | After v4 completes, up to four training workers on GPUs 0–3 and up to two evaluation workers |

The three development years are split into train/validation/internal-test sets; they are not all assigned to training. Normalization and direction calibration follow the trainer's train-only rules. Since the new models use 2021, their internal 2021 results are not directly comparable to v4's held-out 2021 results.

The expanded setup also changes splits and normalization, so it does not isolate data quantity alone. It does not establish another-region generalization or statistical superiority. Equal updates mean fewer exposures per sample with more data. Judge convergence from validation curves before viewing 2022; a different budget requires a separately frozen experiment.

## Install and check paths

Install under `/home/jovyan/swan` (original archive: `SWAN_Three_Year_v5.zip`). Before the first preparation:

```bash
cd /home/jovyan/swan
cat swan_three_year_v5/config.json
```

Default inputs:

- `wavm-Waves_2019_2020_v2.nc`.
- `swan_2021_nc_v2/wavm-Waves.nc`.
- `bnd_2019_v2`, `bnd_2020_v2`, `bnd_2021_v2`.
- Assumed 2022 paths: `swan_2022_nc_v2/wavm-Waves.nc` and `bnd_2022_v2`. Correct these before initial preparation if the real paths differ.
- Output root: `runs/iclr_three_year_v5`.

The installed `swan_repaired_v1` and completed v3 selection/result files are required. The code checks times, grid, mask, units, and dimensions. 2021 must be a complete calendar year; the following January 1 endpoint is excluded. The original years retain the controls' `time_steps` extent. Missing hours are not interpolated and are handled by the sequence-gap filter.

Use the existing NumPy, netCDF4, xarray, pandas, Matplotlib, SciPy, and trainer PyTorch/CUDA environment. Do not reinstall a working PyTorch merely to run this package.

## CPU preparation

```bash
cd /home/jovyan/swan
nohup bash swan_three_year_v5/PREPARE.sh > iclr_three_year_v5_prepare.log 2>&1 &
tail -f iclr_three_year_v5_prepare.log
```

All three selected FNO/FFNO seeds must be complete. Preparation does not read 2022 or launch GPU training, but it competes for storage I/O with active jobs.

A new compressed combined NetCDF is written. The preflight space check uses the estimated uncompressed size plus 2 GiB. Eight float32 variables on a 261x256 grid over roughly 26,000 frames occupy about 52 GiB uncompressed; training/evaluation caches and checkpoints require more. Interrupted temporary files are rewritten. Completed combined files are reused only after provenance/mtime checks.

Configuration is frozen on first preparation. Use a separate package copy/result root for changed experiments; do not edit the frozen config to mix results.

## Train after v4 completion

```bash
cd /home/jovyan/swan
nohup bash swan_three_year_v5/START_TRAIN.sh > iclr_three_year_v5_train.log 2>&1 &
python3 swan_three_year_v5/status.py --watch 10
tail -f iclr_three_year_v5_train.log
```

By default, `runs/iclr_parallel_v4/completed.json` must exist. If it does not, training exits without starting. This is not an automatic waiting queue; rerun after v4 completes. Preparation outputs are reused.

Two-update smoke tests precede six fits with up to four concurrent jobs. OOM or validation errors stop execution without silently shrinking a model. Restart reuses completed results and resumes incomplete fits under the trainer's checkpoint policy.

If eight additional GPUs exist on the same server as physical indices 8–15, change `gpus` before initial preparation and set `require_v4_complete` to false to permit concurrency. This does not combine GPUs on another node. The bypass rejects concurrent use of the original 0–7 pool.

## Evaluate after 2022 and training are complete

```bash
cd /home/jovyan/swan
nohup bash swan_three_year_v5/EVALUATE_2022.sh > iclr_three_year_v5_eval.log 2>&1 &
python3 swan_three_year_v5/status.py --watch 10
tail -f iclr_three_year_v5_eval.log
```

Validate full-year 2022 times, grid, and boundary coverage. The permitted boundary source gap is six hours. The first 12 hours provide context, leaving 8,748 targets. Reuse existing JMA tracks if available, otherwise use the evaluator's download procedure. Freeze candidates and events before viewing predictions.

Events use the 400 km wet-grid proximity rule with 24-hour padding. JMA grade-5 lifetime storms are separately identified. High-wave thresholds are truth Hs >=3 m and >=5 m. Outside-event hours are not necessarily calm.

Controls retain their original normalization/checkpoints; new models use their training normalization. Check that physical-unit truth summaries and time axes agree across all 12 evaluations. Reports contain three-seed mean/SD and three-year minus two-year differences, not significance p-values. A negative signed peak bias or timing error is not automatically an improvement.

## Outputs

Under `runs/iclr_three_year_v5`:

- `controls/`: frozen control candidates and preprocessing/checkpoint provenance.
- `data/prepared.json`: source, time-axis, and missing-hour audit.
- `plan.json`, `protocol.json`: frozen training settings.
- `training/`, `results.csv`, `training_completed.json`: six new fits and completion state.
- `events_2022.json`: frozen windows.
- `evaluation/{two_year,three_year}_{fno,ffno}/`: event/hourly tables and figures.
- `comparison_by_seed.csv`, `comparison_summary.csv`: common-2022 comparisons.
- `completed.json`: full comparison completion.

## Implementation and validation

The isolated trainer copy adds 2021 boundary paths and a distinct provenance version. Architectures, losses, and spectral operators are unchanged. `trainer_changes.json` records original/modified SHA256 values. No server trainer file is patched.

```bash
python3 -m unittest discover -s swan_three_year_v5 -p 'test_v5.py' -v
```

Tests cover year endpoints, duplicate rejection, gap recording, the 2022 calendar/JMA filter, synthetic three-year NetCDF merge/reuse, and 12-run truth agreement/mismatch rejection. The NetCDF test requires its optional backend. Actual B200 training and full 2022 evaluation were not run in the development environment.
