# SWAN repaired benchmark v1

English | [Korean](README_KO.md)

This package was built from the supplied `train.py` and identical legacy source copies. It does not write to the original `~/swan/train.py`, legacy files, `runs/v2_focused_all`, or `runs/v2_followup`. The split patch is integrated; do not run the older `patch_train_fraction.py` or `followup_launcher.py` first.

Read the [time-gap update](UPDATE_TIME_GAP_EN.md) for the supplied 2019-2020 time axis. For subsequent campaign controllers, see the [campaign guide](../docs/CAMPAIGN_20260922.md).

## Initial run

First stop assigning new work through the old benchmark and preserve required checkpoints. This package does not terminate existing processes; it waits for selected GPUs to become free. Install under `/home/jovyan/swan` (original archive: `SWAN_Repaired_Benchmark_v1.zip`).

```bash
cd /home/jovyan/swan
nohup bash swan_repaired_v1/START_REPAIRED.sh --gpus 0,1,2 > repaired_run.log 2>&1 &
tail -f repaired_run.log
```

Default execution runs CPU regression checks, then a two-successful-update real-data smoke test for each of FNO/TNO/FFNO. If all three pass, it trains the three representative pilots, performs final evaluation, and exits. Large follow-up studies are not started automatically.

All pilots use width 64, depth 4, spatial modes 24x24, and seed 42; TNO temporal modes are 4. These are implementation-check anchors, not claimed optimal configurations. Parameter counts differ and are recorded in `training_audit.json`.

```bash
bash swan_repaired_v1/START_REPAIRED.sh --stage smoke --gpus 0,1,2
```

The smoke-only command stops before pilots. Restarting a command reuses verified completed jobs and resumes incomplete jobs from the last complete validation-cycle state. Unsaved work repeats. Changed code/data fingerprints stop execution. Use a new `--root`, such as `runs/repaired_v2`, for a changed experiment.

## Files, inputs, and status

| File | Role |
|---|---|
| `train_repaired.py` | Architectures, repaired FNO/TNO spectral layers, worker |
| `legacy_repaired.py` | Preprocessing, losses, evaluation |
| `repair_support.py` | Splits, direction calibration, sampler, training loop, state |
| `run_repaired.py` | Isolated plans, GPU allocation, result checks |
| `selftest.py` | CPU regressions and optional synthetic NetCDF integration |
| `bench_epoch_report.py` | Read-only training-budget reports |
| `START_REPAIRED.sh` | Check-and-launch entry point |

Default server root is `/home/jovyan/swan`. Required external assets are `wavm-Waves_2019_2020_v2.nc`, `bnd_features.py`, `boundspec_segments.py`, `bnd_2019_v2`, and `bnd_2020_v2`. Existing station CSVs are optional for the legacy station plots; missing stations retain the original skip behavior.

Override paths with `--server-root`, `--data`, `SWAN_BND_DIR_2019`, and `SWAN_BND_DIR_2020`. Use the existing torch, NumPy, pandas, xarray, netCDF4, SciPy, scikit-learn, Matplotlib, and tqdm environment, with a CUDA-compatible PyTorch. No installation or upgrade is automatic.

Default output is `runs/repaired_v1`. The launcher log records starts/completions; each job's detailed output is `attempt_*.log`. NetCDF checks and boundary loading can initially leave the GPU idle. Successful checks are cached by source fingerprint and `time_steps`.

```bash
python3 swan_repaired_v1/run_repaired.py --report
python3 swan_repaired_v1/bench_epoch_report.py --root /home/jovyan/swan/runs/repaired_v1
python3 swan_repaired_v1/bench_epoch_report.py --root /home/jovyan/swan/runs/v2_focused_all
```

The final command reads the old benchmark without changing it.

## Repaired training/evaluation rules

- FNO handles both signed low-frequency blocks; TNO handles four signed blocks. Limits avoid overlapping writes on small/odd grids. FFNO spectral layers are unchanged.
- Direction loss normalizes sin/cos vectors before circular comparison and uses an explicit channel axis. Circular loss does not constrain vector magnitude; raw outputs/vector lengths are saved to diagnose unstable low-radius directions. Direction maps use a cyclic palette, and angular-error maps use minimum differences from 0 to 180 degrees.
- Splits use block size 168, q=5, seed 42, and embargo equal to sequence length. Invalid splits/fractions raise errors without a random/fallback split. Blocks count stored samples; see the time-gap note.
- Fractions 0.25/0.5 target stratified training-block fractions. Rounding can change realized sample fractions; actual indices/counts are saved. Validation/test sets stay fixed.
- Default `train_auto` calibrates only against target times in the common nested 25% training blocks. Validation/test targets are excluded. Supported fractions are at least 0.25. Channel order is [Hs, Tm, sin, cos]; signs do not trigger channel swaps.
- A metadata-verified convention can be fixed with, for example, `--bnd-transform refl+270`. Reflection 270 is not assumed by default, and automatic selection is not proof of a physical convention.
- Final evaluation explicitly loads the EMA checkpoint with lowest validation Hs MAE. Weighted validation loss remains logged but does not select the checkpoint.
- Train and validation MAE use the same EMA weights. Train MAE excludes peak resampling. The online weighted training loss still reflects curriculum and changing weights, so its difference from validation loss is not a direct generalization gap.
- Budgets and validation intervals count successful optimizer updates; early stopping is disabled by default. The interval is `ceil(full-training peak-sampler length / effective batch size)`, with nominal effective batch 4. Pilots run 30 such intervals.
- A logged cycle is this interval, not a native epoch through a reduced dataset. Traversals, batches, optimizer attempts/successes/skips are recorded separately. Fraction experiments use the same full-data-based budget/interval; they do not simply multiply epochs by inverse fraction. Curriculum and log_vars freezing use the update clock.
- The sampler draws twice the number of peak samples with replacement from the upper 5% group; ties affect the actual group size. Curriculum changes spatial loss weights, not sampler length.
- The final incomplete accumulation group divides by its actual batch count. AMP-skipped updates do not advance the scheduler or EMA.
- AdamW applies no weight decay to log_vars. Group-specific OneCycleLR max_lr retains a 0.1 ratio relative to the model body.
- Resume files contain optimizer, scheduler, EMA, scaler, random states, and sampler order/position. Restoration failure stops execution instead of starting with partial state.
- Nonfinite batches are not silently skipped or replaced by fake zeros. Wet-cell source data are checked first. Undefined land direction is masked and filled with zero. Static 2D and time-dependent depth are handled explicitly.
- Each job uses one GPU; multiple GPUs run separate jobs.

Old benchmark metrics are not automatically reused as controls after changing preprocessing/losses. Apply a common policy to final comparison groups. Data fractions can change normalization and peak distributions even at equal updates; they do not isolate a single causal effect of data diversity.

## Follow-up plans and memory

The proposed 22 R/D/L/B jobs plus five full-data controls are preserved as a 27-job follow-up plan, not launched by default.

```bash
python3 swan_repaired_v1/run_repaired.py --stage followup --plan
```

The original memory estimate for repaired FNO `w384 d8 m48` is about 243 GiB for spectral state during EMA evaluation alone, excluding activations/other parameters. It exceeds one B200, so preflight blocks that plan instead of shrinking it silently. Choose a feasible final plan after inspecting pilot results.

An explicitly selected JSON job list can be supplied through `--plan-file`. Changed settings require a new root and successful smoke/pilot checks before follow-up execution. The historical default recommendation was smoke plus pilot only; use the later campaign guide for current scheduling.

## Validation scope

```bash
python3 swan_repaired_v1/selftest.py --extended
```

CPU tests cover even/odd 2D/3D Fourier reconstruction/backpropagation, small full FNO/TNO/FFNO models, circular-loss counterexamples and 359/1-degree wraparound, nested fractions, split preservation, calibration independence from other targets, sampler/dropout resume consistency, optimizer hook/skip behavior, LR ratios, and synthetic NetCDF workers with final best-EMA evaluation and summaries.

Synthetic NetCDF integration used boundary OFF. Server boundary helpers were unavailable in that original development bundle, so they were not executed there; direction-transform logic was separately tested. Server smoke tests cover real-data/GPU execution. CPU resume agreement does not guarantee bitwise identity for every CUDA run. Other architecture structures, coastal weights, Hs/Tm losses/TV terms, and default data paths were preserved from the supplied implementation.
