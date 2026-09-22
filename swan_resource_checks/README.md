# Frozen-model resource and robustness checks (source v2.1.1)

This additive utility uses the existing two-year models and completed per-family
2021 evaluations. It does not launch three-year training, change candidates,
modify existing checkpoints, or inspect 2022. English is the canonical documentation.

Before a manuscript-producing GPU run, tag the delivered source commit as
`v2.1.1`, following the repository policy. `SOURCE_REVISION.json` in the ZIP
records the exact commit; `VERSION` alone is not evidence of an existing Git tag.
The utility saves that source revision in its frozen plan.

## Start

Run in the existing SWAN CUDA environment (numpy, matplotlib, scipy, xarray,
pandas and the trainer's dependencies). The archive contains only this utility;
it does not overwrite campaign controllers.

```bash
cd /home/jovyan/swan
unzip SWAN_Resource_Checks_2_1_1.zip
python3 swan_resource_checks/run.py inspect
nohup bash swan_resource_checks/START.sh --seeds 42 \
  > /home/jovyan/swan/resource_checks_2_1_1.log 2>&1 &
python3 swan_resource_checks/run.py status
tail -n 50 /home/jovyan/swan/resource_checks_2_1_1.log
```

The suggested first pass uses seed 42 for all ready models; it is a diagnostic
pilot, not a statistical superiority result. For three seeds use a new root:

```bash
nohup bash swan_resource_checks/START.sh --seeds 42,43,44 \
  --output /home/jovyan/swan/runs/resource_checks_2_1_1_three_seeds \
  > /home/jovyan/swan/resource_checks_three_seeds.log 2>&1 &
```

Default evaluation root: `/home/jovyan/swan/runs/iclr_parallel_v4/evaluation`.
Expected files: `<family>/models/<family>_s42/result.json`, `hourly.csv`,
`<family>/prepared/<family>_s42.json`, `events_2021.json`, `selected_2021.json`.
Use `--eval-root` if different. Incomplete evaluations are listed as unavailable,
not silently interpreted as failures or included in results. They are not added
mid-run; after they finish, use a new output root with the desired frozen set.

## GPU scheduling and priorities

1. Full-hour event arrays and existing diagnostics for events 2109/2112/2114.
2. Warmed-up inference memory and latency, with the exact checkpoint/model config.
3. Input-error sensitivity on the same frozen events (default one hour in six).
4. Optional short learning-rate pilots, only after all preceding tasks finish.

Each model/seed follows that order. Ready higher-priority work is scheduled first,
but a slow model does not prevent others moving forward. `--gpus auto` inspects
physical GPUs before launch and skips compute-busy devices; `--max-workers 7`
limits this queue to seven concurrent jobs. To reserve GPU 4 regardless of its
state, use `--gpus 0,1,2,3,5,6,7`. Advisory GPU locks coordinate instances of this
utility only; unrelated launchers do not obey them. Avoid concurrent launchers
claiming the same idle devices. Existing processes are never terminated.

`--launch-hours 12` stops NEW launches after 12 hours; active jobs drain normally.
It is not a 12-hour completion promise or hard training time cap. SIGINT/SIGTERM
also drains owned jobs. The same command resumes after a failure or deadline.
A changed sample interval/model set/source requires a new output root.

Default event-array root is the existing `runs/iclr_event_diagnostics_v1`.
Saved arrays are validated against the original hourly evaluation before reuse;
missing hours are inferred. Only event arrays and their existing diagnostic
summaries may be regenerated there. Do not run another diagnostics controller
against that root concurrently. Checkpoint/cache signatures and trainer hashes
must match. Original checkpoints/results are read-only.

## Resource measurements

Batch 1, exact stored sequence length/grid, FP32, no autocast, TF32 disabled,
10 warm-up and 40 repeated timing samples on up to eight fixed event inputs.
Clean reinference must match stored hourly Hs MAE before measurement proceeds.
Record device identity, PyTorch/CUDA version, allocated and reserved peak GiB,
forward CUDA-event time and cached host-to-host time. The latter includes input
copy/transfer, output transfer and Hs denormalization, but excludes disk reads,
feature construction, startup and writing. These are **inference** peaks, not
training peaks. Do not substitute nvidia-smi occupancy for either peak statistic.
No SWAN speedup is claimed without a separately matched SWAN timing baseline.

`resource_accuracy.png/pdf` plots annual Hs MAE against inference resources;
`robustness_<event>.png` plots paired MAE increases for each perturbation.
`resource_high_wave.png` plots sampled event Hs>=5 m MAE. Each point is one seed.
These describe cost–accuracy associations, not the effect of memory on accuracy.
Compare only identical device/precision/input shapes. Concurrent host I/O can
influence cached pipeline timings; a final timing table merits a quiet repeat.
Training peak memory (including optimizer/EMA/accumulation) is not measured here;
that requires instrumentation of the actual training recipe, not a dummy loss.

## Robustness interpretation

Sixteen conditions: clean; wind vector scale ±5/10%; boundary Hs scale ±5/10%;
boundary direction ±5/10 degrees; boundary delay 1/3/6 hours. Perturbations are
applied one variable group at a time, in physical units then renormalized using
saved training ranges. Direction rotation acts on the already transformed
model-coordinate sin/cos vector. Missing boundary cells remain unchanged for
scaling/rotation. Delay replaces all four boundary channels with past values
only; the whole input and delayed context must be continuous. No clipping.

Unchanged SWAN truth measures robustness to erroneous inputs, NOT physical
correctness under altered forcing. Thresholds are stress levels, not measured
ERA5 uncertainty. Saved output includes clean/perturbed MAE, absolute MAE
increase, bias/RMSE, high-wave sufficient statistics and peak quantities.
Default stride 6 is exploratory sampling, not the full-event score. For every
hour use `--stride 1 --output <new-root>`. No significance test treats hours or
pixels as independent; 2021 results are never used to select a new checkpoint.

## Optional learning-rate pilots

Append `--lr-pilots` to the FIRST launch with a new output root. Seven completed
baseline seed-42 entries yield up to 21 fits: 5e-5, 1e-4 and 2e-4, seed 42,
7,695 successful updates, validation every 2,565. The baseline rate is rerun so
all three have the same short OneCycle schedule. This is deliberately NOT the
first 7,695 updates of the original 76,950-update schedule. Architecture,
normalization/data configuration and loss stay as stored; outputs are isolated
under `<output>/pilot_training`. Full trained models are not replaced.

Use only `pilot_validation_only.csv` for pilot assessment. The underlying
unchanged trainer may also write internal-test metrics; this tool never uses
them for selection and does not evaluate the new pilots on 2021/2022. A weak
short pilot cannot establish a slow model's final ranking. Final confirmation
would require a separately approved equal-budget multi-seed experiment with
untouched 2022 evaluation. To omit slow baseline pilots use `--models` to name a
smaller model set for the entire run. More GPUs do not guarantee pilot completion
before the launch window closes.

## Outputs and checks

`plan.json`, `status.json`, per-stage logs/provenance, `resources.csv`,
`robustness.csv`, resource figures, and optional `pilot_validation_only.csv`.
Reports also build from completed tasks on an interrupted run:

```bash
python3 swan_resource_checks/report.py --root /home/jovyan/swan/runs/resource_checks_2_1_1
python3 -m unittest discover -s swan_resource_checks -p 'test_*.py'
```

Bundled `_diagnostics` files are an unmodified compatibility snapshot of the
existing diagnostics utility. They preserve its per-hour validation and saved
array format. `PACKAGE_SHA256.json` records the delivered utility sources.
Local CPU tests check physical-unit perturbations, causal context rejection,
wet-mask metrics, protocol freezing and GPU scheduling. They do not substitute
for executing CUDA/model restoration on the server.
