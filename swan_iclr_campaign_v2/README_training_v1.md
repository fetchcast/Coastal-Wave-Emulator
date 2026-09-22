# SWAN expanded campaign v1

English | [Korean](README_training_v1_KO.md)

This document preserves the v1 training workflow used by the v2 wrapper. For v2 startup and evaluation, use [README.md](README.md).

This historical controller uses `swan_repaired_v1` and the nine completed repaired pilot runs. It writes separate campaign outputs and does not replace the trainer or pilot results. Later scheduling and held-out evaluation packages are described in the [campaign guide](../docs/CAMPAIGN_20260922.md).

## Run and monitor

Install the package under `/home/jovyan/swan`, or extract the original `SWAN_ICLR_Campaign_v1.zip` there.

```bash
cd /home/jovyan/swan
nohup bash swan_iclr_campaign_v1/START.sh > iclr_campaign_v1.log 2>&1 &
python3 /home/jovyan/swan/swan_iclr_campaign_v1/status.py --watch 10
tail -n 50 /home/jovyan/swan/iclr_campaign_v1.log
```

One job is assigned to each free GPU among 0–7. Occupied GPUs are awaited. Do not launch another controller against the same GPUs. A package lock prevents duplicate instances. Ctrl+C in the status command stops monitoring only.

```bash
bash /home/jovyan/swan/swan_iclr_campaign_v1/START.sh --plan
```

Plan-only mode neither trains nor validates the server data fingerprint; actual execution performs those checks. After a stop, confirm the old launcher has exited and rerun the launch command. Completed runs are checked and reused. Incomplete runs resume from the original worker's saved cycle checkpoint; unsaved updates may repeat.

## Prerequisites

- `swan_repaired_v1/{run_repaired.py,train_repaired.py,repair_support.py,legacy_repaired.py}`.
- `runs/repaired_timegap_v1/plan_pilot.json` and `protocol.json`.
- `runs/repaired_timegap_extra_v1/extra_plan.json`.
- The nine existing `run_summary.json` files and best checkpoints.
- The original NetCDF, boundary inputs, and other training assets.

`expected_hashes.json` records the supplied trainer SHA256 values. Source/data mismatches stop execution. Do not edit hashes to bypass verification. The existing Python environment is used without automatic package installation.

## Stages

| Stage | Work |
|---|---|
| Preflight | 29 architecture candidates, two successful updates each, followed by the original pilot-stage train/validation/final evaluation |
| A | 29 new architecture configurations, seed 42, 30 full-data-equivalent cycles |
| B | Two best validation configurations per family, each at max_lr 5e-5 and 2e-4; at most 12 new fits |
| C | Freeze the best seed-42 configuration per family from pilot/A/B, then run seeds 43/44; at most six new fits |
| D | Selected configurations, seed 42, 60 cycles; three newly initialized fits |
| E | Selected configurations, train fractions 0.25/0.5, seeds 42/43/44; 18 fits at 30 full-data-equivalent cycles |

A uses widths 64/128/256 and depths 4/6/8 at spatial modes 24. For FNO/FFNO, exclude the existing 64/4/24 anchor and add width 64/128/256, depth 4, modes 48: 11 candidates per family. For TNO, exclude the anchor and width-256 depths 6/8, then add 64/4/48: seven candidates. TNO temporal modes remain 4 and new TNO architecture jobs use activation checkpointing.

The maximum is 68 new full training runs plus 29 preflight jobs. Existing pilots are reused. Preflight scores are excluded from selection and `results.csv`. Full-data loading/evaluation makes preflight more expensive than a tiny smoke test, and preflight success does not guarantee freedom from later OOM.

The original 1e-4 learning-rate runs remain eligible in final selection. If the pilot anchor wins, its completed seeds 43/44 are reused. D also changes the OneCycleLR horizon and validation opportunities, so it does not isolate extra updates alone. E repeats training seeds, not independent subset-draw seeds. There is no deadline-based automatic shutdown.

## Interpretation and failure handling

Selection uses only `repair_audit.best_val_hs_mae`, not test RMSE. Workers still perform final test evaluation, but the controller does not use that metric to choose candidates. New architecture jobs target nominal effective batch 4; a final incomplete accumulation group can be smaller. Selected pilot configurations retain their batch/checkpointing settings.

This is not exhaustive search or an equal-parameter/equal-GPU-hour comparison. FNO/FFNO/TNO have different search counts. GPU memory is not pooled across eight devices. TNO width 256, depth 8 is excluded. The package does not add new split studies, seven-year data, or manuscript figures.

Explicit CUDA OOM and a memory lower bound above 85% of the GPU budget produce `*.resource_skip.json`; remaining jobs continue. Capacity and batch settings are not silently reduced. Unknown errors, data mismatches, or split mismatches stop the campaign. Successful outputs remain. Resource-skip records are reused on restart; use a new output root for changed settings. `campaign_completed.json` means all stages were processed, not that every candidate succeeded.

## Outputs

Default root: `/home/jovyan/swan/runs/iclr_expanded_v1`.

- `status.json`: stage, active GPUs, queue. Check timestamps and logs after interruption.
- `results.csv`: original pilots and completed main jobs, metrics, parameters, recorded time, and checkpoints.
- `selection.json`: validation-selected configurations and job definitions.
- `preflight/`, `A/`, `B/`, `C/`, `D/`, `E/`: plans, worker outputs, and resource skips.
- `campaign_completed.json`: stage-processing completion.

Advanced overrides are `START.sh --server-root ... --package ... --data ... --root ... --gpus ...`. Pilot roots retain their fixed names under the server root.

## Validation

```bash
python3 -m unittest discover -s swan_iclr_campaign_v1 -p 'test_*.py' -v
```

CPU mocks cover candidate IDs, preservation of source configurations, selection without test metrics, frozen-plan rejection, OOM isolation, and restart reuse. Actual CUDA training, memory peaks, and eight-GPU operation were not tested in the development environment.
