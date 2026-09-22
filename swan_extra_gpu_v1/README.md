# Additional GPU pilot runs and status reporting

English | [Korean](README_KO.md)

This historical pilot add-on schedules six additional seed runs on GPUs 3–7. It does not overwrite the Python files in `swan_repaired_v1` or stop/restart the existing jobs on GPUs 0–2. Use it only after the time-gap update, three successful smoke tests, and creation of the pilot plan. For the later campaign, see the [campaign guide](../docs/CAMPAIGN_20260922.md).

## Run

With the package under `/home/jovyan/swan`:

```bash
cd /home/jovyan/swan
nohup bash swan_extra_gpu_v1/START_EXTRA_GPUS.sh > extra_gpu_run.log 2>&1 &
tail -f /home/jovyan/swan/extra_gpu_run.log
python3 /home/jovyan/swan/swan_extra_gpu_v1/status_repaired.py --watch 10
```

If installing from the original archive, first extract `SWAN_Extra_GPU_v1.zip` in that directory. Add `--plan` to `START_EXTRA_GPUS.sh` to inspect the plan without training. Omit `--watch 10` for a single status report. Ctrl+C exits monitoring, not training.

The reporter reads log tails and displays successful optimizer updates and the latest validation cycle's best Hs MAE. The progress-bar ETA may exclude later validation and final evaluation. `NOT COMPLETED` is not a failure verdict: queued, training, and evaluating jobs can all have this status.

## Additional jobs

| Initial GPU | Model | Seed |
|---|---|---|
| 3 | TNO | 43 |
| 4 | FNO | 43 |
| 5 | FFNO | 43 |
| 6 | TNO | 44 |
| 7 | FFNO | 44 |
| First available selected GPU | FNO | 44 |

This order assumes GPUs 3–7 are free. Occupied GPUs are awaited. Six jobs run with at most five active at once; together with the three original seed-42 jobs, there are nine runs. The original pilot plan is copied, so width 64, depth 4, spatial modes 24, and training budget are preserved. Only the training seed and identifying config ID change; the data-split seed stays fixed.

## Outputs and restart

Original results remain in `runs/repaired_timegap_v1`. Additional results go to `runs/repaired_timegap_extra_v1`. The reporter reads both roots.

Rerun the same launch command after an interruption. Verified completed runs are skipped; incomplete runs resume under the original trainer's checkpoint rules. A lock prevents duplicate add-on launchers. Changed source/data fingerprints are rejected. Do not edit `protocol.json` to bypass checks. On failure, only jobs owned by the additional launcher are stopped; the original GPU 0–2 launcher is untouched.

Successful source checks can be reused after matching their fingerprints. Each job still loads and preprocesses data, so initial GPU utilization can be low. Concurrent jobs can compete for CPU, memory bandwidth, and storage. This package adds neither large-model search nor 2021 evaluation.

## Validation scope

Mock checks covered seed-only changes, preservation of the original plan, failed-smoke rejection, cached source checks, rejection of GPU 0–2 assignments, launch paths, and progress parsing. Five-GPU training was not performed in the package-development environment.
