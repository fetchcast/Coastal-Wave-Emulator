# Direct transition from stage A to C

English | [Korean](README_KO.md)

This historical alternative supports the original v2 launcher started with default paths and no custom arguments. It preserves the trainer and selects each family's best validation Hs MAE from pilot seed 42 plus A. Seeds 43/44 run in a separate C root. B/D/E are omitted; TNO is not forced to a smaller configuration. For the subsequent B/C and per-family evaluation route, see the [campaign guide](../docs/CAMPAIGN_20260922.md).

## Inspect, then apply

Install under `/home/jovyan/swan` (original archive: `SWAN_C_Direct_v1.zip`).

```bash
cd /home/jovyan/swan
python3 swan_c_direct_v1/run_c_direct.py
```

Without `--apply`, this checks source, controller, protocol, pilot results, and the A plan only. Unsupported configurations are rejected.

```bash
nohup python3 -u swan_c_direct_v1/run_c_direct.py --apply > iclr_c_direct_v1.log 2>&1 &
```

Apply sends SIGTERM to the verified v2 parent, waits for child training processes to exit, acquires the original campaign/v2 locks, and resumes unfinished A jobs in their existing root. This is not uninterrupted handover. Updates since the last saved checkpoint can repeat, but checkpoints and completed results are not deleted or reset.

If processes do not exit in 60 seconds, new training does not start. Inspect logs before retrying. Unrelated jobs, unknown processes, and customized launchers are not automatically killed.

GPU numbers may change. Available GPUs handle remaining A jobs and concurrent FNO/FFNO C jobs. TNO C waits for all TNO A candidates. Existing pilot repetitions are reused if the anchor wins. Any active B jobs are stopped during this handover; their results remain but are excluded from selection. The selection pool is uniformly pilot42+A for all families.

## Status and outputs

```bash
python3 /home/jovyan/swan/swan_c_direct_v1/status.py --watch 10
tail -n 40 /home/jovyan/swan/iclr_c_direct_v1.log
```

Status comes from `runs/iclr_c_direct_v1/status.json`. The old `status_v2.py` state stops updating after handover. `C_completed` includes reused pilot repetitions.

| Output | Location |
|---|---|
| Existing A | `runs/iclr_expanded_v1/A` |
| New C | `runs/iclr_c_direct_v1/C` |
| Frozen selections | `selection_fno.json`, `selection_ffno.json`, `selection_tno.json` under the new root |
| Nine selected runs | `selected_9.json`, `results.csv` |
| Completion | `completed.json` |

This changes scheduling only; it does not perform 2021 evaluation or define cyclone events. A subsequent evaluator must read these selections and C paths. Do not restart the original v2 launcher after handover: it can restore the old B/D/E plan. Restart this controller with the same apply command instead.

Select once using the search seed. Do not reselect based on repeated-seed performance. Report actual search scope/cost; this is not an equal-compute benchmark.

## Validation

Four CPU mock tests cover per-family readiness, seed-only configuration changes, concurrent A/C assignment, and frozen-selection rejection. Actual remote GPU handover was not tested in the development environment. Model and preprocessing implementations are unchanged.
