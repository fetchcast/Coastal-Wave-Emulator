# SWAN campaign v2 with held-out 2021 evaluation

English | [Korean](README_KO.md)

This historical wrapper preserves the v1 architecture/training plan and adds independent 2021 evaluation. For the later B/C-only and per-family evaluation workflow, see the [campaign guide](../docs/CAMPAIGN_20260922.md). Detailed v1 training stages are documented in [README_training_v1.md](README_training_v1.md).

## Run and restart

Install under `/home/jovyan/swan` (original archive: `SWAN_ICLR_Campaign_v2.zip`).

```bash
cd /home/jovyan/swan
nohup bash swan_iclr_campaign_v2/START.sh > iclr_campaign_v2.log 2>&1 &
python3 /home/jovyan/swan/swan_iclr_campaign_v2/status_v2.py --watch 10
tail -n 50 /home/jovyan/swan/iclr_campaign_v2.log
```

If v1 is active, v2 waits and reports every 30 seconds; it does not force v1 to exit. After v1 exits, it verifies completed outputs and resumes incomplete work. Both use `runs/iclr_expanded_v1`; the original plan and `campaign.py` are preserved. New evaluation results go to `runs/iclr_2021_v2`. Existing v1 Python files and data are not overwritten.

Ctrl+C in monitoring does not stop training/evaluation. Restart with the same launch command. Locks reject duplicate wrappers. Valid saved hourly evaluation rows are skipped. A malformed CSV tail causes an explicit error: preserve/move that CSV before rerunning the affected evaluation.

## Inputs and model selection

Default inputs under `/home/jovyan/swan`:

- `swan_repaired_v1/` and `wavm-Waves_2019_2020_v2.nc`.
- `swan_2021_nc_v2/wavm-Waves.nc` and `bnd_2021_v2/`.
- The original server `bnd_features.py` and `boundspec_segments.py`.

The supplied inspection found matching grids/masks and all required variables. Its 8,761 records span 2021-01-01 00:00 through 2022-01-01 00:00. Runtime checks repeat the grid/time validation.

The v1 plan retains at most 68 additional main fits and 29 preflight jobs, reusing nine pilots. Only the three seeds of each validation-selected configuration are evaluated in 2021. Selection uses 2019-2020 validation Hs MAE; 2021 is not used for training, learning-rate selection, or checkpoint selection. D/E results do not replace the final configuration. Reselecting after viewing 2021 performance invalidates an unchanged independent-test interpretation.

## Evaluation contract

- Preserve sequence length 12 and next-time target semantics.
- Exclude the 2022-01-01 endpoint. Use all 8,760 calendar-year frames, with the first 12 as context only and no 2020 carry-in. Evaluate 8,748 targets, January 1 12:00 through December 31 23:00.
- Read saved `normalization.json`; do not estimate new ranges from 2021.
- Apply the saved direction manifest's chosen transform to boundary sin/cos. Do not choose a transform against held-out direction targets.
- Preserve the native SWAN target convention, original first-frame depth-gradient feature, EMA wrapper state loaded with `strict=True`, FP32/no-autocast batch-1 inference, and uniform `kcs > 0` weights.
- Reject nonfinite predictions/inputs instead of silently excluding samples. No additional simulation spin-up exclusion is applied.

The original `SEGMENTS` mapping determines which boundary names are used even if 37 files are present. Unused names are recorded in cache metadata. Validate finite values, time coverage, and spacing for each used segment. Interpolate on the union of original boundary observations and requested times; do not extrapolate or fill with zero. A 2022 boundary record may bracket the last 2021 interpolation without becoming an evaluation target. The maximum permitted source gap is six hours and is not automatically relaxed.

## Cache and concurrency

Preprocessing reads 16-frame chunks into memory-mapped inputs/targets. Matching normalization, direction, and source fingerprints share a cache across models; different settings require separate caches. Each cache needs approximately 33 decimal GB plus working space. Original NetCDF files are not merged or edited.

Preparation is sequential on CPU. Evaluation defaults to two workers on available GPUs among 0–7 to limit I/O contention. Do not run another launcher against those GPUs. To choose four workers at initial launch:

```bash
nohup bash /home/jovyan/swan/swan_iclr_campaign_v2/START.sh --eval-workers 4 > /home/jovyan/swan/iclr_campaign_v2.log 2>&1 &
```

Do not launch this while another v2 wrapper is active. `--eval-only` skips training but requires all nine selected checkpoints.

## Outputs

Under `runs/iclr_2021_v2`:

- `selected_2021.json`: frozen entries, normalization sources, and direction transforms.
- `summary_2021.csv`: annual model/seed metrics.
- `models/{model}_s{seed}/result.json`: annual/monthly metrics, coverage, memory, and provenance.
- `models/{model}_s{seed}/hourly.csv` and `evaluate.log`: hourly metrics and progress.
- `cache/{signature}/complete.json`: source/time/boundary checks and unused segments.
- `completed.json`: all nine evaluations complete.

For example:

```bash
tail -n 30 /home/jovyan/swan/runs/iclr_2021_v2/models/fno_s42/evaluate.log
```

`hs_mean_frame_rmse` averages the spatial RMSE of each frame, matching legacy `rmse_m` aggregation. `hs_pooled_rmse` takes the square root after pooling squared errors over time and space. Keep these distinct; the same distinction applies to Tm. Direction uses minimum circular angular differences in degrees, with a separate fraction of predicted direction vectors having radius below 1e-6.

Monthly/hourly outputs support later event analysis, but this version does not automatically define events or draw event figures. Recorded evaluation wall time includes I/O and is not an inference-speed benchmark.

Training handles explicit CUDA OOM as in v1. Any held-out evaluation failure stops the wrapper because all nine results are required. Source/plan mismatches also stop execution; never erase fingerprints to bypass them.

## Validation

```bash
python3 -m unittest discover -s swan_iclr_campaign_v2 -p 'test_*.py' -v
```

Ten CPU mock tests cover plans, selection, restart/isolation, year endpoints, sequence context, missing hours, saved direction transforms, interpolation/extrapolation handling, and physical/circular metrics. Full evaluation on the actual server NetCDF and GPU was not performed in the development environment.
