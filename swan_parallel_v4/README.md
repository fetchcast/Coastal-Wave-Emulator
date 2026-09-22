# SWAN parallel training and held-out evaluation v4

English | [Korean](README_KO.md)

Evaluate completed FNO/FFNO seed sets on 2021 while training seven additional fixed baselines under the repaired protocol. The existing trainer and A/B/C results are preserved; incomplete B/C jobs resume at the same checkpoint paths. The later [v4.1 scheduler](../swan_parallel_v41/README.md) changes GPU allocation but retains this result layout and baseline plan.

## Scheduling and handover

| GPUs | Work |
|---|---|
| 0, 1 | Remaining original TNO B, unchanged selection, then C seeds 43/44 |
| 2, 3 | FNO/FFNO evaluation first, then other ready families |
| 4, 5, 6, 7 | Seven baseline preflights, then 21 fixed training jobs |

A family enters evaluation once all three seeds are ready. Each family's seeds are evaluated sequentially; two families can run in parallel. Baselines queue seed 42 first, then 43/44. v4 does not reassign idle training GPUs to evaluation, so continuous eight-GPU utilization is not guaranteed.

The verified v3 controller is stopped once to avoid competing schedulers. Training resumes from saved checkpoints and unsaved updates can repeat. This is not live transfer of in-memory state. If v3 has already entered its 2021 evaluation phase, automatic handover is refused. Do not kill arbitrary processes to bypass this check.

## Start and monitor

Install under `/home/jovyan/swan` (original archive: `SWAN_Parallel_Typhoon_Baselines_v4.zip`). Inspect without starting/stopping jobs:

```bash
python3 /home/jovyan/swan/swan_parallel_v4/run.py
```

Apply handover and start:

```bash
cd /home/jovyan/swan
nohup bash swan_parallel_v4/START.sh > /home/jovyan/swan/iclr_parallel_v4.log 2>&1 &
python3 /home/jovyan/swan/swan_parallel_v4/status.py --watch 10
```

No separate kill command or edits to installed source are needed. Changed v3/trainer hashes cause refusal. Do not remove verification.

```bash
tail -n 80 /home/jovyan/swan/iclr_parallel_v4.log
tail -n 40 /home/jovyan/swan/runs/iclr_parallel_v4/logs/original.log
tail -n 40 /home/jovyan/swan/runs/iclr_parallel_v4/logs/baselines.log
tail -n 40 /home/jovyan/swan/runs/iclr_parallel_v4/logs/eval_fno.log
tail -n 40 /home/jovyan/swan/runs/iclr_parallel_v4/logs/eval_ffno.log
```

The old v3 log stops updating after handover. Original B/C outputs keep their paths, but their controller log is now `original.log`. Data/cache preparation can initially leave GPUs idle.

## Fixed baselines

| Model | Configuration |
|---|---|
| ConvNeXt-LSTM | dims 96/192/384, depths 2/2/2, recurrent hidden 256 |
| Conv-Swin | base width 48, attention dim 256, depth 6, heads 8, window 8 |
| UNet-LSTM | features 64/128/256/512/1024, recurrent hidden 768 |
| UNet-FFNO | features 128/256/512/1024/2048, spectral width 256, depth 4, modes 16/16 |
| Swin | embed 72, depths 2/2/6/2, heads 3/6/12/24, patch 4, window 8 |
| ConvLSTM | width 128, depth 2 |
| ViT | embed 384, depth 6, heads 6, patch 16 |

These are fixed baselines, not separately optimized architectures or proven replicas of all earlier runs. Details absent from the supplied manuscript were frozen from implementation defaults. Do not claim equal search budgets with FNO/FFNO/TNO.

Baseline training uses 2019-2020, boundary ON, sequence length 12, max_lr 1e-4, weight decay 1e-4, 30 cycles, disabled early stopping, microbatch 1 and accumulation 4. Original family settings are unchanged. Effective batch/update budget can be compared even though microbatches differ. Update budgets, splits, and direction policies are checked; actual parameter counts are recorded.

Seven two-update preflights precede 21 main fits. The evaluator receives family-specific constructor settings and UNet feature lists. Preflight checkpoint/evaluator state layouts are checked strictly on the meta device. Settings appear in `plans.py` and frozen `baseline_plan.json`. OOM does not silently reduce capacity. Failed baseline preflight stops that lane; original TNO work and eligible evaluations can continue. Inspect `status.json` errors and worker logs.

Historical baseline checkpoints are not automatically reused because matching code, training, splits, and normalization have not been established. Completed jobs from this package are reused on restart.

## Evaluation and outputs

All families share the content-keyed 2021 cache, with a single cache writer. Allow approximately 33 GB plus working space and separate checkpoint storage. Selection for the original families remains minimum validation Hs MAE from pilot42+A+B; 2021 scores are not selection inputs.

Default root: `/home/jovyan/swan/runs/iclr_parallel_v4`.

| Output | Meaning |
|---|---|
| `status.json`, `controller_config.json`, `baseline_plan.json` | Live state and frozen configuration |
| `baselines/preflight/`, `baselines/fixed/` | Baseline preflight/training results |
| `baselines/evaluator_layout_checks.json` | Checkpoint/evaluator compatibility |
| `results.csv` | Original-period metrics for families with three completed seeds; verify resumed-time accounting before treating wall time as total cost |
| `evaluation/<model>/` | Per-family selected entries, event tables, hourly metrics, and figures |
| `available_event_metrics_by_seed.csv` | Combined completed-family metrics |
| `available_event_metrics_seed_summary.csv` | Completed-family means and seed SD |
| `shared_2021/` | Shared preprocessing cache |
| `logs/` | Lane and evaluation logs |
| `completed.json` | All ten families, three seeds each, evaluated |
| `partial_completed.json` | Terminal incomplete/error state, not a live progress file |

Missing families are absent, not scored as zero. 2021 is a held-out year in the same geographic domain, not another-region generalization. Event definitions and thresholds remain those frozen in v3; overlaps count once in union metrics. Outside-event hours are not necessarily calm. Tm anomalies are flagged, not removed/clipped.

Spatial maps use seed 42; numerical summaries include all three seeds. Truth/prediction share a color scale in each figure, but separate family folders can have different scales. Compare CSV values, not color intensity across files.

The earlier `swan_evidence_v1/analyze.py` expects nine runs in one v3 directory. Do not apply it directly to the v4 layout. This controller produces its own accuracy/event summaries, not new ten-model significance tests or measured inference-speed comparisons. It does not rerun the old full v3 evaluator in parallel.

## Restart and validation

Rerun the same START command after resolving the error and confirming the prior controller exited. Completed outputs are reused and incomplete jobs resume. Locks reject duplicate controllers. Initial handover refuses to launch if old jobs have not exited in 90 seconds or unexpected GPU processes remain; unrelated processes are not terminated.

```bash
python3 -m unittest discover -s swan_parallel_v4 -p 'test_*.py' -v
```

CPU tests cover fixed plans, constructor arguments, seed aggregation, incomplete-result exclusion, isolated temporary-process handover, and synthetic 8,748-hour, three-seed non-Fourier event reports/figures. Actual B200 training and full 2021 evaluation require server execution.
