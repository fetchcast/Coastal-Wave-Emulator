# SWAN parallel v4.1 GPU allocation

English | [Korean](README_KO.md)

This update adds GPUs 2/3 to baseline training. Model configurations, learning rates, seeds, update budgets, and evaluation criteria are unchanged. It does not launch v5. See [the v4 guide](../swan_parallel_v4/README.md) for baseline settings, evaluation definitions, outputs, and dependencies.

| GPUs | Assignment |
|---|---|
| 0, 1 | Finish the original TNO B/C lane, then evaluate 2021 |
| 2–7 | Run the existing 21 baseline jobs with up to six concurrent fits |

The output root remains `runs/iclr_parallel_v4`, with existing C paths preserved. Evaluation waits until the original training lane exits. This layout targets the phase when original TNO work is nearly finished; it does not guarantee that every GPU stays busy.

## Handover and startup

Install under `/home/jovyan/swan` (original archive: `SWAN_Parallel_v41_GPU23.zip`).

```bash
cd /home/jovyan/swan
nohup bash swan_parallel_v41/START.sh > iclr_parallel_v41.log 2>&1 &
tail -f iclr_parallel_v41.log
```

The handover sends SIGTERM only to the verified v4 controller and checks that its children exit. If exit cannot be confirmed, new jobs do not start. Incomplete training resumes from the last saved checkpoint; updates after that point may repeat. In-memory state is not transferred live. Completed outputs are reused, while an interrupted evaluation may restart.

Do not launch old START scripts concurrently. v4 source files are not modified. There is no copied replacement trainer: the installed `swan_repaired_v1` is used. New controller provenance is recorded separately in `controller_config_v41.json`.

## Monitor

```bash
python3 /home/jovyan/swan/swan_parallel_v41/status.py
tail -n 40 /home/jovyan/swan/runs/iclr_parallel_v4/logs/baselines.log
nvidia-smi
```

The old v4 status reporter can also read the shared result root. Idle GPUs can reflect a small remaining queue or resource constraints.

## Validation

Existing CPU tests and additional GPU-allocation/evaluation-order tests were run. Real server handover and CUDA execution must be checked on the target machine.

```bash
python3 -m unittest discover -s swan_parallel_v41 -p 'test_*.py' -v
```
