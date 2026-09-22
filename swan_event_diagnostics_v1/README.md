# SWAN event diagnostics v1

English | [Korean](README_KO.md)

Standalone diagnostics for completed v4/v4.1 evaluations. Existing training processes, checkpoints, selection, normalization, and training/evaluation outputs are not modified.

## Scope and saved arrays

Defaults are FNO/FFNO/TNO, seeds 42/43/44, and the 2021 Lupit, Omais, and Chanthu windows. Read the frozen entries in `selected_2021.json` and windows in `events_2021.json`; do not reselect or retrain models.

The original evaluator saved a spatial snapshot only at each event's truth maximum-Hs time. This package validates and reuses those snapshots and re-infers missing event hours from the prepared cache and frozen checkpoint. `targets.npy` contains normalized truth, not predictions. `hourly.csv` does not contain enough information to reconstruct spatial fields.

New arrays contain Hs in meters only. The package does not fix Tm quality problems or reevaluate direction fields. It also audits saved 2019-2020 split exposure, but historical-event re-inference is not implemented. Do not label the entire training period as an independent test.

## Run

Install under `/home/jovyan/swan` (original archive: `SWAN_Event_Diagnostics_v1.zip`).

```bash
cd /home/jovyan/swan
python3 swan_event_diagnostics_v1/run.py inspect
nohup bash swan_event_diagnostics_v1/START.sh --gpus 2,3 > /home/jovyan/swan/iclr_event_diagnostics_v1.log 2>&1 &
tail -f /home/jovyan/swan/iclr_event_diagnostics_v1.log
```

Default GPUs are 2/3. An existing compute process on a selected GPU causes refusal, not termination. This package does not negotiate reservations with other controllers. Do not allocate the same GPUs to another campaign. GPU 4 is not used by the default command. Ctrl+C in `tail -f` stops monitoring only.

Default output: `/home/jovyan/swan/runs/iclr_event_diagnostics_v1`. Missing original evaluation caches cause an error; this package does not regenerate the entire annual cache automatically.

```bash
cat /home/jovyan/swan/runs/iclr_event_diagnostics_v1/status.json
tail -n 30 /home/jovyan/swan/runs/iclr_event_diagnostics_v1/logs/tno_s42.log
nvidia-smi
```

Initial checkpoint SHA256 calculation can delay startup. The default raw prediction/truth arrays total about 1.5 GB before compression; allow at least 3 GB for figures and working space. Runtime depends on actual server inference speed.

Restart with the same command to reuse valid saved frames under matching provenance. Corrupt frames or changed checkpoints/normalization are rejected. Use a new `--output` root if code, models, or event lists change. Do not run duplicate controllers.

To analyze all nine completed families, use the following instead of the three-family launch; do not run both on the same GPUs:

```bash
nohup bash swan_event_diagnostics_v1/START.sh \
  --gpus 2,3 \
  --models fno,ffno,tno,conv_swin,swin,convnext_lstm,convlstm,u_ffno,vit \
  --output /home/jovyan/swan/runs/iclr_event_diagnostics_all9_v1 \
  > /home/jovyan/swan/iclr_event_diagnostics_all9_v1.log 2>&1 &
```

## Outputs and definitions

| File | Content |
|---|---|
| `all_event_metrics.csv`, `summary_event_metrics.csv` | Event MAE/RMSE/bias, prediction at the truth peak location/time, peak timing differences; summaries include seed mean, sample SD, and n_seeds |
| `all_wave_bins.csv`, `summary_wave_bins.csv` | Truth-Hs-bin errors, counts, sample fractions, and contributions to total event MAE |
| `common_bin_standardized_mae.csv` | Exploratory comparison with common weights over bins populated in every selected event |
| `all_detection.csv`, `summary_detection.csv` | TP/FP/FN/TN, recall, precision, false-alarm ratio, false-positive rate, and CSI at 3/5 m |
| `all_phases.csv` | Errors before, during, and after a truth-peak-centered +/-12-hour window |
| `models/{model}_s{seed}/events/{id}/hourly_diagnostics.csv` | Spatial errors and time series at the fixed truth-peak grid cell |
| `time_series.png`, `spatial_comparison.png` in each event folder | Seed-42 figures by default; use `--plot-all-seeds` for all seeds |
| `spatial_errors.npz` | Event-mean spatial MAE and bias |
| `frames/{target_index}.npz` | `pred_hs`, `true_hs`, `time`, `units`; use the event's `mask.npy` |

Bins are negative Hs, 0–1, 1–2, 2–3, 3–4, 4–5, 5–6, and >=6 m, including each left boundary and excluding the right. The standardized comparison uses pooled sample weights only on shared bin support. Read `retained_sample_fraction` and `common_bins` alongside its score. It is not annual performance, causal attribution, or a replacement for raw event MAE.

Both truth and prediction use >=3 m and >=5 m thresholds. Undefined ratios have missing values, not zeros. Peak-centered phases do not prove physical growth/decay stages. Numerical outputs include all requested seeds even if only seed-42 figures are drawn.

Only True cells in `mask.npy` are evaluated. Do not interpret land values. Times retain the original evaluation's UTC convention. Spatial axes are grid rows/columns, not latitude/longitude or coastal-distance classes. A predicted domain maximum may occur elsewhere; distinguish it from the fixed truth-peak-cell series. Window-edge peaks can limit interpretation of timing differences.

Every frame's Hs MAE is compared against original `hourly.csv`, with absolute tolerance 2e-6 m and relative tolerance 2e-4 for float32/unit-conversion rounding. Larger differences stop execution. Agreement of a scalar metric does not prove every field value is identical.

These diagnostics were proposed after viewing 2021 results and are exploratory. If models are reselected or tuned using them, 2021 cannot remain an unchanged final independent test. No significance tests treat correlated cells/hours as independent samples.

## Historical split-exposure audit

Run on CPU:

```bash
python3 swan_event_diagnostics_v1/audit_training_period.py \
  --output /home/jovyan/swan/runs/historical_exposure_audit_v1
```

Saved split indices are input starts, so `seq_length` is added to locate targets. Outputs `{model}_s{seed}_timeline.csv` report train/val/test/excluded target membership, input/target overlap with training, and time continuity. Membership in the training split is not the number of times the sampler used that record. Forcing-input exposure is distinguished from target-wave exposure.

To summarize historical events, populate [historical_events.template.csv](historical_events.template.csv) with `id,name,start,end`, using official tracks or an existing fixed rule. Times use UTC `YYYY-MM-DDTHH:00:00`, with the end included. No arbitrary example dates are supplied.

```bash
python3 swan_event_diagnostics_v1/audit_training_period.py \
  --events-csv /home/jovyan/swan/historical_events.csv \
  --output /home/jovyan/swan/runs/historical_event_exposure_v1
```

This is preparation for historical diagnosis, not historical inference. Do not mix historical preprocessing/boundaries into the 2021 cache or reuse that cache for another year.

## Collect a review archive

```bash
python3 swan_event_diagnostics_v1/collect_report.py \
  --output /home/jovyan/swan/SWAN_Event_Diagnostics_Report_v1.zip
```

Only CSV/JSON/PNG/logs are included, not large arrays or checkpoints. For the all-nine run, add `--root /home/jovyan/swan/runs/iclr_event_diagnostics_all9_v1`.

## Dependencies and validation

Use existing Python, NumPy, pandas, PyTorch, and Matplotlib. The historical audit also requires xarray. No dependencies are installed or upgraded automatically.

```bash
python3 -m unittest discover -s swan_event_diagnostics_v1 -p test_diagnostics.py -v
```

Synthetic checks cover reconstruction of MAE from bin contributions, threshold confusion counts, truth-peak values, missing SD for one seed, split target offsets, time-gap rejection, and figures. Full-checkpoint B200 inference was not run in development. `evaluate_2021.py` and `campaign.py` are copied v4.1 helper modules; existing server files are untouched.
