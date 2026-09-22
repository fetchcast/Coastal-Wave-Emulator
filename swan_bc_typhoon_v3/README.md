# SWAN B/C scheduling and held-out cyclone evaluation v3

English | [Korean](README_KO.md)

This package uses completed A results and the existing B plan. It does not change model architectures, losses, data splits, direction calibration, or normalization. Existing `swan_repaired_v1` and `swan_iclr_campaign_v2` installations are required. The subsequent parallel controller is documented in the [campaign guide](../docs/CAMPAIGN_20260922.md).

## Workflow

Reuse the verified 29 A jobs and nine pilots. Preserve the 12 existing B jobs and reuse completed B outputs. Once a family's four B jobs finish, freeze the lowest validation Hs MAE configuration from pilot seed 42, A, and B. Test RMSE, 2021 performance, and training time are not selection scores. TNO 128/6 is not preferred automatically: 256/4 wins if its validation score is lower.

FNO/FFNO C can run alongside unfinished TNO B. C runs seeds 43/44 of the frozen configuration and reuses its existing seed-42 run. If the pilot wins, reuse all three pilot seeds. Compatible existing C outputs are reused or resumed at their original paths; otherwise new C outputs use a separate root.

No new w/d/m or learning-rate candidates are added. D/E are omitted. After all nine selected outputs are verified, the wrapper runs 2021 evaluation and event analysis.

## Install, inspect, and start

Install under `/home/jovyan/swan` (original archive: `SWAN_BC_Typhoon_v3.zip`). Inspection without starting/stopping jobs:

```bash
python3 /home/jovyan/swan/swan_bc_typhoon_v3/run_campaign.py
```

Inspection can download JMA tracks if absent. To apply the handover and start:

```bash
cd /home/jovyan/swan
nohup bash swan_bc_typhoon_v3/START.sh > /home/jovyan/swan/iclr_bc_typhoon_v3.log 2>&1 &
```

Do not manually kill the old launcher first. The wrapper validates source/protocol/results/B plan and obtains JMA input before sending SIGTERM to the verified default v2 launcher. Relative launch paths are resolved through the process working directory. It waits for children to exit and acquires the original locks before resuming B. Unrelated Python processes are not killed.

Handover is not uninterrupted. Resume files are written at evaluation-cycle boundaries, so unsaved updates or interrupted evaluation/save work may repeat. Existing completed results and checkpoints are preserved. Modified installs, customized launcher arguments, or an already active old 2021 evaluator cause handover refusal. Do not bypass hash checks.

```bash
python3 /home/jovyan/swan/swan_bc_typhoon_v3/status.py --watch 10
tail -n 80 /home/jovyan/swan/iclr_bc_typhoon_v3.log
```

Restart with the same START command after the previous wrapper exits. Frozen plans/source/protocols cannot be mixed with changed settings. Do not restart the old v2 START command. Training permits up to eight single-GPU jobs; evaluation defaults to two workers, configurable at startup with `--eval-workers 4`. Some GPUs can be idle when only a few jobs remain.

## Frozen held-out protocol

Default sources are `wavm-Waves_2019_2020_v2.nc`, `swan_2021_nc_v2/wavm-Waves.nc`, and `bnd_2021_v2` under the server root. Apply saved 2019-2020 normalization/direction settings without fitting or reselection on 2021. Evaluate 8,748 hours from January 1 12:00 through December 31 23:00; the first 12 hours supply context, and the 2022 endpoint is excluded. Validate time spacing, boundary coverage, and grid agreement. No extra simulation spin-up period is excluded.

Matching preprocessing settings share approximately 33 GB of cache plus working space. Incompatible normalization settings are rejected, not mixed. Rules in `evaluation_protocol.json` are frozen in the result directory before evaluation.

## Cyclone and high-wave definitions

Use official JMA RSMC Tokyo best tracks. Interpolate positions hourly and retain the preceding observed grade between records. Define each event from the first through last TS-or-higher center at most 400 km from a wet-grid center, padded by 24 hours on each side and clipped to available evaluation times.

Named tropical cyclones are included. Storms that reached JMA grade 5 at least once are also summarized separately. Event tables can overlap, but union metrics count each hour once. `outside_TC_windows` does not mean calm conditions. Track proximity is an evaluation-window rule, not attribution of every wave to that storm or a complete inventory of remote swell/extratropical events.

High-wave errors use truth Hs >=3 m and >=5 m at wet cell-hours. Empty subsets remain missing, not zero. Report three-seed means and sample SD (`ddof=1`); seed SD is not an event-population uncertainty interval. No significance test treats correlated hours as independent observations.

JMA input is downloaded once and its SHA256 is frozen. If unavailable online, extract the text from the [official archive](https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/Besttracks/bst_all.zip) as `jma_besttrack.txt` in the package directory. See the [format specification](https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/Besttracks/e_format_bst.html). Download failure does not terminate existing training; no storm list is invented before the source is available.

## Outputs

Training root: `runs/iclr_bc_typhoon_v3`.

- `selection_{fno,ffno,tno}.json`: frozen configurations.
- `candidate_comparison_*.json`: validation scores, settings, and recorded times.
- `selected_9.json`, `results.csv`: the nine selected runs.
- `search/results.csv`: pilot/A/B results; `C/`: new repetitions.

Evaluation root: `runs/iclr_typhoon_2021_v3`.

- `events_2021.json`, `selected_2021.json`: frozen event and checkpoint/preprocessing provenance.
- `summary_2021.csv`: annual metrics for nine runs.
- `event_metrics_by_seed.csv`, `event_metrics_seed_summary.csv`: annual, event-union, outside-event, and individual-event metrics and seed summaries.
- `models/*/hourly.csv`, `result.json`: hourly and annual/monthly metrics, runtime, and memory.
- `tm_quality_flags.csv`: truth Tm >30 s and negative truth/prediction flags.
- `models/*/snapshots/*.npz`: physical-unit maps at each event's truth maximum-Hs time, all seeds.
- `figures/`: event maximum-Hs/error time series and seed-42 Hs/Tm/direction maps.

Hs is in m, Tm in s, and direction errors are minimum circular differences in degrees. Domain prediction and truth maxima can occur at different locations, so prediction at the actual truth-peak location/time is recorded separately. Time series include all three seeds. Truth extremes are not removed.

The Tm 30 s threshold is a QC flag, not an automatic exclusion/clipping rule. This code does not claim to fix the historical 2019-2020 Tm source issue at t=531.

## Validation

```bash
python3 -m unittest discover -s swan_bc_typhoon_v3 -p 'test_*.py' -v
```

Sixteen CPU tests cover B/C concurrency, reuse/restart order, seed settings, relative process paths, JMA parsing/proximity, boundary interpolation, year endpoints, high-wave masks, and circular errors. Synthetic annual inputs also exercised overlapping-event aggregation and figure/table generation. Actual B200 training, full server I/O, JMA downloads, and remote handover were not all exercised in development. Use the existing PyTorch/NumPy/pandas/xarray/SciPy/Matplotlib/NetCDF environment; no automatic installation occurs.
