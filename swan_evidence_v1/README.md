# SWAN evidence analysis v1

English | [Korean](README_KO.md)

A separate reader for `swan_bc_typhoon_v3` results. It does not change training, candidate selection, evaluation definitions, or existing results. An existing output directory is rejected instead of overwritten. CPU analysis requires Python 3.10+ and NumPy; GPU timing uses the installed SWAN training environment.

This version expects the original v3 nine-run directory layout. Do not point it directly at the v4 per-family tree. See the [campaign guide](../docs/CAMPAIGN_20260922.md).

## Scope

| Analysis | Input and interpretation |
|---|---|
| Seed comparisons | Completed 2021 evaluation for nine frozen runs; paired sign-randomization tests with Holm correction, reported as exploratory |
| Cyclone performance | Frozen v3 JMA windows and hourly metrics; annual, individual-event, union, lifetime-typhoon-union, and outside-event scopes |
| Temporal dependence | Annual Hs MAE differences with 24/72/168-hour moving-block bootstrap sensitivity intervals, conditional on these trained models and year |
| Geographic transfer | Requires completed external-region results and metadata; otherwise NOT_EVALUATED |
| Inference latency | Prepared v3 signature/cache/checkpoint on a free GPU; FP32 batch-1 forward and cached CPU-input-to-CPU-output timing |
| SWAN speed ratio | Measured solver-core wall time for the same physical task; a limited core/cached comparison, not full-pipeline acceleration |

The package does not preprocess a new region, adapt the model grid, or execute SWAN. Check coordinates, bathymetry, boundary/forcing semantics, targets, and model compatibility before external-region evaluation. 2021 represents temporal holdout in the original domain, not geographic transfer.

## CPU analysis

Install under `/home/jovyan/swan` (original archive: `SWAN_Evidence_Analysis_v1.zip`).

```bash
cd /home/jovyan/swan
python3 swan_evidence_v1/analyze.py \
  --eval-root /home/jovyan/swan/runs/iclr_typhoon_2021_v3 \
  --output /home/jovyan/swan/runs/evidence_analysis_v1
```

If evaluation is incomplete, only a WAITING status is written. After completion, rerun into a new directory, such as `evidence_analysis_v1_final`. This command neither waits automatically nor starts GPU evaluation. It requires v3's `analysis_completed.json` and checks all nine complete 8,748-hour series, seeds, truth summaries, and selected configs. Mismatches raise errors.

Outputs are `typhoon_and_annual_summary.csv`, `typhoon_and_annual_by_seed.csv`, `seed_comparisons.csv`, `temporal_block_sensitivity.csv`, `STATUS.json`, and `provenance.json`. Negative A-minus-B error differences favor A.

### Statistical limits

- Three paired seeds give eight sign assignments and a minimum two-sided exact p-value of 0.25. Do not manufacture independent replicates from cells/hours to obtain significance.
- Matching seed numbers define pairs, not guaranteed exchangeability across architectures. Seed 42 also participated in selection, so these comparisons are exploratory.
- Holm correction covers all reported scope/metric/model-pair tests. Nonsignificance does not demonstrate equivalence.
- The block bootstrap resamples the time series of seed-averaged loss differences, not an ensemble prediction. Its interval is not a seed-population or other-year confidence interval. Report sensitivity to 24/72/168-hour blocks; seasonality and long dependence limit nominal coverage claims.
- Events, seeds, and hours are not independent replicated trials. Event seed SD is not an event-population interval.
- `outside_TC_windows` does not mean calm. `typhoon_lifetime_union` covers complete defined windows of storms that reached typhoon grade at some point, not only hours at that grade.
- Domain maxima can move spatially. These diagnostics differ from buoy-specific peaks, and errors relative to SWAN differ from errors relative to ocean observations.

## Inference timing

Run models sequentially on a genuinely free GPU. The example uses physical GPU 0 and refuses execution if it is occupied. Do not stop training to make room. Signatures/caches must already exist from v3 evaluation preparation.

```bash
cd /home/jovyan/swan
for model in fno ffno tno; do
  python3 swan_evidence_v1/benchmark_inference.py \
    --signature "/home/jovyan/swan/runs/iclr_typhoon_2021_v3/models/${model}_s42/signature.json" \
    --gpu 0 \
    --output "/home/jovyan/swan/runs/inference_evidence_v1/${model}.json" || break
done
```

Separate processes release each model's memory. Timing uses eight actual seasonal inputs, ten warmups, 40 repeats, batch 1, FP32, TF32 off, and CUDA synchronization. Preloading the eight CPU arrays requires host memory. Timing uses seed 42 only and does not alter training/evaluation precision or replace three-seed accuracy analysis.

`forward_gpu` uses GPU events and excludes transfers. `cached_host_to_host` includes CPU array copying, H2D, forward, and D2H. Both exclude file reads, forcing/boundary preprocessing, physical-unit restoration, and output saving. Startup is recorded separately. Outputs include median, p10/p90, raw timings, GPU/PyTorch/CUDA details, peak allocation, and checkpoint hash. The occupancy check is instantaneous; do not launch another job during measurement.

```bash
python3 swan_evidence_v1/compare_speed.py \
  /home/jovyan/swan/runs/inference_evidence_v1/fno.json \
  /home/jovyan/swan/runs/inference_evidence_v1/ffno.json \
  /home/jovyan/swan/runs/inference_evidence_v1/tno.json \
  --output /home/jovyan/swan/runs/inference_evidence_v1/comparison.csv
```

For SWAN comparison, copy `swan_baseline.template.json` and enter measured wall time, physical output-frame count, hardware, command, and task specification. `emulator_input_shape` must match the timing JSON. Set `same_domain_forcing_output_contract` true only after verifying that contract. Use comparable output frames, not SWAN integration steps; record CPU cores/MPI ranks.

Adding `--swan /absolute/path/swan_baseline.json` computes solver-core wall seconds divided by output frames times cached-surrogate median seconds. This is an estimated core/cached ratio. Full deployment acceleration requires matching end-to-end wall-time measurements for both systems. Training time is not part of inference acceleration.

## External regions

`regions.template.json` has an empty regions list and is rejected by default. After actual evaluation on a new region, populate entries using `region_example.schema.json`. The metrics CSV requires:

```text
model,seed,scope,hs_mae,checkpoint_sha256
```

Each scope needs FNO/FFNO/TNO at seeds 42/43/44, Hs MAE in meters, and unchanged frozen checkpoint hashes matching the manifest. Exclusions, coordinate resampling, input contracts, and fixed training normalization must be verified and documented by the experimenter; the aggregator does not independently prove absence of leakage. Fine-tuning on the target region is not zero-shot transfer.

```bash
python3 swan_evidence_v1/geography.py \
  --manifest /absolute/path/regions.json \
  --output /home/jovyan/swan/runs/geographic_evidence_v1
```

Results in several regions still do not establish universal geographic generalization.

## Validation and references

```bash
python3 -m unittest discover -s swan_evidence_v1 -p 'test_*.py' -v
```

CPU statistical tests and synthetic nine-run annual integration were checked. Actual B200 timing and full 2021 execution were not performed in development.

- [Permutation tests and exchangeability](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)
- [CUDA execution and timing](https://docs.pytorch.org/docs/stable/notes/cuda.html)
