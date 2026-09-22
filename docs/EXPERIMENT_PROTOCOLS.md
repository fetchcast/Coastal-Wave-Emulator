# Experiment–protocol correspondence

Version labels classify source/protocol families; actual run provenance requires
its saved job, code hashes, data signature, split and checkpoint metadata.
The table records supplied defaults, not an assertion that a server run finished.

| Experiment | Protocol family | Development / evaluation | Selection and repeats | Legacy source / default result root |
|---|---|---|---|---|
| Original 209-job benchmark | v2.0 historical | 2019–2020 internal split | Legacy epoch/early-stop and final-raw evaluation | `train.py` at tag `v2.0-run-20260904`; `runs/v2_focused_all` |
| Repaired pilot | v2.1 | 2019–2020 internal train/validation/test | FNO, FFNO, TNO; seeds 42/43/44 | `swan_repaired_v1`; `runs/repaired_timegap_v1`, `runs/repaired_timegap_extra_v1` |
| A architecture sweep | v2.1 | Same repaired development protocol | w/d/m candidates; seed 42 | `swan_iclr_campaign_v2`; `runs/iclr_expanded_v1/A` |
| B learning-rate sweep | v2.1 | Same repaired development protocol | Two validation-selected A candidates/family, two additional rates; seed 42 | `runs/iclr_expanded_v1/B` |
| C seed repeats | v2.1 | Same repaired development protocol | Candidate frozen by validation Hs MAE; seeds 43/44 plus selected seed 42 | `swan_bc_typhoon_v3`; `runs/iclr_bc_typhoon_v3/C` |
| Seven fixed baselines | v2.1 | Same repaired data/budget recipe | Fixed architecture settings; seeds 42/43/44; not equal HPO effort to FNO/FFNO/TNO | `swan_parallel_v4`, `swan_parallel_v41`; `runs/iclr_parallel_v4` |
| Frozen 2021 evaluation | v2.1 | 2021 excluded from fitting and original candidate selection | Saved normalization, EMA and direction transform; JMA event definitions | `swan_bc_typhoon_v3`, `swan_parallel_v41`; `runs/iclr_typhoon_2021_v3` or `runs/iclr_parallel_v4/evaluation` |
| Event diagnostics | v2.1 exploratory analysis | Saved 2021 predictions or identical-checkpoint reinference | Frozen event windows and wet mask; no retraining | `swan_event_diagnostics_v1`; `runs/iclr_event_diagnostics_v1` |
| Three-year experiment | v2.2 planned | 2019–2021 development, 2022 held out | FNO/FFNO; frozen choices; seeds 42/43/44; fresh split/normalization | Staged `swan_three_year_v5`; consult its frozen plan for actual output root |

Repaired defaults use 76,950 successful optimizer updates and validation every
2,565 updates for the original two-year campaign, best validation-Hs-MAE EMA,
train-only preprocessing and time-gap filtering. Check saved jobs before assuming
these defaults for any run. Three-year development is not literally all frames
used for gradient updates: validation/test partitions remain distinct.

Do not pool legacy final-raw results with repaired best-EMA results. Do not treat
three-year results as a pure data-size ablation: split and normalization also
change. 2021 diagnostics used to design subsequent experiments are exploratory;
reserve untouched 2022 for confirmation. Temporal holdout is not another-sea
generalization. Fixed-baseline comparisons must disclose unequal tuning budgets.

For every manuscript row/figure, retain: experiment ID, result root, saved job,
checkpoint hash, trainer hashes, data/split/normalization signatures, selection
metric, seeds, evaluated time range/mask, metric units/aggregation, and plotting
script hash. The supplied capacity plot states pilot anchors + A, seed 42 only,
B/C excluded; verify its CSV and run manifests before assigning any Git commit.
A source import date or package hash alone cannot prove the code of an older run.
