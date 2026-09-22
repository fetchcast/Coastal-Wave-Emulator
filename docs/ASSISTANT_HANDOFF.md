# Cross-assistant handoff

GPT and Claude can work on this repository through separate clients. Git commits and the working tree are the shared state; their conversation histories are not automatically shared.

1. Fetch the remote before editing. Inspect `git status` and read `CLAUDE.md`, `TASKS.md`, and the current campaign guide.
2. Use the latest `v2-benchmark` commit as the starting point. For concurrent work, use separate task branches and pull requests.
3. Preserve unrelated edits. Never force-push, amend a published commit, or rewrite pushed history.
4. Before publishing, fetch again and reconcile any newer commits. A non-fast-forward update must fail rather than overwrite another assistant's work.
5. Record changed files, the reason for changes, validation performed, and outstanding limitations in the commit or PR description.
6. Do not run a training launcher, handover, or evaluation job merely because code was pulled. Check GPU allocations and the active experiment first.
7. Do not modify source files used by active experiments. Preserve source hashes and existing result roots.

## Current handoff, September 22, 2026

- The delivered campaign sources through v4.1 and event diagnostics are included.
- The repaired trainer core is unchanged from the September 12 branch snapshot.
- v4.1 is the latest supplied controller. It continues using the v4 result root.
- v5 is staged code and has not been certified as a completed experiment.
- The event diagnostic package performs 2021 Hs inference/analysis and a historical split-exposure audit. Historical event inference is not implemented there.
- Input robustness tests, GPU memory-accuracy measurements, and new hyperparameter search are proposals, not delivered implementations.
- The final UNet-LSTM run and remote results must be checked on the server; this repository update does not certify their current status.
- No data, results, prediction arrays, or additional model weights were added.

For a sequential handoff on a clean local checkout:

```bash
git fetch origin
git switch v2-benchmark
git pull --ff-only origin v2-benchmark
git log -5 --oneline
```

Do not run these checkout/pull commands over the active training source directory without first checking the running jobs and local modifications. Use a separate checkout for review when jobs are active.
