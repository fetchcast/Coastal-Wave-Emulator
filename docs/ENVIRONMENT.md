# Environment setup

Use a separate environment for CPU package tests. Do not replace PyTorch or
NumPy in an environment running training. These are minimum requirements,
not a lockfile or a claim that every version combination was tested.

```bash
python3 -m venv .venv-tests
source .venv-tests/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
export MPLBACKEND=Agg
python -m unittest discover -s swan_bc_typhoon_v3 -p 'test_*.py'
```

Run each flat-layout package's tests in a separate Python process to prevent
collisions between modules named `campaign`, `events`, and `evaluate_2021`.
The repository-root `tests/` uses `python -m pytest tests`. The repaired
self-test additionally imports xarray. Event geometry requires scipy;
plotting requires matplotlib. An import failure means dependencies are missing,
not that the scientific test passed or failed on its assertions.

GPU runs require a CUDA-enabled PyTorch build compatible with the installed
GPU/driver, plus the existing server dependencies (including tqdm and wavespectra
for training/boundary preparation). Root `requirements.txt` is the older demo's
requirements; it is not a complete campaign or test environment specification.
Record `python --version`, `python -m pip freeze`, `nvidia-smi`, PyTorch/CUDA
versions, precision and device identity with each resource comparison.

The v2.1.1 readiness-test change also updates the test hash in v4/v41 source
manifests and their package manifests. Keep that set of files from one checkout;
do not copy individual manifests into a running server. Runtime trainer and
controller sources were not changed by this maintenance fix.
