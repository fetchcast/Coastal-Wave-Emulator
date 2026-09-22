#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python3 -u selftest.py
exec python3 -u run_repaired.py "$@"
