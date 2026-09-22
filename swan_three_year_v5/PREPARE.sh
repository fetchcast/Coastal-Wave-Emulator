#!/usr/bin/env bash
set -euo pipefail
V5_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 -u "$V5_DIR/freeze_candidates.py"
python3 -u "$V5_DIR/prepare_data.py"
