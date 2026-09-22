#!/usr/bin/env bash
set -euo pipefail
V5_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$V5_DIR/PREPARE.sh"
exec python3 -u "$V5_DIR/train.py"
