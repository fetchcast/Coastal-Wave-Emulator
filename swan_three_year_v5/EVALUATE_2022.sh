#!/usr/bin/env bash
set -euo pipefail
V5_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -u "$V5_DIR/evaluate.py"
