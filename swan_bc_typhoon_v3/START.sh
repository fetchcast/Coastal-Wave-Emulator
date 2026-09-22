#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export MPLBACKEND=Agg
exec python3 -u run_campaign.py --apply "$@"
