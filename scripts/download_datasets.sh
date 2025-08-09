#!/usr/bin/env bash
set -euo pipefail

# This script *does not* fetch restricted datasets (e.g., MIMIC-III).
# It only scaffolds directories and optionally grabs open ones if available.

mkdir -p data
echo "Scaffolded data/ directory."

echo "For MIMIC-III: follow PhysioNet access steps, then run scripts/convert_mimic3.py"
echo "For METR-LA: download the raw files and run scripts/convert_metr_la.py"
echo "For Time-MMD: if unavailable, run scripts/convert_time_mmd.py --synthesize"
