#!/bin/bash

# Exit immediately if any command fails
set -e

# Optional: better error handling for pipes
set -o pipefail

cd "$(dirname "$0")"

echo "Setting up the file structure..."
python3 setup_project.py

echo "Installing Requirements..."
pip3 install -r requirements.txt

echo "Verifying Requirements Installation..."
python3 verify_installation.py

echo "Starting pipeline..."

# Phase 1
python3 run_phase1ab.py
printf '\a'

python3 run_phase1c.py
printf '\a'

python3 run_phase1d.py
printf '\a'

# Phase 2
python3 run_phase2a.py
printf '\a'

python3 run_phase2b.py
printf '\a'

python3 run_phase2c.py
printf '\a'

python3 run_phase2d.py
printf '\a'

python3 run_phase2e.py
printf '\a'

# Phase 3
python3 run_phase3a.py
printf '\a'

python3 run_phase3b.py
printf '\a'

python3 run_phase3c.py
printf '\a'

# Phase 4
python3 run_phase4.py
printf '\a'

# Phase 6+
python3 run_phase6.py
printf '\a'

python3 run_phase7.py
printf '\a'

python3 run_phase8.py
printf '\a'

python3 run_phase9.py
printf '\a'

python3 run_phase10.py
printf '\a'

python3 run_phase11.py
printf '\a'

echo "Pipeline completed!"