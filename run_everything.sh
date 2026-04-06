#!/bin/zsh

cd "${0:A:h}"

speak() {
  # Uses macOS built-in text-to-speech
  say "$1"
}

beep() {
  # Simple terminal bell
  printf '\a'
}

run() {
  local message="$1"
  shift

  if [[ -n "$message" ]]; then
    echo "$message"
  fi

  "$@"
  local status=$?

  if [[ $status -ne 0 ]]; then
    echo
    echo "ERROR: Step failed - stopping pipeline."
    beep
    speak "Error occurred. Pipeline stopped."
    exit 1
  fi
}

echo "Setting up the file structure..."
run "Setting up the file structure..." python3 setup_project.py
run "Installing Requirements..." python3 -m pip install -r requirements.txt
run "Verifying Requirements Installation..." python3 verify_installation.py

echo "Starting pipeline..."

# Phase 1
run "" python3 run_phase1ab.py
beep
run "" python3 run_phase1c.py
beep
run "" python3 run_phase1d.py
beep
speak "Phase 1 complete"

# Phase 2
run "" python3 run_phase2a.py
beep
run "" python3 run_phase2b.py
beep
run "" python3 run_phase2c.py
beep
run "" python3 run_phase2d.py
beep
run "" python3 run_phase2e.py
beep
speak "Phase 2 complete"

# Phase 3
run "" python3 run_phase3a.py
beep
run "" python3 run_phase3b.py
beep
run "" python3 run_phase3c.py
beep
speak "Phase 3 complete"

# Phase 4 (and 5)
run "" python3 run_phase4.py
beep
speak "Phase 4 and 5 complete"

# Phase 6+
run "" python3 run_phase6.py
beep
run "" python3 run_phase7.py
beep
run "" python3 run_phase8.py
beep
run "" python3 run_phase9.py
beep
run "" python3 run_phase10.py
beep
run "" python3 run_phase11.py
beep
speak "Pipeline complete"

echo "Pipeline completed!"

echo
echo "If you need to visualize this in the UI, please hit yes or y."
printf "Launch the UI now? [y/N]: "
read -r answer

case "$answer" in
  y|Y|yes|YES|Yes)
    speak "Launching user interface."
    python3 run_ui.py
    ;;
  *)
    speak "Exiting without launching user interface."
    ;;
esac

printf "\nPress Enter to exit..."
read -r
exit 0
