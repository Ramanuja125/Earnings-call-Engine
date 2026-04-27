#!/bin/bash

# =========================
# Setup
# =========================
run() {
    if [ -n "$1" ]; then
        echo "$1"
    fi

    shift
    "$@"

    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Step failed - stopping pipeline."
        printf '\a'
        speak "Error occurred. Pipeline stopped."
        read -p "Press Enter to continue..."
        exit 1
    fi
}

speak() {
    say "$1"
}

cd "$(dirname "$0")"

run "Setting up the file structure..." python3 setup_project.py
run "Installing Requirements..." pip3 install -r requirements.txt
run "Verifying Requirements Installation..." python3 verify_installation.py

echo "Starting pipeline..."

# =========================
# Phase 1
# =========================
run "" python3 run_phase1ab.py; printf '\a'
run "" python3 verify_phase1ab.py; printf '\a'
run "" python3 run_phase1c.py; printf '\a'
run "" python3 verify_phase1c.py; printf '\a'
run "" python3 run_phase1d.py; printf '\a'
run "" python3 verify_phase1d.py; printf '\a'

speak "Phase 1 complete"

# =========================
# Phase 2
# =========================
run "" python3 run_phase2a.py; printf '\a'
run "" python3 verify_phase2a.py; printf '\a'
run "" python3 run_phase2b.py; printf '\a'
run "" python3 run_phase2c.py; printf '\a'
run "" python3 verify_phase2c.py; printf '\a'
run "" python3 run_phase2d.py; printf '\a'
run "" python3 verify_phase2d.py; printf '\a'
run "" python3 run_phase2e.py; printf '\a'

speak "Phase 2 complete"

# =========================
# Phase 3
# =========================
run "" python3 run_phase3a.py; printf '\a'
run "" python3 verify_phase3a.py; printf '\a'
run "" python3 run_phase3b.py; printf '\a'
run "" python3 verify_phase3b.py; printf '\a'
run "" python3 run_phase3c.py; printf '\a'
run "" python3 verify_phase3c.py; printf '\a'

speak "Phase 3 complete"

# =========================
# Phase 4 (includes 5)
# =========================
run "" python3 run_phase4.py; printf '\a'
run "" python3 verify_phase4.py; printf '\a'

speak "Phase 4 and 5 complete"

# =========================
# Phase 6+
# =========================
for i in {6..11}
do
    run "" python3 run_phase${i}.py
    printf '\a'
    speak "Phase $i complete"
done

speak "Pipeline complete"

# =========================
# UI Prompt
# =========================
echo ""
read -p "Do you want to launch the UI (run_ui.py)? (y/n): " choice

if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    echo "Launching UI..."
    speak "Launching user interface"
    streamlit run app.py
else
    echo "Skipping UI..."
    speak "Exiting without launching user interface"
fi

read -p "Press Enter to exit..."
