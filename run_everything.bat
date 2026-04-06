@echo off
cd /d %~dp0

REM =========================
REM Setup
REM =========================
call :run "Setting up the file structure..." py setup_project.py
call :run "Installing Requirements..." pip install -r requirements.txt
call :run "Verifying Requirements Installation..." python verify_installation.py

echo Starting pipeline...

REM =========================
REM Phase 1
REM =========================
call :run "" py run_phase1ab.py
powershell -c "[console]::beep(800,400)"

call :run "" py verify_phase1ab.py
powershell -c "[console]::beep(800,400)"

call :run "" py run_phase1c.py
powershell -c "[console]::beep(800,400)"

call :run "" py verify_phase1c.py
powershell -c "[console]::beep(800,400)"

call :run "" py run_phase1d.py
powershell -c "[console]::beep(800,400)"

call :run "" py verify_phase1d.py
powershell -c "[console]::beep(800,400)"

call :speak "Phase 1 complete"

REM =========================
REM Phase 2
REM =========================
call :run "" py run_phase2a.py
powershell -c "[console]::beep(800,400)"

call :run "" py verify_phase2a.py
powershell -c "[console]::beep(800,400)"

call :run "" py verify_phase2b.py
powershell -c "[console]::beep(800,400)"

call :run "" py run_phase2c.py
powershell -c "[console]::beep(800,400)"

call :run "" py verify_phase2c.py
powershell -c "[console]::beep(800,400)"

call :run "" py run_phase2d.py
powershell -c "[console]::beep(800,400)"

call :run "" py verify_phase2d.py
powershell -c "[console]::beep(800,400)"

call :run "" py run_phase2e.py
powershell -c "[console]::beep(1000,400)"

call :speak "Phase 2 complete"

REM =========================
REM Phase 3
REM =========================
call :run "" py run_phase3a.py
powershell -c "[console]::beep(800,400)"

call :run "" py verify_phase3a.py
powershell -c "[console]::beep(800,400)"

call :run "" py run_phase3b.py
powershell -c "[console]::beep(800,400)"

call :run "" py verify_phase3b.py
powershell -c "[console]::beep(800,400)"

call :run "" py run_phase3c.py
powershell -c "[console]::beep(1200,400)"

call :run "" py verify_phase3c.py
powershell -c "[console]::beep(800,400)"

call :speak "Phase 3 complete"

REM =========================
REM Phase 4 (includes 5)
REM =========================
call :run "" py run_phase4.py
powershell -c "[console]::beep(1400,400)"

call :run "" py verify_phase4.py
powershell -c "[console]::beep(800,400)"

call :speak "Phase 4 and 5 complete"

REM =========================
REM Phase 6+
REM =========================

call :run "" py run_phase6.py
powershell -c "[console]::beep(1200,400)"
call :speak "Phase 6 complete"

call :run "" py run_phase7.py
powershell -c "[console]::beep(1200,400)"
call :speak "Phase 7 complete"

call :run "" py run_phase8.py
powershell -c "[console]::beep(1200,400)"
call :speak "Phase 8 complete"

call :run "" py run_phase9.py
powershell -c "[console]::beep(1200,400)"
call :speak "Phase 9 complete"

call :run "" py run_phase10.py
powershell -c "[console]::beep(1200,400)"
call :speak "Phase 10 complete"

call :run "" py run_phase11.py
powershell -c "[console]::beep(1600,600)"
call :speak "Phase 11 complete"

call :speak "Pipeline complete"


REM =========================
REM UI Prompt
REM =========================
echo.
choice /C YN /M "Do you want to launch the UI (run_ui.py)?"

REM IMPORTANT: check in descending order
if errorlevel 2 goto skip_ui
if errorlevel 1 goto run_ui

:run_ui
echo Launching UI...
call :speak "Launching user interface"
py run_ui.py
goto end

:skip_ui
echo Skipping UI...
call :speak "Exiting without launching user interface"

:end
pause
exit /b


REM =========================
REM FUNCTIONS
REM =========================

:run
if not "%~1"=="" echo %~1
%~2 %~3 %~4 %~5 %~6 %~7 %~8 %~9

if errorlevel 1 (
    echo.
    echo ERROR: Step failed - stopping pipeline.
    powershell -c "[console]::beep(400,800)"
    call :speak "Error occurred. Pipeline stopped."
    pause
    exit /b 1
)
exit /b


:speak
powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('%~1')"
exit /b
