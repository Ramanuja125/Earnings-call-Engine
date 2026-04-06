@echo off
cd /d %~dp0

call :run "Setting up the file structure..." py setup_project.py
call :run "Installing Requirements..." pip install -r requirements.txt
call :run "Verifying Requirements Installation..." python verify_installation.py

echo Starting pipeline...

REM Phase 1
call :run "" py run_phase1ab.py
powershell -c "[console]::beep(800,400)"

call :run "" py run_phase1c.py
powershell -c "[console]::beep(800,400)"

call :run "" py run_phase1d.py
powershell -c "[console]::beep(800,400)"

REM Phase 2
call :run "" py run_phase2a.py
powershell -c "[console]::beep(800,400)"

call :run "" py run_phase2b.py
powershell -c "[console]::beep(800,400)"

call :run "" py run_phase2c.py
powershell -c "[console]::beep(800,400)"

call :run "" py run_phase2d.py
powershell -c "[console]::beep(800,400)"

call :run "" py run_phase2e.py
powershell -c "[console]::beep(1000,400)"

REM Phase 3
call :run "" py run_phase3a.py
powershell -c "[console]::beep(800,400)"

call :run "" py run_phase3b.py
powershell -c "[console]::beep(800,400)"

call :run "" py run_phase3c.py
powershell -c "[console]::beep(1200,400)"

REM Phase 4
call :run "" py run_phase4.py
powershell -c "[console]::beep(1400,400)"

REM Phase 6+
call :run "" py run_phase6.py
powershell -c "[console]::beep(1200,400)"

call :run "" py run_phase7.py
powershell -c "[console]::beep(1200,400)"

call :run "" py run_phase8.py
powershell -c "[console]::beep(1200,400)"

call :run "" py run_phase9.py
powershell -c "[console]::beep(1200,400)"

call :run "" py run_phase10.py
powershell -c "[console]::beep(1200,400)"

call :run "" py run_phase11.py
powershell -c "[console]::beep(1600,600)"

echo Pipeline completed!
pause
exit /b

:run
if not "%~1"=="" echo %~1
%~2 %~3 %~4 %~5 %~6 %~7 %~8 %~9

if errorlevel 1 (
    echo.
    echo ❌ ERROR: Step failed — stopping pipeline.
    pause
    exit /b 1
)
exit /b