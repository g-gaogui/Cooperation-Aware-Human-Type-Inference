@echo off
setlocal

REM Default: start from step 1
set START_STEP=%1
if "%START_STEP%"=="" set START_STEP=1

set "MAZE_NAME=maze_2.txt"

for %%F in ("%MAZE_NAME%") do set "FOLDER=%%~nF"

if not exist "%FOLDER%" mkdir "%FOLDER%"

echo Starting from step %START_STEP%...
echo.

if %START_STEP% LEQ 1 (
    echo ========================================
    echo Step 1: Render maze...
    echo ========================================

    julia --project=. .\render_maze.jl "maze_dataset\%MAZE_NAME%" "%FOLDER%\%FOLDER%.png"

    if errorlevel 1 goto :error
)

if %START_STEP% LEQ 2 (
    echo ========================================
    echo Step 2: Running value iteration...
    echo ========================================
    python value_iteration.py --maze "maze_dataset\%MAZE_NAME%" --out "%FOLDER%\value_iteration_results.json"
    if errorlevel 1 goto :error
)

if %START_STEP% LEQ 3 (
    echo.
    echo ========================================
    echo Step 3: Visualizing single-agent trajectory...
    echo ========================================
    julia --project=. visualize_trajectory.jl "maze_dataset\%MAZE_NAME%" "%FOLDER%\value_iteration_results.json" "%FOLDER%"
    if errorlevel 1 goto :error
)

if %START_STEP% LEQ 4 (
    echo.
    echo ========================================
    echo Step 4: Running joint value iteration...
    echo ========================================
    python joint_value_iteration.py --maze "maze_dataset\%MAZE_NAME%" --out "%FOLDER%\joint_value_iteration_results.json"
    if errorlevel 1 goto :error
)

if %START_STEP% LEQ 5 (
    echo.
    echo ========================================
    echo Step 5: Visualizing joint trajectory...
    echo ========================================
    julia --project=. joint_visualize_trajectory.jl "maze_dataset\%MAZE_NAME%" "%FOLDER%\joint_value_iteration_results.json" "%FOLDER%"
    if errorlevel 1 goto :error
)


if %START_STEP% LEQ 6 (
    echo.
    echo ========================================
    echo Step 6: Running mismatch analysis...
    echo ========================================
    python mismatch.py  --single "%FOLDER%\value_iteration_results.json" --joint "%FOLDER%\joint_value_iteration_results.json" --maze "maze_dataset\%MAZE_NAME%" --out "%FOLDER%\mismatch_results.json"
    if errorlevel 1 goto :error
)

if %START_STEP% LEQ 7 (
    echo.
    echo ========================================
    echo Step 7: Visualizing mismatch trajectory...
    echo ========================================
    julia --project=. mismatch_visualize_trajectory.jl "maze_dataset\%MAZE_NAME%" "%FOLDER%\mismatch_results.json" "%FOLDER%"
    if errorlevel 1 goto :error
)

if %START_STEP% LEQ 8 (
    echo.
    echo ========================================
    echo Step 8: Sample single and joint trajectory...
    echo ========================================
    python sample_single_trajectory.py  --q_file "%FOLDER%\value_iteration_results.json" --maze_file "maze_dataset\%MAZE_NAME%" --out_file "%FOLDER%\observed_ind_H_trajectory.json" --beta 20
    if errorlevel 1 goto :error
    python sample_joint_trajectory.py  --q_file "%FOLDER%\joint_value_iteration_results.json" --maze_file "maze_dataset\%MAZE_NAME%" --out_file "%FOLDER%\observed_col_H_trajectory.json" --beta 20
    if errorlevel 1 goto :error
)

if %START_STEP% LEQ 9 (
    echo.
    echo ========================================
    echo Step 9: Predict type based on trajectory...
    echo ========================================
    python predict.py  --single "%FOLDER%\value_iteration_results.json" --joint "%FOLDER%\joint_value_iteration_results.json"  --trajectory "%FOLDER%\observed_ind_H_trajectory.json" --maze "maze_dataset\%MAZE_NAME%" --out "%FOLDER%\predicted_type_single.json" --beta 20
    if errorlevel 1 goto :error
    python predict.py  --single "%FOLDER%\value_iteration_results.json" --joint "%FOLDER%\joint_value_iteration_results.json"  --trajectory "%FOLDER%\observed_col_H_trajectory.json" --maze "maze_dataset\%MAZE_NAME%" --out "%FOLDER%\predicted_type_joint.json" --beta 20
    if errorlevel 1 goto :error
)

if %START_STEP% LEQ 10 (
    echo.
    echo ========================================
    echo Step 10: Visualizing predictions...
    echo ========================================
    julia --project=. predict_visualize_trajectory.jl "maze_dataset\%MAZE_NAME%" "%FOLDER%\predicted_type_single.json" "%FOLDER%" "predicted_type_single.gif" "predicted_type_single.png"
    if errorlevel 1 goto :error
    julia --project=. predict_visualize_trajectory.jl "maze_dataset\%MAZE_NAME%" "%FOLDER%\predicted_type_joint.json" "%FOLDER%" "predicted_type_joint.gif" "predicted_type_joint.png"
    if errorlevel 1 goto :error
)

if %START_STEP% LEQ 11 (
    echo.
    echo ========================================
    echo Step 11: Train collaborative assistant policy against fixed independent human policy...
    echo ========================================
    python fixed_human_value_iteration.py  --maze "maze_dataset\%MAZE_NAME%" --out "%FOLDER%\col_A_against_ind_H.json" --human-q-values "%FOLDER%\value_iteration_results.json"
    if errorlevel 1 goto :error
)

if %START_STEP% LEQ 12 (
    echo.
    echo ========================================
    echo Step 12: Belief aware assistant RTDP...
    echo ========================================
    python belief_aware_assistant_rtdp.py train --maze "maze_dataset\%MAZE_NAME%" --independent-human-q-values "%FOLDER%\value_iteration_results.json" --joint-q-values "%FOLDER%\joint_value_iteration_results.json" --assistant-independent-q-values "%FOLDER%\col_A_against_ind_H.json" --out "%FOLDER%\belief_aware_assistant_rtdp.json"  --seed 42 --beta 20 
    if errorlevel 1 goto :error
    python belief_aware_assistant_rtdp.py test --maze "maze_dataset\%MAZE_NAME%" --out-dir "%FOLDER%" --independent-human-q-values "%FOLDER%\value_iteration_results.json" --joint-q-values "%FOLDER%\joint_value_iteration_results.json" --policy "%FOLDER%\belief_aware_assistant_rtdp.json" --seed 42
    if errorlevel 1 goto :error
)

if %START_STEP% LEQ 13 (
    echo.
    echo ========================================
    echo Step 13: Visualize RTDP trajectory...
    echo ========================================
    julia --project=. rtdp_visualize_trajectory.jl "maze_dataset\%MAZE_NAME%" "%FOLDER%\belief_aware_assistant_rtdp_test_collaborative.json" "%FOLDER%" "belief_aware_assistant_rtdp_test_collaborative.gif"
    if errorlevel 1 goto :error
    julia --project=. rtdp_visualize_trajectory.jl "maze_dataset\%MAZE_NAME%" "%FOLDER%\belief_aware_assistant_rtdp_test_independent.json" "%FOLDER%" "belief_aware_assistant_rtdp_test_independent.gif"
    if errorlevel 1 goto :error
)

echo.
echo ========================================
echo All selected steps completed successfully!
echo ========================================
pause
exit /b 0

:error
echo.
echo ========================================
echo ERROR: The previous step failed.
echo ========================================
pause
exit /b 1