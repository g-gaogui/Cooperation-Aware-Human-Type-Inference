
#!/bin/bash

# Stop execution if any command fails
set -e

# Show Python prints immediately when stdout is piped (e.g. Jupyter).
export PYTHONUNBUFFERED=1

# Default: start from step 1
START_STEP=${1:-1}

MAZE_NAME="maze_2.txt"
FOLDER="${MAZE_NAME%.*}"

# Create output folder if it does not exist
mkdir -p "$FOLDER"

echo "Starting from step $START_STEP..."
echo

echo "========================================"
echo "Installing Julia packages if needed..."
echo "========================================"
julia --project=. -e "using Pkg; Pkg.instantiate()"
echo

if [ "$START_STEP" -le 1 ]; then
    echo "========================================"
    echo "Step 1: Render maze..."
    echo "========================================"

    julia --project=. render_maze.jl \
        "maze_dataset/$MAZE_NAME" \
        "$FOLDER/$FOLDER.png"
fi

if [ "$START_STEP" -le 2 ]; then
    echo "========================================"
    echo "Step 2: Running value iteration..."
    echo "========================================"

    python value_iteration.py \
        --maze "maze_dataset/$MAZE_NAME" \
        --out "$FOLDER/value_iteration_results.json"
fi

if [ "$START_STEP" -le 3 ]; then
    echo
    echo "========================================"
    echo "Step 3: Visualizing single-agent trajectory..."
    echo "========================================"

    julia --project=. visualize_trajectory.jl \
        "maze_dataset/$MAZE_NAME" \
        "$FOLDER/value_iteration_results.json" \
        "$FOLDER"
fi

if [ "$START_STEP" -le 4 ]; then
    echo
    echo "========================================"
    echo "Step 4: Running joint value iteration..."
    echo "========================================"

    python joint_value_iteration.py \
        --maze "maze_dataset/$MAZE_NAME" \
        --out "$FOLDER/joint_value_iteration_results.json"
fi

if [ "$START_STEP" -le 5 ]; then
    echo
    echo "========================================"
    echo "Step 5: Visualizing joint trajectory..."
    echo "========================================"

    julia --project=. joint_visualize_trajectory.jl \
        "maze_dataset/$MAZE_NAME" \
        "$FOLDER/joint_value_iteration_results.json" \
        "$FOLDER"
fi

if [ "$START_STEP" -le 6 ]; then
    echo
    echo "========================================"
    echo "Step 6: Running mismatch analysis..."
    echo "========================================"

    python mismatch.py \
        --single "$FOLDER/value_iteration_results.json" \
        --joint "$FOLDER/joint_value_iteration_results.json" \
        --maze "maze_dataset/$MAZE_NAME" \
        --out "$FOLDER/mismatch_results.json"
fi

if [ "$START_STEP" -le 7 ]; then
    echo
    echo "========================================"
    echo "Step 7: Visualizing mismatch trajectory..."
    echo "========================================"

    julia --project=. mismatch_visualize_trajectory.jl \
        "maze_dataset/$MAZE_NAME" \
        "$FOLDER/mismatch_results.json" \
        "$FOLDER"
fi

if [ "$START_STEP" -le 8 ]; then
    echo
    echo "========================================"
    echo "Step 8: Sample single and joint trajectory..."
    echo "========================================"

    python sample_single_trajectory.py \
        --q_file "$FOLDER/value_iteration_results.json" \
        --maze_file "maze_dataset/$MAZE_NAME" \
        --out_file "$FOLDER/observed_ind_H_trajectory.json" \
        --beta 20

    python sample_joint_trajectory.py \
        --q_file "$FOLDER/joint_value_iteration_results.json" \
        --maze_file "maze_dataset/$MAZE_NAME" \
        --out_file "$FOLDER/observed_col_H_trajectory.json" \
        --beta 20
fi

if [ "$START_STEP" -le 9 ]; then
    echo
    echo "========================================"
    echo "Step 9: Predict type based on trajectory..."
    echo "========================================"

    python predict.py \
        --single "$FOLDER/value_iteration_results.json" \
        --joint "$FOLDER/joint_value_iteration_results.json" \
        --trajectory "$FOLDER/observed_ind_H_trajectory.json" \
        --maze "maze_dataset/$MAZE_NAME" \
        --out "$FOLDER/predicted_type_single.json" \
        --beta 20

    python predict.py \
        --single "$FOLDER/value_iteration_results.json" \
        --joint "$FOLDER/joint_value_iteration_results.json" \
        --trajectory "$FOLDER/observed_col_H_trajectory.json" \
        --maze "maze_dataset/$MAZE_NAME" \
        --out "$FOLDER/predicted_type_joint.json" \
        --beta 20
fi

if [ "$START_STEP" -le 10 ]; then
    echo
    echo "========================================"
    echo "Step 10: Visualizing predictions..."
    echo "========================================"

    julia --project=. predict_visualize_trajectory.jl \
        "maze_dataset/$MAZE_NAME" \
        "$FOLDER/predicted_type_single.json" \
        "$FOLDER" \
        "predicted_type_single.gif" \
        "predicted_type_single.png"

    julia --project=. predict_visualize_trajectory.jl \
        "maze_dataset/$MAZE_NAME" \
        "$FOLDER/predicted_type_joint.json" \
        "$FOLDER" \
        "predicted_type_joint.gif" \
        "predicted_type_joint.png"
fi

if [ "$START_STEP" -le 11 ]; then
    echo
    echo "========================================"
    echo "Step 11: Train collaborative assistant policy against fixed independent human policy..."
    echo "========================================"

    python fixed_human_value_iteration.py \
        --maze "maze_dataset/$MAZE_NAME" \
        --out "$FOLDER/col_A_against_ind_H.json" \
        --human-q-values "$FOLDER/value_iteration_results.json"
fi

if [ "$START_STEP" -le 12 ]; then
    echo
    echo "========================================"
    echo "Step 12: Belief aware assistant RTDP..."
    echo "========================================"

    python belief_aware_assistant_rtdp.py train \
        --maze "maze_dataset/$MAZE_NAME" \
        --independent-human-q-values "$FOLDER/value_iteration_results.json" \
        --joint-q-values "$FOLDER/joint_value_iteration_results.json" \
        --assistant-independent-q-values "$FOLDER/col_A_against_ind_H.json" \
        --out "$FOLDER/belief_aware_assistant_rtdp.json" \
        --seed 42 \
        --beta 20

    python belief_aware_assistant_rtdp.py test \
        --maze "maze_dataset/$MAZE_NAME" \
        --out-dir "$FOLDER" \
        --independent-human-q-values "$FOLDER/value_iteration_results.json" \
        --joint-q-values "$FOLDER/joint_value_iteration_results.json" \
        --policy "$FOLDER/belief_aware_assistant_rtdp.json" \
        --seed 42
fi

if [ "$START_STEP" -le 13 ]; then
    echo
    echo "========================================"
    echo "Step 13: Visualize RTDP trajectory..."
    echo "========================================"

    julia --project=. rtdp_visualize_trajectory.jl \
        "maze_dataset/$MAZE_NAME" \
        "$FOLDER/belief_aware_assistant_rtdp_test_collaborative.json" \
        "$FOLDER" \
        "belief_aware_assistant_rtdp_test_collaborative.gif"

    julia --project=. rtdp_visualize_trajectory.jl \
        "maze_dataset/$MAZE_NAME" \
        "$FOLDER/belief_aware_assistant_rtdp_test_independent.json" \
        "$FOLDER" \
        "belief_aware_assistant_rtdp_test_independent.gif"
fi

echo
echo "========================================"
echo "All selected steps completed successfully!"
echo "========================================"
