# Cooperation-Aware Human-Type Inference

A research implementation for modelling human cooperation, inferring human behavioural types from observed actions, and planning assistance under uncertainty in a **Keys–Doors–Gems (KDG)** environment.

## Overview

This project studies how an assistant can distinguish between two types of human behaviour:

- **Independent human:** acts according to a policy optimised for completing the task alone.
- **Collaborative human:** acts according to a policy optimised jointly with an assistant, taking the assistant's contribution into account.

The project first constructs these behavioural models, then uses observed human actions to infer which model better explains the human's behaviour. Finally, it plans assistant actions while maintaining uncertainty over the human's type.

The implementation combines Python-based planning, simulation, inference and RTDP with Julia-based PDDL trajectory visualisation.

## Research contributions and key ideas

### 1. Independent and collaborative behavioural models

The independent human policy is computed with value iteration in a single-agent KDG state space. The collaborative policy is computed in a turn-taking joint state space that represents both the human and assistant.

The joint model includes separate agent positions and key inventories, a shared set of unlocked doors, and an explicit turn variable. The assistant can move, collect keys and unlock doors, but the human is the only agent that can collect the goal gems.

### 2. A policy-mismatch model of interaction

The project also models an interaction in which the human follows their independent policy while the assistant follows a collaborative policy. This captures a setting where the assistant is willing to help, but the human does not necessarily anticipate or coordinate with that help.

The implementation projects the joint state into the human's independently modelled state, accounting for keys held by the assistant and doors already unlocked. It then selects legal actions from the corresponding Q-values.

### 3. Stochastic human behaviour and Bayesian type inference

Human actions are sampled from Boltzmann-rational policies derived from the independent and collaborative Q-value tables. The inverse-temperature parameter controls how strongly the action distribution favours high-value actions.

Given an observed trajectory, the inference procedure evaluates its likelihood under both behavioural models and accumulates log-likelihoods. Bayes' rule is then used to update the posterior probability that the human is independent or collaborative.

This provides a principled way to interpret behaviour over time rather than classifying a human from a single action.

### 4. Belief-aware assistant planning with RTDP

The final component plans assistant actions while maintaining a belief over the human's behavioural type. At human turns, the assistant predicts actions using a belief-weighted mixture of the two human models and updates its belief after observing the action. At assistant turns, it selects actions using a belief-conditioned policy.

The assistant policy is trained using Real-Time Dynamic Programming (RTDP). The implementation uses a discretised belief state and heuristic initialisation based on assistant Q-values computed for the independent-human and collaborative settings. The resulting policy is evaluated against simulated humans of both fixed types.

## End-to-end workflow

The end-to-end pipeline is organised into the following stages:

| Step | Component | Purpose |
|---:|---|---|
| 1 | `render_maze.jl` | Render the selected ASCII maze |
| 2–3 | `value_iteration.py`, `visualize_trajectory.jl` | Compute and visualise the independent human policy |
| 4–5 | `joint_value_iteration.py`, `joint_visualize_trajectory.jl` | Compute and visualise the joint human–assistant policy |
| 6–7 | `mismatch.py`, `mismatch_visualize_trajectory.jl` | Simulate and visualise the policy-mismatch interaction |
| 8 | `sample_single_trajectory.py`, `sample_joint_trajectory.py` | Sample human trajectories under the independent and collaborative models |
| 9–10 | `predict.py`, `predict_visualize_trajectory.jl` | Infer human type from sampled trajectories and visualise the predictions |
| 11 | `fixed_human_value_iteration.py` | Compute an assistant policy against a fixed independent human policy |
| 12 | `belief_aware_assistant_rtdp.py` | Train and test the belief-aware assistant policy using RTDP |
| 13 | `rtdp_visualize_trajectory.jl` | Visualise the RTDP assistant's test trajectories |

The default end-to-end script uses `maze_dataset/maze_2.txt` and writes generated results into a folder named after the maze file (by default, `maze_2/`).

## Requirements

### Python

- Python 3
- The Python scripts use the Python standard library; no third-party Python packages are required by the source files in this project.

### Julia

- **Julia 1.11.9**

The Julia scripts use the following packages:

- `PDDL.jl`
- `PDDLViz.jl`
- `CairoMakie.jl`
- `JSON3.jl`

From the repository root, instantiate the Julia project environment:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

The repository should include its Julia project files (for example, `Project.toml` and, if used, `Manifest.toml`) so that the required package versions can be resolved consistently.

## Supporting repository

This project is related to and builds on ideas from **CLIPS (Cooperative Language-Guided Inverse Plan Search)**:

- Repository: [cosilab/CLIPS.jl](https://github.com/cosilab/CLIPS.jl)

CLIPS is a cooperative inverse-planning framework for inferring human goals and providing goal assistance. It provides relevant research context for this project. The Python planning and inference pipeline in this repository implements the KDG models and algorithms described above; CLIPS.jl should be regarded as a supporting research repository, not as a Python package imported by these scripts.

## Running the project

Run commands from the project root directory, where the scripts and Julia project environment are located.

### Windows

Run the complete pipeline with:

```bat
run.bat
```

To resume from a particular pipeline stage, pass its step number. For example, to start at Step 8:

```bat
run.bat 8
```

The batch script runs all stages from the specified step through Step 13. Ensure that the outputs required by the selected starting step already exist.

### macOS / Linux

Run the complete pipeline with:

```bash
chmod +x run.sh
./run.sh
```

To resume from a particular stage:

```bash
./run.sh 8
```

The supplied source archive used to prepare this README contained `run.bat` but did not contain `run.sh`. The macOS/Linux command above documents the intended entry point; make sure the corresponding `run.sh` script is present in the repository before relying on it.

## Project structure

The main source files are:

| File | Description |
|---|---|
| `value_iteration.py` | KDG environment support and independent human value iteration |
| `joint_value_iteration.py` | Joint human–assistant value iteration |
| `mismatch.py` | Mixed interaction: independent human and collaborative assistant |
| `sample_single_trajectory.py` | Samples human actions from the independent model |
| `sample_joint_trajectory.py` | Samples human actions from the collaborative model |
| `predict.py` | Bayesian human-type inference from observed trajectories |
| `fixed_human_value_iteration.py` | Assistant planning against a fixed independent human policy |
| `belief_aware_assistant_rtdp.py` | Belief-aware assistant policy training and evaluation with RTDP |
| `render_maze.jl` | Renders the ASCII maze |
| `visualize_trajectory.jl` | Visualises the independent human trajectory |
| `joint_visualize_trajectory.jl` | Visualises the joint human–assistant trajectory |
| `mismatch_visualize_trajectory.jl` | Visualises the mismatch interaction |
| `predict_visualize_trajectory.jl` | Visualises human-type prediction results |
| `rtdp_visualize_trajectory.jl` | Visualises RTDP assistant evaluation trajectories |
| `run.bat` | Windows end-to-end pipeline script |
| `run.sh` | Intended macOS/Linux end-to-end pipeline script |

## Maze input

The default pipeline expects the maze file at:

```text
maze_dataset/maze_{x}.txt
```

The maze describes the KDG grid world using ASCII symbols. The maze must contain the required human start and, for joint human–assistant planning, an assistant start. Check the maze parser in `value_iteration.py` for the accepted symbols and input constraints before creating a new maze.

## Outputs

The pipeline generates intermediate JSON files containing value functions, Q-values, policies, trajectories, sampled observations, and posterior predictions. Julia visualisation scripts use these outputs to generate maze images and trajectory animations.

Generated results are stored in the output directory associated with the selected maze (for example, `maze_2/` for the default maze). The later stages depend on outputs from earlier stages, so when resuming the pipeline from a later step, retain the corresponding intermediate files.

## Project information

- **Project developer:** Gao Gui
- **Supervisor:** Prof. Tan Zhi-Xuan
- **Environment:** Julia 1.11.9
- **Supporting research repository:** [CLIPS.jl](https://github.com/cosilab/CLIPS.jl)
- **End-to-end execution:** `run.bat` (Windows); `run.sh` (macOS/Linux)

