import json
import math
import random
from pathlib import Path
import argparse

from value_iteration import (
    load_maze,
    initial_state,
    successors,
    is_goal,
    state_key,
)

def generate_human_trajectory(
    q_file: Path,
    maze_file: Path,
    out_file: Path,
    beta: float,
    goal: str = "both",
    max_steps: int = 1000,
    seed: int | None = None,
) -> None:
    """
    Generate a stochastic human trajectory from single-human Q-values
    using a Boltzmann-rational policy.

    Output:
        {
            "path": [
                "right",
                "right",
                "pickup(key:red)",
                ...
            ]
        }

    This format is compatible with load_observed_actions().
    """

    if beta < 0:
        raise ValueError("beta must be non-negative")

    if seed is not None:
        random.seed(seed)

    # ------------------------------------------------------------
    # Load maze
    # ------------------------------------------------------------
    maze = load_maze(maze_file)

    # ------------------------------------------------------------
    # Parse goal
    # ------------------------------------------------------------
    goal_lower = goal.lower()

    if goal_lower in ("both", "all", "red+yellow", "yellow+red"):
        goal_gems = frozenset({"gem:red", "gem:yellow"})
    else:
        goal_gems = frozenset({f"gem:{goal_lower}"})

    # ------------------------------------------------------------
    # Load Q-values
    # ------------------------------------------------------------
    with q_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "goals" in data:
        if goal_lower not in data["goals"]:
            raise ValueError(
                f"Goal {goal_lower!r} not found in Q-value file. "
                f"Available goals: {list(data['goals'].keys())}"
            )

        q_values = data["goals"][goal_lower]["q_values"]

    elif "q_values" in data:
        q_values = data["q_values"]

    else:
        # Allow a raw {state_key: {action: Q}} file.
        q_values = data

    # ------------------------------------------------------------
    # All keys available
    #
    # This is the primary scenario used by solve_goal().
    # ------------------------------------------------------------
    initial_keys = frozenset(maze.keys.keys())

    # ------------------------------------------------------------
    # Initial human state
    # ------------------------------------------------------------
    state = initial_state(maze)

    path = []

    # ------------------------------------------------------------
    # Boltzmann policy
    # ------------------------------------------------------------
    def boltzmann_probabilities(q_dict):
        valid = {
            action: float(q)
            for action, q in q_dict.items()
            if q is not None and math.isfinite(float(q))
        }

        if not valid:
            return {}

        actions = list(valid.keys())

        if beta == 0:
            p = 1.0 / len(actions)
            return {action: p for action in actions}

        scaled = {
            action: beta * q
            for action, q in valid.items()
        }

        max_scaled = max(scaled.values())

        weights = {
            action: math.exp(value - max_scaled)
            for action, value in scaled.items()
        }

        total = sum(weights.values())

        return {
            action: weight / total
            for action, weight in weights.items()
        }

    # ------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------
    for step in range(max_steps):

        # Stop once the goal is achieved.
        if is_goal(state, goal_gems):
            break

        # Build exactly the same state key used by value_iteration.py.
        current_key = state_key(
            state,
            initial_keys=initial_keys,
        )

        q_dict = q_values.get(current_key)

        if q_dict is None:
            raise RuntimeError(
                f"No Q-values found for state at step {step}.\n"
                f"State key:\n{current_key}\n"
                f"State:\n{state}"
            )

        # Convert Q-values to P(a | s).
        probabilities = boltzmann_probabilities(q_dict)

        if not probabilities:
            raise RuntimeError(
                f"No valid actions available at state:\n{current_key}"
            )

        # Sample human action.
        actions = list(probabilities.keys())
        weights = [probabilities[a] for a in actions]

        action = random.choices(
            actions,
            weights=weights,
            k=1,
        )[0]

        # --------------------------------------------------------
        # Apply the action using the exact transition function
        # used by value iteration.
        # --------------------------------------------------------
        next_state = None

        for successor_action, successor_state, _ in successors(
            maze,
            state,
            unavailable_keys=frozenset(),
        ):
            if successor_action == action:
                next_state = successor_state
                break

        if next_state is None:
            raise RuntimeError(
                f"Action {action!r} is not executable from state "
                f"{state}"
            )

        path.append(action)
        state = next_state

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    output = {
        "path": path
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(path)} human actions.")
    print(f"Final state: {state}")
    print(f"Goal reached: {is_goal(state, goal_gems)}")
    print(f"Wrote trajectory to: {out_file}")



def main():
    parser = argparse.ArgumentParser(description="Generate human trajectory")

    parser.add_argument(
        "--q_file",
        type=Path,
        default=Path("mazes/value_iteration_results.json"),
    )
    parser.add_argument(
        "--maze_file",
        type=Path,
        default=Path("mazes/sample.txt"),
    )
    parser.add_argument(
        "--out_file",
        type=Path,
        default=Path("mazes/observed_trajectory.json"),
    )
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--goal", type=str, default="both")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    generate_human_trajectory(
        q_file=args.q_file,
        maze_file=args.maze_file,
        out_file=args.out_file,
        beta=args.beta,
        goal=args.goal,
        max_steps=args.max_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()