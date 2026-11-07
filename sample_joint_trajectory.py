import json
import math
import random
import argparse
from pathlib import Path
from typing import Optional


def generate_human_trajectory(
    maze_file: str | Path,
    q_file: str | Path,
    out_file: str | Path,
    human_goal: str = "both",
    beta: float = 1.0,
    max_steps: int = 50,
    seed: Optional[int] = None,
) -> list[str]:
    """
    Generate a stochastic human trajectory from the collaborative
    human policy learned by joint value iteration.

    The assistant is assumed to always WAIT.

    Parameters
    ----------
    maze_file:
        Path to the ASCII maze.

    q_file:
        JSON output produced by joint_value_iteration.py.

    out_file:
        Where to save the generated trajectory JSON.

    human_goal:
        Human goal specification, e.g. "both", "red", or "yellow".

    beta:
        Rationality / inverse-temperature parameter.

        beta = 0:
            uniform random choice.

        larger beta:
            more strongly favors higher-Q actions.

    max_steps:
        Maximum number of HUMAN actions to generate.
        This prevents an infinite trajectory when the collaborative
        policy keeps choosing H:wait because it expects the assistant
        to act.

    seed:
        Optional random seed.

    Returns
    -------
    list[str]
        The generated human actions, without the "H:" prefix.
    """

    if beta < 0:
        raise ValueError("beta must be non-negative")

    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    if seed is not None:
        random.seed(seed)

    # Import the exact state-transition machinery used by joint VI.
    from joint_value_iteration import (
        initial_joint_state,
        is_goal,
        joint_state_key,
        joint_successors,
    )
    from value_iteration import load_maze, parse_goal_spec

    maze_file = Path(maze_file)
    q_file = Path(q_file)
    out_file = Path(out_file)

    # ------------------------------------------------------------
    # Load maze
    # ------------------------------------------------------------
    maze = load_maze(maze_file)

    # Parse goal using the same function as joint VI.
    human_label, goal_H = parse_goal_spec(human_goal)

    # ------------------------------------------------------------
    # Load joint-VI results
    # ------------------------------------------------------------
    with q_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    goal_key = f"H:{human_label}"

    if "goals" not in data:
        raise ValueError(
            f"Expected joint-VI JSON to contain 'goals'. "
            f"Available top-level keys: {list(data.keys())}"
        )

    if goal_key not in data["goals"]:
        raise ValueError(
            f"Goal {goal_key!r} not found in joint-VI results. "
            f"Available goals: {list(data['goals'].keys())}"
        )

    solution = data["goals"][goal_key]

    if "q_values" not in solution:
        raise ValueError(
            f"Joint-VI result for {goal_key!r} does not contain 'q_values'."
        )

    q_values = solution["q_values"]

    # ------------------------------------------------------------
    # Start from the exact initial joint state used by joint VI.
    # ------------------------------------------------------------
    state = initial_joint_state(maze)

    trajectory = []

    # ------------------------------------------------------------
    # Helper: Boltzmann sampling over Q-values
    # ------------------------------------------------------------
    def sample_action(action_q_values: dict[str, float]) -> str:
        finite_actions = []

        for action, q in action_q_values.items():
            if q is None:
                continue

            q = float(q)

            if not math.isfinite(q):
                continue

            finite_actions.append((action, q))

        if not finite_actions:
            raise RuntimeError(
                "No finite Q-values available for the current state."
            )

        # beta = 0 -> uniform distribution.
        if beta == 0:
            return random.choice(
                [action for action, _ in finite_actions]
            )

        # Stable softmax.
        logits = [beta * q for _, q in finite_actions]
        max_logit = max(logits)

        weights = [
            math.exp(logit - max_logit)
            for logit in logits
        ]

        actions = [
            action for action, _ in finite_actions
        ]

        return random.choices(
            actions,
            weights=weights,
            k=1,
        )[0]

    # ------------------------------------------------------------
    # Helper: apply a particular action using the SAME transition
    # function as joint VI.
    # ------------------------------------------------------------
    def apply_action(state, target_action):
        for action, next_state, _, _ in joint_successors(maze, state):
            if action == target_action:
                return next_state

        raise RuntimeError(
            f"Action {target_action!r} is not a valid successor "
            f"of the current joint state."
        )

    # ------------------------------------------------------------
    # Generate trajectory.
    #
    # Each iteration corresponds to ONE human decision:
    #
    #     human chooses H:action
    #             ↓
    #     assistant is forced to A:wait
    #             ↓
    #     next human decision
    #
    # This deliberately does NOT follow the assistant policy from
    # joint VI.
    # ------------------------------------------------------------
    for step in range(max_steps):

        # Goal check before generating another human action.
        if is_goal(state, goal_H):
            break

        # The state should always be a human-turn state here.
        turn = state[-1]

        if turn != "human":
            raise RuntimeError(
                f"Expected human turn, but joint state has turn={turn!r}"
            )

        # --------------------------------------------------------
        # Look up Q-values for the current joint state.
        # --------------------------------------------------------
        state_key = joint_state_key(state)

        if state_key not in q_values:
            raise RuntimeError(
                f"Current joint state is missing from the Q-table "
                f"at human step {step}.\n"
                f"State key: {state_key}"
            )

        state_q = q_values[state_key]

        # Only human actions are relevant at a human turn.
        human_q = {
            action: q
            for action, q in state_q.items()
            if action.startswith("H:")
        }

        if not human_q:
            raise RuntimeError(
                f"No human actions found in Q-table for state "
                f"at human step {step}."
            )

        # --------------------------------------------------------
        # Sample human action.
        # --------------------------------------------------------
        human_action = sample_action(human_q)

        # Store WITHOUT "H:" so that the output has the same
        # format as the previous generate_human_trajectory().
        raw_human_action = human_action[2:]
        trajectory.append(raw_human_action)

        # --------------------------------------------------------
        # Apply human action.
        # --------------------------------------------------------
        state = apply_action(state, human_action)

        # If human has reached the goal, stop immediately.
        # There is no need to append an assistant wait after
        # the terminal action.
        if is_goal(state, goal_H):
            break

        # --------------------------------------------------------
        # Assistant does NOT follow its collaborative policy.
        #
        # We explicitly assume the robot never moves.
        # Therefore its action is always A:wait.
        # --------------------------------------------------------
        state = apply_action(state, "A:wait")

    # ------------------------------------------------------------
    # Save trajectory.
    # ------------------------------------------------------------
    out_file.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "goal": human_label,
        "beta": beta,
        "max_steps": max_steps,
        "assistant_policy": "always_wait",
        "path": trajectory,
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Generated {len(trajectory)} human actions.")
    print(f"Wrote trajectory to {out_file}")

    return trajectory


def main():
    parser = argparse.ArgumentParser(description="Generate human trajectory")

    parser.add_argument(
        "--q_file",
        type=Path,
        default=Path("mazes/joint_value_iteration_results.json"),
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
        max_steps=args.max_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()