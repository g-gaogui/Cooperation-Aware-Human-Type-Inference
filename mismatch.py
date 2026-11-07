"""
Mixed-policy rollout for the Doors-Keys-Gems maze.

Reads:
  1. single-human value-iteration JSON
  2. joint human+assistant value-iteration JSON

At each turn:
  - Human: uses the original single-human Q table. The human ignores
    assistant position/inventory, but the shared unlocked-door state is kept.
    Among actions with the maximum single-human Q value, an action is chosen
    only if its resulting position is not occupied by the assistant.
    More generally, all Q-ranked actions are considered in descending order
    until a collision-free action is found.
  - Assistant: uses the joint Q table on the full joint state.

The actual transition is reconstructed from the maze primitives, so the
rollout is evaluated in the real joint environment rather than by assuming
the two policies share the same state model.

Put this file in the same directory as value_iteration.py and
joint_value_iteration.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

from value_iteration import load_maze, parse_goal_spec, _finite_or_none
from joint_value_iteration import (
    JointState,
    initial_joint_state,
    is_goal,
    joint_successors,
    joint_state_key,
)


def single_state_key(
    pos: Tuple[int, int],
    keys: FrozenSet[str],
    unlocked: FrozenSet[str],
    gems: FrozenSet[str],
    initial_keys: Optional[FrozenSet[str]] = None,
) -> str:
    """Exactly the state-key format used by the single-human VI JSON.

    `initial_keys` mirrors value_iteration.state_key: when given, it's
    inserted right after "keys" so the lookup lands in the right
    key-availability scenario's Q/V table.
    """
    d: Dict[str, object] = {
        "pos": list(pos),
        "keys": sorted(keys),
    }
    if initial_keys is not None:
        d["initial_keys"] = sorted(initial_keys)
    d["unlocked"] = sorted(unlocked)
    d["gems"] = sorted(gems)
    return json.dumps(d, separators=(",", ":"))


def human_visible_initial_keys(maze, keysA: FrozenSet[str], unlocked: FrozenSet[str]) -> FrozenSet[str]:
    """
    Key types the human think the environment has.
    If assistant takes keys and the corresponding door is locked, then the human should act in the environment as if the key is not available.
    If a door is unlocked, then the human should act in the environment as if the corresponding key is available.
    """
    doors_to_keys = lambda unlocked: {f"key:{door.split(':')[1]}" for door in unlocked}
    if not hasattr(human_visible_initial_keys, "histkeysA"):
        human_visible_initial_keys.histkeysA = set()
    human_visible_initial_keys.histkeysA = human_visible_initial_keys.histkeysA | keysA
    all_key_types = frozenset(maze.keys.keys())
    return (all_key_types - human_visible_initial_keys.histkeysA) | doors_to_keys(unlocked)


def load_goal_solution(data: dict, goal_key: str) -> dict:
    """Find the goal solution, accepting either 'red' or 'H:red' style keys."""
    goals = data.get("goals", {})

    candidates = [goal_key]
    if goal_key.startswith("H:"):
        candidates.append(goal_key[2:])
    else:
        candidates.append(f"H:{goal_key}")

    for key in candidates:
        if key in goals:
            return goals[key]

    if len(goals) == 1:
        return next(iter(goals.values()))

    raise KeyError(
        f"Could not find goal {goal_key!r} in JSON. "
        f"Available goals: {list(goals)}"
    )


def get_q_values(table: dict, state_key: str) -> Dict[str, float]:
    q = table.get(state_key)
    if q is None:
        raise KeyError(f"State is missing from Q table: {state_key}")
    return {
        action: value
        for action, value in q.items()
        if value is not None
    }


def choose_human_action(
    state: JointState,
    single_q: dict,
    maze,
    allow_wait: bool = True
) -> Tuple[str, JointState, float, float, Dict[str, float], str]:
    """
    Choose the selfish human action.

    The human sees only:
        (posH, keysH, unlocked, gemsH)

    Collision handling is applied AFTER ranking actions by the single-human
    Q value: if the best action would move onto posA, try the next-best Q
    action. This is equivalent to minimizing the single-human cost subject
    to the joint collision constraint.

    Returns (action, next_state, human_cost, assistant_cost, q_values,
    source). The transition is resolved here, from the same
    joint_successors(maze, state) call used for the executability/
    collision check, and handed straight back to the caller — so the
    executed transition is guaranteed to be the one that was validated,
    rather than being re-derived by a second, separate call later.
    """
    posH, keysH, gemsH, posA, keysA, unlocked, _turn = state
    print(f"Choosing human action for state: {state}")
    initial_keys = human_visible_initial_keys(maze, keysA, unlocked)
    print(f"initial_keys for human: {initial_keys}")
    h_key = single_state_key(posH, keysH, unlocked, gemsH, initial_keys)
    q_values = get_q_values(single_q, h_key)
    # Highest Q = lowest cost, since Q/V are negative cost-to-go values.
    ranked = sorted(q_values.items(), key=lambda item: (item[1], item[0] != "H:wait"), reverse=True)
    # print(ranked)
    # Build the actually executable joint transitions.
    successors = {
        action: (next_state, hc, ac)
        for action, next_state, hc, ac in joint_successors(maze, state)
        if action.startswith("H:")
    }
    # print(successors)
    for raw_action, _q in ranked:
        joint_action = f"H:{raw_action}"

        if joint_action not in successors:
            # The single-human model can recommend an action that is not
            # executable in the actual joint state, e.g. because a key has
            # already been taken by the assistant. Try the next Q action.
            continue
        if not allow_wait and joint_action == "H:wait":
            continue

        next_state, hc, ac = successors[joint_action]
        next_posH = next_state[0]

        # The maze forbids occupying the assistant's current cell.
        if next_posH == posA:
            continue
        print(f"Selected human action: {joint_action}")    
        return (
            joint_action,
            next_state,
            hc,
            ac,
            q_values,
            "single_human_q",
        )

    raise RuntimeError(
        "No executable collision-free human action exists for state "
        f"{joint_state_key(state)}. Human Q candidates: {q_values}"
    )


def choose_assistant_action(
    state: JointState,
    joint_q: dict,
    maze,
    allow_wait: bool = True
) -> Tuple[str, JointState, float, float, Dict[str, float], str]:
    """Choose the cooperative assistant action from the joint Q table.

    Returns (action, next_state, human_cost, assistant_cost, q_values,
    source) — see choose_human_action's docstring for why the transition
    is resolved here rather than by a second later call.
    """
    state_key = joint_state_key(state)
    q_values = get_q_values(joint_q, state_key)

    ranked = sorted(q_values.items(), key=lambda item: (item[1], item[0] != "A:wait"), reverse=True)
    successors = {
        action: (next_state, hc, ac)
        for action, next_state, hc, ac in joint_successors(maze, state)
        if action.startswith("A:")
    }
    print(f"Choosing assistant action for state: {state}")
    for action, _q in ranked:
        if action in successors:
            if not allow_wait and action == "A:wait":
                continue
            next_state, hc, ac = successors[action]
            print(f"Selected assistant action: {action}")
            return action, next_state, hc, ac, q_values, "joint_q"

    raise RuntimeError(
        "No executable assistant action exists for state "
        f"{state_key}. Assistant Q candidates: {q_values}"
    )


def apply_action(
    state: JointState,
    action: str,
    maze,
) -> Tuple[JointState, float, float]:
    """Apply an H:/A: action using the same transition model as joint VI."""
    for act, next_state, human_cost, assistant_cost in joint_successors(maze, state):
        if act == action:
            return next_state, human_cost, assistant_cost

    raise RuntimeError(
        f"Action {action!r} is not executable from state {joint_state_key(state)}"
    )


def build_entry(
    t: int,
    state: JointState,
    action: Optional[str],
    pddl_action: Optional[str],
    utility: Optional[float],
    human_cost_to_go: Optional[float],
    assistant_cost_to_go: Optional[float],
    q_values: Dict[str, float],
    human_utility: Optional[float],
    actor_q_source: Optional[str],
    human_state_key: Optional[str],
) -> dict:
    posH, keysH, gemsH, posA, keysA, unlocked, turn = state

    return {
        "t": t,
        "x": posH[0],
        "y": posH[1],
        "rx": posA[0],
        "ry": posA[1],
        "turn": turn,
        "utility": _finite_or_none(utility) if utility is not None else None,
        "human_cost_to_go": (
            _finite_or_none(human_cost_to_go)
            if human_cost_to_go is not None
            else None
        ),
        "assistant_cost_to_go": (
            _finite_or_none(assistant_cost_to_go)
            if assistant_cost_to_go is not None
            else None
        ),
        "q_values": {
            action: _finite_or_none(value)
            for action, value in q_values.items()
        },
        "action": action,
        "pddl_action": pddl_action,
        # Extra fields preserve the original trajectory format while making
        # the mixed-policy provenance explicit.
        "human_utility": (
            _finite_or_none(human_utility)
            if human_utility is not None
            else None
        ),
        "actor_q_source": actor_q_source,
        "human_state_key": human_state_key,
    }


def run_mixed_rollout(
    maze,
    single_solution: dict,
    joint_solution: dict,
    human_goal_spec: str,
    max_steps: int = 50000,
) -> dict:
    human_label, goal_H = parse_goal_spec(human_goal_spec)

    single_q = single_solution["q_values"]
    single_v = single_solution["utility"]

    joint_q = joint_solution["q_values"]
    joint_v = joint_solution["utility"]

    start = initial_joint_state(maze)

    # The output follows the joint result's trajectory schema.
    trajectory: List[dict] = []
    path: List[str] = []
    path_steps: List[dict] = []
    pddl_plan: List[str] = []

    total_human_cost = 0.0
    total_assistant_cost = 0.0

    def projected_human_key(s: JointState) -> str:
        posH, keysH, gemsH, _posA, keysA, unlocked, _turn = s
        initial_keys = human_visible_initial_keys(maze, keysA, unlocked)
        return single_state_key(posH, keysH, unlocked, gemsH, initial_keys)

    def add_entry(
        t: int,
        s: JointState,
        action: Optional[str],
        pddl_action: Optional[str],
        q_values: Dict[str, float],
        q_source: Optional[str],
    ):
        sk = joint_state_key(s)
        hk = projected_human_key(s)

        # Joint utility is the closest semantic match to the original
        # trajectory's "utility" field. It may be absent because the mixed
        # rollout can reach states never visited by cooperative VI.
        utility = joint_v.get(sk)

        # At a human turn this is the utility of the projected solo state.
        human_utility = single_v.get(hk)

        # Keep cost-to-go fields tied to the cooperative joint table when
        # available. The mixed rollout's actual costs are reported separately.
        hctg = None
        actg = None

        trajectory.append(
            build_entry(
                t=t,
                state=s,
                action=action,
                pddl_action=pddl_action,
                utility=utility,
                human_cost_to_go=hctg,
                assistant_cost_to_go=actg,
                q_values=q_values,
                human_utility=human_utility,
                actor_q_source=q_source,
                human_state_key=hk,
            )
        )

    # Start entry: no action has yet been selected.
    start_hk = projected_human_key(start)
    start_q = get_q_values(single_q, start_hk)
    add_entry(
        0,
        start,
        None,
        None,
        start_q,
        "single_human_q",
    )

    state = start
    prev_human_action = None
    prev_assistant_action = None
    for step in range(1, max_steps + 1):
        if is_goal(state, goal_H):
            break

        if state[-1] == "human":
            action, next_state, hc, ac, q_values, q_source = choose_human_action(
                state, single_q, maze, allow_wait=not (prev_human_action == 'H:wait' and prev_assistant_action == 'A:wait')
            )
            prev_human_action = action
        else:
            action, next_state, hc, ac, q_values, q_source = choose_assistant_action(
                state, joint_q, maze, allow_wait=not (prev_human_action == 'H:wait' and prev_assistant_action == 'A:wait')
            )
            prev_assistant_action = action

        total_human_cost += hc
        total_assistant_cost += ac

        # Convert the action to PDDL using the same helper as joint VI.
        from joint_value_iteration import to_pddl_action

        agent = "human" if action.startswith("H:") else "robot"
        raw_action = action.split(":", 1)[1]
        pddl_action = to_pddl_action(maze, agent, raw_action)

        path.append(action)
        pddl_plan.append(pddl_action)

        path_steps.append(
            {
                "step": step,
                "action": action,
                "human_cost": hc,
                "assistant_cost": ac,
            }
        )

        add_entry(
            step,
            next_state,
            action,
            pddl_action,
            q_values,
            q_source,
        )

        state = next_state

    reached_goal = is_goal(state, goal_H)

    # Replace trajectory step-cost fields now that the full path is known.
    # The action stored in each entry is the action that led INTO that state.
    trajectory[0]["human_step_cost"] = 0.0
    trajectory[0]["assistant_step_cost"] = 0.0
    trajectory[0]["step_cost"] = 0.0

    for i, path_step in enumerate(path_steps, start=1):
        trajectory[i]["human_step_cost"] = path_step["human_cost"]
        trajectory[i]["assistant_step_cost"] = path_step["assistant_cost"]
        trajectory[i]["step_cost"] = (
            path_step["human_cost"] + path_step["assistant_cost"]
        )
        # Realized cost-to-go: what each agent actually pays from each state onward.
        # trajectory[t] is the state after t steps, and path_steps[t] is the step
        # that leaves it, so cost-to-go at t = sum of path_steps[t:].
        n = len(path_steps)
        h_ctg = [0.0] * (n + 1)   # h_ctg[n] = 0 at the final state
        a_ctg = [0.0] * (n + 1)
        for t in range(n - 1, -1, -1):
            h_ctg[t] = h_ctg[t + 1] + path_steps[t]["human_cost"]
            a_ctg[t] = a_ctg[t + 1] + path_steps[t]["assistant_cost"]

        if reached_goal:
            for t, entry in enumerate(trajectory):
                entry["human_cost_to_go"] = _finite_or_none(h_ctg[t])
                entry["assistant_cost_to_go"] = _finite_or_none(a_ctg[t])
            start_human_cost = h_ctg[0]
            start_assistant_cost = a_ctg[0]
        else:
            start_human_cost = None
            start_assistant_cost = None

    # Actual mixed-policy cost is the quantity relevant to this rollout.
    combined_cost = (
        total_human_cost + total_assistant_cost
        if reached_goal
        else None
    )

    # Preserve the joint result's table structure in the output. The mixed
    # trajectory is the new object being evaluated; the original utility/Q
    # tables remain available for reproducibility.
    result = {
        "goal": f"H:{human_label}",
        "human_goal_gems": sorted(goal_H),
        "solver": "mixed_policy_rollout",
        "policy_description": (
            "Human uses single-human Q; assistant uses joint cooperative Q. "
            "Human action selection is additionally constrained by the "
            "joint collision rule."
        ),
        "reached_goal": reached_goal,
        "path_cost": combined_cost,
        "human_path_cost": total_human_cost if reached_goal else None,
        "assistant_path_cost": total_assistant_cost if reached_goal else None,
        "path": path,
        "path_steps": path_steps,
        "trajectory": trajectory,
        "pddl_plan": pddl_plan,
        "n_steps": len(path),
        "start_utility": joint_v.get(joint_state_key(start)),
        "start_human_utility": single_v.get(start_hk),
        "start_human_cost": start_human_cost,
        "start_assistant_cost": start_assistant_cost,
        "start_q_values": trajectory[0]["q_values"],
        # Keep both source tables in the output, under their original names.
        "utility": joint_solution["utility"],
        "q_values": joint_solution["q_values"],
        "policy": joint_solution.get("policy", {}),
        "single_human_utility": single_solution["utility"],
        "single_human_q_values": single_solution["q_values"],
        "single_human_policy": single_solution.get("policy", {}),
        "pddl_names": maze.pddl_names,
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mixed selfish-human / cooperative-assistant policy"
    )
    parser.add_argument("--single", type=Path, default=Path(__file__).resolve().parent / "mazes" / "value_iteration_results.json")
    parser.add_argument("--joint", type=Path, default=Path(__file__).resolve().parent / "mazes" / "joint_value_iteration_results.json")
    parser.add_argument(
        "--maze",
        type=Path,
        default=Path(__file__).resolve().parent / "mazes" / "sample.txt",
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "mazes" / "mixed_rollout_results.json")
    parser.add_argument("--human-goal", default="both")
    parser.add_argument("--max-steps", type=int, default=50000)
    args = parser.parse_args()

    with args.single.open("r", encoding="utf-8") as f:
        single_data = json.load(f)

    with args.joint.open("r", encoding="utf-8") as f:
        joint_data = json.load(f)

    maze = load_maze(args.maze)

    single_solution = load_goal_solution(single_data, args.human_goal)
    joint_solution = load_goal_solution(joint_data, args.human_goal)

    result = run_mixed_rollout(
        maze,
        single_solution,
        joint_solution,
        args.human_goal,
        args.max_steps,
    )

    output = {
        "maze": str(args.maze),
        "solver": "mixed_policy_rollout",
        "goals": {result["goal"]: result},
        "pddl_names": maze.pddl_names,
        "source_files": {
            "single_human": str(args.single),
            "joint": str(args.joint),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Reached goal: {result['reached_goal']}")
    print(f"Steps: {result['n_steps']}")
    print(f"Human cost: {result['human_path_cost']}")
    print(f"Assistant cost: {result['assistant_path_cost']}")
    print(f"Wrote mixed rollout to {args.out}")


if __name__ == "__main__":
    main()