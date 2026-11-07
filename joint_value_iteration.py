"""
Joint value iteration for the Doors-Keys-Gems ASCII maze (human + assistant).

Builds directly on value_iteration.py (step 1): same maze loader, same
move/pickup/unlock primitives, same JSON output shape per goal
(utility / q_values / policy / trajectory / pddl_plan / ...), so the
existing Julia visualizer needs only small extensions rather than a
rewrite, and step 1's human-only state is embedded verbatim inside
the joint state (see DESIGN NOTES below).

Put this file in the same directory as value_iteration.py.

Run from my_project/ or repo root:
  python my_project/joint_value_iteration.py
  python my_project/joint_value_iteration.py --human-goal red --assistant-goal yellow

======================== DESIGN NOTES (read first) =========================

1. JOINT STATE
     (posH, keysH, gemsH, posA, keysA, unlocked, turn)
   `unlocked` (doors) is SHARED: either agent unlocking a door opens it
   for both. `keys_held` / `gems_held` are per-agent. This is exactly
   step 1's State = (pos, keys_held, unlocked, gems_held) duplicated
   per agent, with `unlocked` pulled out to the shared level.

2. TURN-TAKING, NOT SIMULTANEOUS JOINT ACTIONS.
   Step 1's PDDL export already alternates "(human-action) (wait robot)"
   to preserve turn order. This file follows that convention: at each
   MDP step exactly one agent acts (real action, cost 1) or explicitly
   waits (cost 0), then turn flips. This sidesteps having to define a
   collision-resolution rule for two agents moving simultaneously.
   >>> If your actual domain requires simultaneous joint actions
   >>> instead, the fix is localized: replace `joint_successors` with
   >>> a version that enumerates the cross-product of both agents'
   >>> individual actions per step; everything downstream (VI, cost
   >>> tracking, serialization) is unaffected.

3. SEPARATE HUMAN / ASSISTANT COST.
   Every transition yields `(human_cost, assistant_cost)`. Only the
   acting agent pays the action cost, so a human action has `(1, 0)` and
   an assistant action has `(0, 1)`. The VI objective is still the total
   team action cost `human_cost + assistant_cost`, but the trajectory and
   final output preserve the two costs separately, including per-step
   costs and separate cost-to-go values.

4. GOAL: ONLY THE HUMAN COLLECTS THE GOAL GEM(S).
   The assistant is an assisting agent: it may move, pick up keys, and
   unlock doors, but it cannot pick up gems. The episode terminates as
   soon as the human has collected all gems specified by `--human-goal`.
   There is no assistant goal.

5. WAIT ACTION.
   Both agents can always choose to wait (cost 0) on their turn. This
   is necessary so the state space doesn't force an agent to move
   pointlessly once its part of the goal is done, or when it must let
   the other agent pass through a doorway first.

6. COLLISION.
   An agent cannot move onto the cell the other agent currently
   occupies (checked against the OTHER agent's position at the start
   of the turn, which is safe since only one agent moves per turn).

=============================================================================
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Literal, Optional, Tuple

from value_iteration import (  # step 1 — reused as-is
    Maze,
    MOVES,
    load_maze,
    parse_goal_spec,
    passable,
    _finite_or_none,
)

Pos = Tuple[int, int]
Turn = Literal["human", "assistant"]
# (posH, keysH, gemsH, posA, keysA, unlocked, turn)
JointState = Tuple[Pos, FrozenSet[str], FrozenSet[str], Pos, FrozenSet[str], FrozenSet[str], Turn]
JointQTable = Dict[JointState, Dict[str, float]]


def initial_joint_state(maze: Maze) -> JointState:
    if maze.robot_start is None:
        raise ValueError("Maze must contain assistant/robot start 'm' for joint planning")
    return (
        maze.human_start, frozenset(), frozenset(),
        maze.robot_start, frozenset(),
        frozenset(), "human",
    )


def is_goal(state: JointState, goal_H: FrozenSet[str]) -> bool:
    _, _, gemsH, _, _, _, _ = state
    return goal_H.issubset(gemsH)


def agent_successors(
    maze: Maze,
    pos: Pos,
    keys_held: FrozenSet[str],
    gems_held: FrozenSet[str],
    unlocked: FrozenSet[str],
    other_pos: Pos,
    other_keys_held: FrozenSet[str],
    allow_gem_pickup: bool,
) -> Iterable[Tuple[str, Pos, FrozenSet[str], FrozenSet[str], FrozenSet[str], float]]:
    """
    Real (non-wait) actions for ONE agent, given the other agent's
    current position (to block moving onto an occupied cell) and the
    shared door-unlock state. Mirrors value_iteration.successors(),
    generalized to be agent-agnostic. Cost is always 1.0 per action.
    """
    x, y = pos

    for name, (dx, dy) in MOVES.items():
        nxt = (x + dx, y + dy)
        if nxt == other_pos:
            continue
        if passable(maze, nxt, unlocked):
            yield name, nxt, keys_held, gems_held, unlocked, 1.0

    key_id = None
    key_is_used = lambda kid, unlocked: any(kid.split(':')[1] in door for door in unlocked)
    for kid, kpos in maze.keys.items():
        if kpos == pos and kid not in keys_held and kid not in other_keys_held and not key_is_used(kid, unlocked):
            key_id = kid
            break
    if key_id is not None:
        yield f"pickup({key_id})", pos, keys_held | {key_id}, gems_held, unlocked, 1.0

    if allow_gem_pickup:
        gem_id = None
        for gid, (gpos, _) in maze.gems.items():
            if gpos == pos and gid not in gems_held:
                gem_id = gid
                break
        if gem_id is not None:
            yield f"pickup({gem_id})", pos, keys_held, gems_held | {gem_id}, unlocked, 1.0

    for door_id, (dpos, color) in maze.doors.items():
        if door_id in unlocked:
            continue
        if abs(dpos[0] - x) + abs(dpos[1] - y) != 1:
            continue
        key_id = f"key:{color}"
        if key_id not in keys_held:
            continue
        yield (
            f"unlock({key_id},{door_id})",
            pos, keys_held - {key_id}, gems_held, unlocked | {door_id}, 1.0,
        )


def joint_successors(
    maze: Maze, state: JointState
) -> Iterable[Tuple[str, JointState, float, float]]:
    """Yield (action_label, next_state, human_cost, assistant_cost)."""
    posH, keysH, gemsH, posA, keysA, unlocked, turn = state

    HUMAN_WAIT_COST = 0.1
    ASSISTANT_WAIT_COST = 0.1
    if turn == "human":
        for act, npos, nkeys, ngems, nunlocked, cost in agent_successors(
            maze, posH, keysH, gemsH, unlocked, posA, keysA, True
        ):
            yield (
                f"H:{act}",
                (npos, nkeys, ngems, posA, keysA, nunlocked, "assistant"),
                cost, 0.0,
            )
        yield (
            "H:wait",
            (posH, keysH, gemsH, posA, keysA, unlocked, "assistant"),
            HUMAN_WAIT_COST, 0.0,
        )
    else:
        for act, npos, nkeys, ngems, nunlocked, cost in agent_successors(
            maze, posA, keysA, frozenset(), unlocked, posH, keysH, False
        ):
            yield (
                f"A:{act}",
                (posH, keysH, gemsH, npos, nkeys, nunlocked, "human"),
                0.0, cost,
            )
        yield (
            "A:wait",
            (posH, keysH, gemsH, posA, keysA, unlocked, "human"),
            0.0, ASSISTANT_WAIT_COST,
        )


def explore_reachable(
    maze: Maze, start: JointState, goal_H: FrozenSet[str]
) -> List[JointState]:
    seen = {start}
    q = deque([start])
    while q:
        s = q.popleft()
        if is_goal(s, goal_H):
            continue  # don't expand past goal states
        for _, sp, _, _ in joint_successors(maze, s):
            if sp not in seen:
                seen.add(sp)
                q.append(sp)
    return list(seen)


def explore_reachable_with_successors(maze, start, goal_H):
    seen = {start}
    q = deque([start])
    successors = {}

    while q:
        s = q.popleft()

        if is_goal(s, goal_H):
            successors[s] = []
            continue

        succ = list(joint_successors(maze, s))
        successors[s] = succ

        for _, sp, _, _ in succ:
            if sp not in seen:
                seen.add(sp)
                q.append(sp)

    return list(seen), successors

def joint_value_iteration(
    maze: Maze,
    start: JointState,
    goal_H: FrozenSet[str],
    max_iters: int = 20_000,
) -> Tuple[Dict[JointState, float], JointQTable, Dict[JointState, str], int, List[JointState], Dict]:
    """
    Synchronous VI over reachable joint states. Combined reward is
    -(human_cost + assistant_cost) per step; V(s) = -combined-cost-to-go.
    """
    states, successors = explore_reachable_with_successors(maze, start, goal_H)
    goal_states = {
        s for s in states
        if is_goal(s, goal_H)
    }
    V: Dict[JointState, float] = {
        s: 0.0 if s in goal_states else float("-inf") for s in states
    }

    n_iters = 0
    for n_iters in range(1, max_iters + 1):
        changed = False
        V_new = dict(V)
        for s in states:
            if s in goal_states:
                V_new[s] = 0.0
                continue
            best = float("-inf")
            for _, sp, ch, ca in successors[s]:
                cand = -(ch + ca) + V[sp]
                if cand > best:
                    best = cand
            if best != V[s]:
                changed = True
            V_new[s] = best
        V = V_new
        if not changed:
            break
    else:
        raise RuntimeError(f"Joint VI did not converge in {max_iters} iterations")

    Q: JointQTable = {}
    policy: Dict[JointState, str] = {}
    for s in states:
        acts: Dict[str, float] = {}
        if s in goal_states:
            Q[s] = acts
            continue
        best_act, best_q = None, float("-inf")
        for act, sp, ch, ca in successors[s]:
            q = -(ch + ca) + V[sp]
            acts[act] = q
            if q > best_q:
                best_q, best_act = q, act
        Q[s] = acts
        if best_act is not None and best_q > float("-inf"):
            policy[s] = best_act

    return V, Q, policy, n_iters, states, successors, goal_states


def evaluate_agent_costs(
    maze: Maze,
    states: List[JointState],
    policy: Dict[JointState, str],
    goal_H: FrozenSet[str],
    successors: Dict[JointState, List[Tuple[str, JointState, float, float]]],
    goal_states: FrozenSet[JointState],
    max_iters: int = 20_000,
) -> Tuple[Dict[JointState, float], Dict[JointState, float]]:
    """
    Second pass, ON TOP OF the already-optimal joint policy: recover
    human-cost-to-go and assistant-cost-to-go SEPARATELY at every
    state, by policy evaluation (not re-optimizing). Converges because
    the optimal policy cannot cycle without incurring infinite cost.
    """
    BIG = float("inf")
    HC: Dict[JointState, float] = {s: 0.0 if s in goal_states else BIG for s in states}
    AC: Dict[JointState, float] = dict(HC)

    succ_by_state: Dict[JointState, Tuple[JointState, float, float]] = {}
    for s in states:
        if s in goal_states:
            continue
        act = policy.get(s)
        if act is None:
            continue
        for a, sp, ch, ca in successors[s]:
            if a == act:
                succ_by_state[s] = (sp, ch, ca)
                break

    for _ in range(max_iters):
        changed = False
        for s in states:
            if s not in succ_by_state:
                continue
            sp, ch, ca = succ_by_state[s]
            new_hc = ch + HC.get(sp, BIG)
            new_ac = ca + AC.get(sp, BIG)
            if new_hc != HC[s]:
                HC[s] = new_hc
                changed = True
            if new_ac != AC[s]:
                AC[s] = new_ac
                changed = True
        if not changed:
            break

    return HC, AC


def greedy_joint_path(
    maze: Maze,
    start: JointState,
    policy: Dict[JointState, str],
    goal_H: FrozenSet[str],
    successors: Dict[JointState, List[Tuple[str, JointState, float, float]]],
    goal_states: FrozenSet[JointState],
    max_steps: int = 50_000,
) -> List[Tuple[str, JointState, JointState]]:
    """Follow the joint policy from start until the goal. Returns
    a list of (action_label, state_before, state_after)."""
    path: List[Tuple[str, JointState, JointState]] = []
    s = start
    seen: set = set()
    for _ in range(max_steps):
        if s in goal_states:
            return path
        if s in seen:
            break  # shouldn't happen for an optimal policy; guard anyway
        seen.add(s)
        act = policy.get(s)
        if act is None:
            break
        nxt = None
        for a, sp, _, _ in successors[s]:
            if a == act:
                nxt = sp
                break
        if nxt is None:
            break
        path.append((act, s, nxt))
        s = nxt
    return path


def joint_state_key(state: JointState) -> str:
    posH, keysH, gemsH, posA, keysA, unlocked, turn = state
    return json.dumps(
        {
            "posH": list(posH), "keysH": sorted(keysH), "gemsH": sorted(gemsH),
            "posA": list(posA), "keysA": sorted(keysA),
            "unlocked": sorted(unlocked), "turn": turn,
        },
        separators=(",", ":"),
    )


def to_pddl_action(maze: Maze, agent: str, action: str) -> str:
    """action is the RAW action (without the 'H:'/'A:' prefix)."""
    if action == "wait":
        return f"(wait {agent})"
    if action in MOVES:
        return f"({action} {agent})"
    if action.startswith("pickup(") and action.endswith(")"):
        logical = action[len("pickup(") : -1]
        return f"(pickup {agent} {maze.pddl_names[logical]})"
    if action.startswith("unlock(") and action.endswith(")"):
        inner = action[len("unlock(") : -1]
        key_logical, door_logical = inner.split(",")
        return f"(unlock {agent} {maze.pddl_names[key_logical]} {maze.pddl_names[door_logical]})"
    raise ValueError(f"Unknown action for PDDL export: {action}")


def build_trajectory(
    maze: Maze,
    path: List[Tuple[str, JointState, JointState]],
    start: JointState,
    V: Dict[JointState, float],
    Q: JointQTable,
    HC: Dict[JointState, float],
    AC: Dict[JointState, float],
) -> Tuple[List[dict], List[str]]:
    """
    One trajectory entry per joint-state visited (including start).
    Keeps step 1's field names (x, y, utility, q_values, action) for
    the HUMAN, and adds rx/ry/assistant_* fields alongside — so a
    trajectory entry is a strict superset of step 1's format.
    """

    def entry(t: int, s: JointState, act_label: Optional[str], pddl_act: Optional[str]) -> dict:
        posH, keysH, gemsH, posA, keysA, unlocked, turn = s
        return {
            "t": t,
            "x": posH[0], "y": posH[1],
            "rx": posA[0], "ry": posA[1],
            "turn": turn,
            "utility": V.get(s, float("-inf")),
            "human_cost_to_go": HC.get(s, float("inf")),
            "assistant_cost_to_go": AC.get(s, float("inf")),
            "q_values": dict(Q.get(s, {})),
            "action": act_label,
            "pddl_action": pddl_act,
        }

    trajectory = [entry(0, start, None, None)]
    trajectory[0]["human_step_cost"] = 0.0
    trajectory[0]["assistant_step_cost"] = 0.0
    trajectory[0]["step_cost"] = 0.0
    pddl_plan: List[str] = []
    s = start
    for i, (act_label, s_before, s_after) in enumerate(path, start=1):
        agent_tag, raw_act = act_label.split(":", 1)
        agent_name = "human" if agent_tag == "H" else "robot"
        pddl_act = to_pddl_action(maze, agent_name, raw_act)
        pddl_plan.append(pddl_act)
        step_human_cost = 1.0 if agent_tag == "H" and raw_act != "wait" else 0.0
        step_assistant_cost = 1.0 if agent_tag == "A" and raw_act != "wait" else 0.0
        e = entry(i, s_after, act_label, pddl_act)
        e["human_step_cost"] = step_human_cost
        e["assistant_step_cost"] = step_assistant_cost
        e["step_cost"] = step_human_cost + step_assistant_cost
        trajectory.append(e)
        s = s_after

    return trajectory, pddl_plan


def serialize_q(Q: JointQTable) -> Dict[str, Dict[str, Optional[float]]]:
    return {
        joint_state_key(s): {a: _finite_or_none(q) for a, q in acts.items()}
        for s, acts in Q.items()
    }


def solve_joint(maze: Maze, human_goal_spec: str) -> dict:
    human_label, goal_H = parse_goal_spec(human_goal_spec)
    for g in goal_H:
        if g not in maze.gems:
            raise ValueError(f"Goal gem {g} not present in maze (have {list(maze.gems)})")

    start = initial_joint_state(maze)
    V, Q, policy, n_iters, states, successors, goal_states = joint_value_iteration(maze, start, goal_H)
    HC, AC = evaluate_agent_costs(maze, states, policy, goal_H, successors, goal_states)
    path = greedy_joint_path(maze, start, policy, goal_H, successors, goal_states)

    combined_cost = -V[start] if V[start] > float("-inf") else float("inf")
    human_cost = HC.get(start, float("inf"))
    assistant_cost = AC.get(start, float("inf"))

    trajectory, pddl_plan = ([], [])
    if combined_cost != float("inf"):
        trajectory, pddl_plan = build_trajectory(maze, path, start, V, Q, HC, AC)
        for step in trajectory:
            step["utility"] = _finite_or_none(step["utility"])
            step["human_cost_to_go"] = _finite_or_none(step["human_cost_to_go"])
            step["assistant_cost_to_go"] = _finite_or_none(step["assistant_cost_to_go"])
            step["q_values"] = {a: _finite_or_none(q) for a, q in step["q_values"].items()}

    V_out = {joint_state_key(s): v for s, v in V.items() if v > float("-inf")}
    pi_out = {joint_state_key(s): a for s, a in policy.items()}
    Q_out = serialize_q(Q)
    n_q = sum(len(acts) for acts in Q.values())

    return {
        "goal": f"H:{human_label}",
        "human_goal_gems": sorted(goal_H),
        "solver": "joint_value_iteration",
        "n_value_iteration_sweeps": n_iters,
        "path_cost": _finite_or_none(combined_cost),
        "human_path_cost": _finite_or_none(human_cost),
        "assistant_path_cost": _finite_or_none(assistant_cost),
        "path": [act for act, _, _ in path],
        "path_steps": [
            {
                "step": i,
                "action": act,
                "human_cost": 1.0 if act.startswith("H:") and act != "H:wait" else 0.0,
                "assistant_cost": 1.0 if act.startswith("A:") and act != "A:wait" else 0.0,
            }
            for i, (act, _, _) in enumerate(path, start=1)
        ],
        "trajectory": trajectory,
        "pddl_plan": pddl_plan,
        "n_states_with_finite_value": len(V_out),
        "n_policy_entries": len(pi_out),
        "n_q_entries": n_q,
        "utility": V_out,
        "q_values": Q_out,
        "policy": pi_out,
        "start_utility": _finite_or_none(V[start]),
        "start_human_cost": _finite_or_none(human_cost),
        "start_assistant_cost": _finite_or_none(assistant_cost),
        "start_q_values": {a: _finite_or_none(q) for a, q in Q.get(start, {}).items()},
        "pddl_names": dict(maze.pddl_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Joint value iteration (human + assistant) for the maze"
    )
    parser.add_argument(
        "--maze", type=Path,
        default=Path(__file__).resolve().parent / "mazes" / "sample.txt",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent / "mazes" / "joint_value_iteration_results.json",
    )
    parser.add_argument("--human-goal", default="both")
    args = parser.parse_args()

    maze = load_maze(args.maze)
    print(f"Loaded maze {args.maze} ({maze.width}x{maze.height})")
    print(f"Human start: {maze.human_start}   Assistant start: {maze.robot_start}")
    print()

    sol = solve_joint(maze, args.human_goal)

    results = {
        "maze": str(args.maze),
        "solver": "joint_value_iteration",
        "goals": {sol["goal"]: sol},
        "pddl_names": maze.pddl_names,
    }

    if sol["path_cost"] is None:
        print("No joint plan found reaching the human's goal.")
    else:
        print(f"VI sweeps: {sol['n_value_iteration_sweeps']}")
        print(f"Combined path cost: {sol['path_cost']}")
        print(f"  human cost:     {sol['human_path_cost']}")
        print(f"  assistant cost: {sol['assistant_path_cost']}")
        print(f"Reachable states with finite V: {sol['n_states_with_finite_value']}")
        print(f"Q entries: {sol['n_q_entries']}")
        print("Joint action sequence:")
        for i, a in enumerate(sol["path"], 1):
            print(f"  {i:3d}. {a}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote joint V, Q, policy, and trajectory to {args.out}")


if __name__ == "__main__":
    main()