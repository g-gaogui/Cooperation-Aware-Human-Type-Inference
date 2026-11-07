"""Optimize the assistant policy against a fixed, independent human policy.

The independent human's Q table is loaded from the JSON output of
value_iteration.py; human value iteration is NOT rerun here. The Q table is
used to recover the deterministic human policy by taking argmax_a Q(s, a).
Only entries whose `initial_keys` matches the set of key types available in
this maze are used, matching the independent solver's all-keys-available run.

The output follows the same outer JSON structure as value_iteration.py:
maze / solver / goals / pddl_names, with utility, q_values, policy,
trajectory, pddl_plan, and related fields inside each goal entry.

Example:
  python fixed_human_value_iteration.py --human-q-values mazes/value_iteration_results.json
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Literal, Optional, Tuple

from value_iteration import (
    Maze, MOVES, WAIT_ACTION, load_maze, parse_goal_spec, passable,
    _finite_or_none,
)

Pos = Tuple[int, int]
Turn = Literal["human", "assistant"]
# (posH, keysH, gemsH, posA, keysA, unlocked, turn)
JointState = Tuple[Pos, FrozenSet[str], FrozenSet[str], Pos, FrozenSet[str], FrozenSet[str], Turn]
JointQTable = Dict[JointState, Dict[str, float]]

HUMAN_WAIT_COST = 0.1
ASSISTANT_WAIT_COST = 0.1


def initial_joint_state(maze: Maze) -> JointState:
    if maze.robot_start is None:
        raise ValueError("Maze must contain assistant/robot start 'm'")
    return (maze.human_start, frozenset(), frozenset(), maze.robot_start,
            frozenset(), frozenset(), "human")


def is_joint_goal(state: JointState, goal_h: FrozenSet[str]) -> bool:
    return goal_h.issubset(state[2])


def agent_successors(
    maze: Maze, pos: Pos, keys_held: FrozenSet[str], gems_held: FrozenSet[str],
    unlocked: FrozenSet[str], other_pos: Pos, other_keys_held: FrozenSet[str],
    allow_gem_pickup: bool,
) -> Iterable[Tuple[str, Pos, FrozenSet[str], FrozenSet[str], FrozenSet[str], float]]:
    """Legal non-wait actions for one agent; every real action costs 1."""
    x, y = pos
    for name, (dx, dy) in MOVES.items():
        nxt = (x + dx, y + dy)
        if nxt != other_pos and passable(maze, nxt, unlocked):
            yield name, nxt, keys_held, gems_held, unlocked, 1.0

    # A key cannot be picked up if either agent already holds it, or if it was
    # consumed by unlocking a door of the same color.
    for kid, kpos in maze.keys.items():
        color = kid.split(":", 1)[1]
        key_consumed = any(door_id == f"door:{color}" for door_id in unlocked)
        if (kpos == pos and kid not in keys_held and kid not in other_keys_held
                and not key_consumed):
            yield f"pickup({kid})", pos, keys_held | {kid}, gems_held, unlocked, 1.0
            break

    if allow_gem_pickup:
        for gid, (gpos, _) in maze.gems.items():
            if gpos == pos and gid not in gems_held:
                yield f"pickup({gid})", pos, keys_held, gems_held | {gid}, unlocked, 1.0
                break

    for door_id, (dpos, color) in maze.doors.items():
        if door_id in unlocked or abs(dpos[0] - x) + abs(dpos[1] - y) != 1:
            continue
        kid = f"key:{color}"
        if kid in keys_held:
            yield (f"unlock({kid},{door_id})", pos, keys_held - {kid}, gems_held,
                   unlocked | {door_id}, 1.0)


def human_q_state_key(
    pos: Pos,
    keys_held: FrozenSet[str],
    unlocked: FrozenSet[str],
    gems_held: FrozenSet[str],
    initial_keys: FrozenSet[str] = frozenset(),
) -> str:
    """Serialize a human state exactly like value_iteration.py's Q-table keys."""
    return json.dumps({
        "pos": list(pos),
        "keys": sorted(keys_held),
        "initial_keys": sorted(initial_keys),
        "unlocked": sorted(unlocked),
        "gems": sorted(gems_held),
    }, separators=(",", ":"))


def _goal_entry_from_json(data: dict, goal_label: str) -> dict:
    """Find the requested goal's result entry in a human value-iteration JSON."""
    goals = data.get("goals")
    if isinstance(goals, dict):
        candidates = [goal_label, f"H:{goal_label}", f"human:{goal_label}"]
        for key in candidates:
            entry = goals.get(key)
            if isinstance(entry, dict) and isinstance(entry.get("q_values"), dict):
                return entry
        matching = [v for k, v in goals.items()
                    if isinstance(v, dict) and isinstance(v.get("q_values"), dict)
                    and (str(k).lower().endswith(goal_label.lower())
                         or str(v.get("goal", "")).lower().endswith(goal_label.lower()))]
        if len(matching) == 1:
            return matching[0]
        entries = [v for v in goals.values()
                   if isinstance(v, dict) and isinstance(v.get("q_values"), dict)]
        if len(entries) == 1:
            return entries[0]
        raise ValueError(
            f"Could not identify human Q table for goal {goal_label!r}; "
            f"available goal entries: {list(goals)}"
        )
    if isinstance(data.get("q_values"), dict):
        return data
    raise ValueError("Input JSON does not contain a q_values table (directly or under 'goals').")


def load_fixed_human_policy(
    q_json_path: Path, goal_label: str, initial_keys: FrozenSet[str]
):
    """Load the independent human policy from its serialized Q table.

    The policy is recovered from Q values rather than trusting a serialized
    policy field. `initial_keys` identifies the independent solver's run for
    the key types available in this maze.
    """
    try:
        data = json.loads(q_json_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Human Q-table JSON not found: {q_json_path}") from exc
    entry = _goal_entry_from_json(data, goal_label)
    q_table = entry["q_values"]
    policy = {}
    skipped_initial_keys = 0
    expected_initial_keys = sorted(initial_keys)
    for serialized_state, action_values in q_table.items():
        try:
            raw_state = json.loads(serialized_state)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(raw_state, dict):
            continue
        # The independent solver serializes multiple key-availability variants
        # in one Q table. Select the all-key-types-available variant for this maze.
        if sorted(raw_state.get("initial_keys") or []) != expected_initial_keys:
            skipped_initial_keys += 1
            continue
        try:
            state = (
                tuple(raw_state["pos"]),
                frozenset(raw_state.get("keys", [])),
                frozenset(raw_state.get("unlocked", [])),
                frozenset(raw_state.get("gems", [])),
            )
        except (KeyError, TypeError):
            continue
        if not isinstance(action_values, dict) or not action_values:
            continue
        valid = [(action, value) for action, value in action_values.items()
                 if isinstance(value, (int, float)) and value == value]
        if valid:
            # Stable tie-breaking follows the action order in the source JSON.
            policy[state] = max(valid, key=lambda item: item[1])[0]
    if not policy:
        raise ValueError(
            f"No usable human states with initial_keys={expected_initial_keys} were found in {q_json_path}. "
            "Check that the Q table contains the all-key-types-available run for this maze."
        )
    return policy, len(q_table), skipped_initial_keys


def fixed_human_action(
    human_state: Tuple[Pos, FrozenSet[str], FrozenSet[str], FrozenSet[str]],
    human_policy: Dict[Tuple[Pos, FrozenSet[str], FrozenSet[str], FrozenSet[str]], str],
) -> str:
    """Look up the fixed human-only policy; wait if this state is not covered."""
    return human_policy.get(human_state, WAIT_ACTION)


def joint_successors_fixed_human(
    maze: Maze, state: JointState,
    human_policy: Dict[Tuple[Pos, FrozenSet[str], FrozenSet[str], FrozenSet[str]], str],
) -> Iterable[Tuple[str, JointState, float, float]]:
    """Yield (label, next state, human cost, assistant cost).

    Human turns have exactly one prescribed action, except that an illegal
    prescribed action is replaced with H:wait. Assistant turns remain choices.
    """
    pos_h, keys_h, gems_h, pos_a, keys_a, unlocked, turn = state
    if turn == "human":
        hs = (pos_h, keys_h, unlocked, gems_h)  # State layout in value_iteration.py
        prescribed = fixed_human_action(hs, human_policy)
        legal = list(agent_successors(
            maze, pos_h, keys_h, gems_h, unlocked, pos_a, keys_a, True
        ))
        match = next((item for item in legal if item[0] == prescribed), None)
        if match is None:
            yield ("H:wait", (pos_h, keys_h, gems_h, pos_a, keys_a, unlocked, "assistant"),
                   HUMAN_WAIT_COST, 0.0)
        else:
            act, npos, nkeys, ngems, nunlocked, cost = match
            yield (f"H:{act}", (npos, nkeys, ngems, pos_a, keys_a, nunlocked, "assistant"),
                   cost, 0.0)
    else:
        legal = list(agent_successors(
            maze, pos_a, keys_a, frozenset(), unlocked, pos_h, keys_h, False
        ))
        for act, npos, nkeys, _ngems, nunlocked, cost in legal:
            yield (f"A:{act}", (pos_h, keys_h, gems_h, npos, nkeys, nunlocked, "human"),
                   0.0, cost)
        yield ("A:wait", (pos_h, keys_h, gems_h, pos_a, keys_a, unlocked, "human"),
               0.0, ASSISTANT_WAIT_COST)


def explore_with_successors(maze: Maze, start: JointState, goal_h: FrozenSet[str], human_policy):
    seen = {start}
    queue = deque([start])
    succ_by_state = {}
    while queue:
        s = queue.popleft()
        if is_joint_goal(s, goal_h):
            succ_by_state[s] = []
            continue
        succ = list(joint_successors_fixed_human(maze, s, human_policy))
        succ_by_state[s] = succ
        for _, sp, _, _ in succ:
            if sp not in seen:
                seen.add(sp)
                queue.append(sp)
    return list(seen), succ_by_state


def joint_state_key(state: JointState) -> str:
    pos_h, keys_h, gems_h, pos_a, keys_a, unlocked, turn = state
    return json.dumps({
        "posH": list(pos_h), "keysH": sorted(keys_h), "gemsH": sorted(gems_h),
        "posA": list(pos_a), "keysA": sorted(keys_a),
        "unlocked": sorted(unlocked), "turn": turn,
    }, separators=(",", ":"))


def value_iteration_fixed_human(maze: Maze, start: JointState, goal_h: FrozenSet[str], human_policy,
                                max_iters: int = 20_000):
    states, succ_by_state = explore_with_successors(maze, start, goal_h, human_policy)
    goals = {s for s in states if is_joint_goal(s, goal_h)}
    V = {s: (0.0 if s in goals else float("-inf")) for s in states}
    for n_iters in range(1, max_iters + 1):
        changed = False
        new_v = dict(V)
        for s in states:
            if s in goals:
                new_v[s] = 0.0
                continue
            best = max((-(ch + ca) + V[sp] for _, sp, ch, ca in succ_by_state[s]),
                       default=float("-inf"))
            if best != V[s]:
                changed = True
            new_v[s] = best
        V = new_v
        if not changed:
            break
    else:
        raise RuntimeError(f"Fixed-human value iteration did not converge in {max_iters} iterations")

    Q: JointQTable = {}
    policy = {}
    for s in states:
        acts = {}
        if s not in goals:
            for act, sp, ch, ca in succ_by_state[s]:
                acts[act] = -(ch + ca) + V[sp]
            if acts:
                policy[s] = max(acts, key=acts.get)
        Q[s] = acts
    return V, Q, policy, n_iters, states, succ_by_state, goals


def evaluate_costs(states, policy, succ_by_state, goals, max_iters=20_000):
    inf = float("inf")
    human_cost = {s: (0.0 if s in goals else inf) for s in states}
    assistant_cost = dict(human_cost)
    chosen = {}
    for s in states:
        if s in goals or s not in policy:
            continue
        chosen[s] = next(((sp, ch, ca) for act, sp, ch, ca in succ_by_state[s]
                          if act == policy[s]), None)
    for _ in range(max_iters):
        changed = False
        for s, item in chosen.items():
            if item is None:
                continue
            sp, ch, ca = item
            nh = ch + human_cost.get(sp, inf)
            na = ca + assistant_cost.get(sp, inf)
            if nh != human_cost[s]: human_cost[s] = nh; changed = True
            if na != assistant_cost[s]: assistant_cost[s] = na; changed = True
        if not changed:
            break
    return human_cost, assistant_cost


def rollout(start, policy, succ_by_state, goals, max_steps=50_000):
    path, s, seen = [], start, set()
    for _ in range(max_steps):
        if s in goals:
            return path
        if s in seen or s not in policy:
            return []
        seen.add(s)
        act = policy[s]
        match = next((sp for a, sp, _, _ in succ_by_state[s] if a == act), None)
        if match is None:
            return []
        path.append((act, s, match))
        s = match
    return []


def to_pddl_action(maze: Maze, agent: str, action: str) -> str:
    if action == "wait": return f"(wait {agent})"
    if action in MOVES: return f"({action} {agent})"
    if action.startswith("pickup(") and action.endswith(")"):
        logical = action[len("pickup("):-1]
        return f"(pickup {agent} {maze.pddl_names[logical]})"
    if action.startswith("unlock(") and action.endswith(")"):
        key, door = action[len("unlock("):-1].split(",")
        return f"(unlock {agent} {maze.pddl_names[key]} {maze.pddl_names[door]})"
    raise ValueError(f"Unknown action: {action}")


def solve(maze: Maze, goal_spec: str, human_q_values_path: Path):
    label, goal_h = parse_goal_spec(goal_spec)
    missing = [g for g in goal_h if g not in maze.gems]
    if missing:
        raise ValueError(f"Goal gem(s) not present in maze: {missing}")
    if maze.robot_start is None:
        raise ValueError("Maze must contain assistant/robot start 'm'")

    # Read the independent human's serialized Q table and derive its fixed
    # deterministic policy by argmax. Do not run human value iteration again.
    initial_keys = frozenset(maze.keys.keys())
    human_policy, human_q_entries, skipped_initial_keys = load_fixed_human_policy(
        human_q_values_path, label, initial_keys
    )
    start = initial_joint_state(maze)
    V, Q, policy, n_iters, states, succ_by_state, goals = value_iteration_fixed_human(
        maze, start, goal_h, human_policy
    )
    HC, AC = evaluate_costs(states, policy, succ_by_state, goals)
    path = rollout(start, policy, succ_by_state, goals)
    combined_cost = -V[start] if V[start] > float("-inf") else float("inf")

    trajectory, pddl_plan = [], []
    if combined_cost != float("inf") and (start in goals or (path and path[-1][2] in goals)):
        def entry(t, s, action=None, pddl_action=None):
            ph, kh, gh, pa, ka, unlocked, turn = s
            return {
                "t": t, "x": ph[0], "y": ph[1], "rx": pa[0], "ry": pa[1],
                "turn": turn, "utility": _finite_or_none(V[s]),
                "human_cost_to_go": _finite_or_none(HC[s]),
                "assistant_cost_to_go": _finite_or_none(AC[s]),
                "q_values": {a: _finite_or_none(q) for a, q in Q[s].items()},
                "action": action, "pddl_action": pddl_action,
            }
        trajectory = [entry(0, start)]
        for i, (act, _before, after) in enumerate(path, 1):
            agent_tag, raw = act.split(":", 1)
            agent = "human" if agent_tag == "H" else "robot"
            pddl = to_pddl_action(maze, agent, raw)
            pddl_plan.append(pddl)
            e = entry(i, after, act, pddl)
            e["human_step_cost"] = 1.0 if agent_tag == "H" and raw != "wait" else (HUMAN_WAIT_COST if agent_tag == "H" else 0.0)
            e["assistant_step_cost"] = 1.0 if agent_tag == "A" and raw != "wait" else (ASSISTANT_WAIT_COST if agent_tag == "A" else 0.0)
            e["step_cost"] = e["human_step_cost"] + e["assistant_step_cost"]
            trajectory.append(e)

    V_out = {joint_state_key(s): v for s, v in V.items() if v > float("-inf")}
    Q_out = {joint_state_key(s): {a: _finite_or_none(q) for a, q in acts.items()} for s, acts in Q.items()}
    pi_out = {joint_state_key(s): a for s, a in policy.items()}
    return {
        "goal": f"H:{label}", "human_goal_gems": sorted(goal_h),
        "solver": "fixed_human_policy_assistant_value_iteration",
        "human_policy_source": str(human_q_values_path),
        "human_policy_recovered_from": "argmax of independent human q_values",
        "human_q_table_entries": human_q_entries,
        "human_q_entries_skipped_nonempty_initial_keys": skipped_initial_keys,
        "n_value_iteration_sweeps": n_iters,
        "objective": "minimize combined human + assistant action cost with the human's single-agent policy fixed",
        "human_model": "deterministic policy computed in the single-agent environment; it does not re-optimize in response to assistant actions",
        "assistant_key_behavior": "learned by value iteration; key pickup is not hard-coded as either required or forbidden",
        "human_policy_legality_fallback": "if the prescribed human action is physically illegal in the joint state, human waits; the human policy is not re-optimized",
        "path_cost": _finite_or_none(combined_cost),
        "human_path_cost": _finite_or_none(HC.get(start, float("inf"))),
        "assistant_path_cost": _finite_or_none(AC.get(start, float("inf"))),
        "path": [a for a, _, _ in path], "trajectory": trajectory, "pddl_plan": pddl_plan,
        "n_states_with_finite_value": len(V_out), "n_policy_entries": len(pi_out),
        "n_q_entries": sum(len(a) for a in Q.values()),
        "utility": V_out, "q_values": Q_out, "policy": pi_out,
        "start_utility": _finite_or_none(V[start]),
        "start_human_cost": _finite_or_none(HC.get(start, float("inf"))),
        "start_assistant_cost": _finite_or_none(AC.get(start, float("inf"))),
        "start_q_values": {a: _finite_or_none(q) for a, q in Q[start].items()},
        "pddl_names": dict(maze.pddl_names),
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize assistant policy with fixed human-only policy")
    parser.add_argument("--maze", type=Path, default=Path(__file__).resolve().parent / "mazes" / "sample.txt")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "mazes" / "fixed_human_value_iteration_results.json")
    parser.add_argument("--human-goal", default="both")
    parser.add_argument(
        "--human-q-values", type=Path,
        default=Path(__file__).resolve().parent / "mazes" / "value_iteration_results.json",
        help="JSON output from the independent human value_iteration.py solver",
    )
    args = parser.parse_args()
    maze = load_maze(args.maze)
    sol = solve(maze, args.human_goal, args.human_q_values)
    results = {"maze": str(args.maze), "solver": sol["solver"], "goals": {sol["goal"]: sol}, "pddl_names": maze.pddl_names}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Loaded maze: {args.maze} ({maze.width}x{maze.height})")
    print(f"Human start: {maze.human_start}; assistant start: {maze.robot_start}")
    print(f"Goal: {sol['goal']}; combined cost: {sol['path_cost']}")
    print(f"  human cost: {sol['human_path_cost']}; assistant cost: {sol['assistant_path_cost']}")
    print(f"Loaded human Q table: {args.human_q_values}")
    print(f"Human policy states recovered: {len(load_fixed_human_policy(args.human_q_values, parse_goal_spec(args.human_goal)[0], frozenset(maze.keys.keys()))[0])}")
    print(f"Assistant VI sweeps: {sol['n_value_iteration_sweeps']}; Q entries: {sol['n_q_entries']}")
    print("Path:")
    for i, act in enumerate(sol["path"], 1): print(f"  {i:3d}. {act}")
    print(f"Wrote results to {args.out}")

if __name__ == "__main__":
    main()
