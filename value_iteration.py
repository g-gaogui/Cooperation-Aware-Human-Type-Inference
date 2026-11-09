"""Independent-human value iteration on a Doors–Keys–Gems maze.

Solves every 2^|K| key-availability scenario. Utilities are negative
cost-to-go. State keys include `initial_keys` so scenarios do not collide.
The all-keys-available scenario supplies the reported path and policy.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

# 1-based coordinates; y increases downward.
MOVES = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

KEY_CHARS = {"r": "red", "y": "yellow", "b": "blue", "e": "green", "p": "pink"}
DOOR_CHARS = {"R": "red", "Y": "yellow", "B": "blue", "E": "green", "P": "pink"}
GEM_COLOR_ORDER = ["red", "yellow", "blue", "green"]  # gem1..gem4, as in ascii.jl

Pos = Tuple[int, int]  # (x, y)
State = Tuple[Pos, FrozenSet[str], FrozenSet[str], FrozenSet[str]]  # pos, keys, unlocked, gems
QTable = Dict[State, Dict[str, float]]

# Utility assigned when no requested goal gem is reachable.
UNREACHABLE_VALUE = -1_000_000.0
WAIT_ACTION = "wait"


@dataclass
class Maze:
    width: int
    height: int
    walls: FrozenSet[Pos]
    human_start: Pos
    robot_start: Optional[Pos]
    keys: Dict[str, Pos]  # key_id -> position, key_id like "key:red"
    doors: Dict[str, Tuple[Pos, str]]  # door_id -> (pos, color)
    gems: Dict[str, Tuple[Pos, str]]  # gem_id -> (pos, color)
    ascii_rows: List[str]
    pddl_names: Dict[str, str]  # logical id -> PDDL name, ascii.jl order

    def in_bounds(self, pos: Pos) -> bool:
        x, y = pos
        return 1 <= x <= self.width and 1 <= y <= self.height

    def is_wall(self, pos: Pos) -> bool:
        return pos in self.walls

    def door_at(self, pos: Pos) -> Optional[str]:
        for door_id, (dpos, _) in self.doors.items():
            if dpos == pos:
                return door_id
        return None

    def key_at(self, pos: Pos, keys_held: FrozenSet[str]) -> Optional[str]:
        for key_id, kpos in self.keys.items():
            if kpos == pos and key_id not in keys_held:
                return key_id
        return None

    def gem_at(self, pos: Pos, gems_held: FrozenSet[str]) -> Optional[str]:
        for gem_id, (gpos, _) in self.gems.items():
            if gpos == pos and gem_id not in gems_held:
                return gem_id
        return None


def load_maze(path: Path) -> Maze:
    rows: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append(s)
    if not rows:
        raise ValueError(f"Empty maze: {path}")

    height = len(rows)
    width = max(len(r) for r in rows)
    rows = [r.ljust(width) for r in rows]

    walls: set[Pos] = set()
    keys: Dict[str, Pos] = {}
    doors: Dict[str, Tuple[Pos, str]] = {}
    gems: Dict[str, Tuple[Pos, str]] = {}
    pddl_names: Dict[str, str] = {}
    human_start: Optional[Pos] = None
    robot_start: Optional[Pos] = None
    gem_count = 0
    key_count = 0
    door_count = 0

    for y, row in enumerate(rows, start=1):
        for x, ch in enumerate(row, start=1):
            pos = (x, y)
            if ch == "W":
                walls.add(pos)
            elif ch == "h":
                human_start = pos
            elif ch == "m":
                robot_start = pos
            elif ch in KEY_CHARS:
                color = KEY_CHARS[ch]
                logical = f"key:{color}"
                key_count += 1
                keys[logical] = pos
                pddl_names[logical] = f"key{key_count}"
            elif ch in DOOR_CHARS:
                color = DOOR_CHARS[ch]
                logical = f"door:{color}"
                door_count += 1
                doors[logical] = (pos, color)
                pddl_names[logical] = f"door{door_count}"
            elif ch in ("g", "G"):
                gem_count += 1
                color = GEM_COLOR_ORDER[gem_count - 1]
                logical = f"gem:{color}"
                gems[logical] = (pos, color)
                pddl_names[logical] = f"gem{gem_count}"
            elif ch in (".", " "):
                pass
            else:
                raise ValueError(f"Unknown maze char {ch!r} at {pos}")

    if human_start is None:
        raise ValueError("Maze must contain human 'h'")
    if "gem:red" not in gems or "gem:yellow" not in gems:
        raise ValueError(
            "Maze must contain at least two gems (gem1=red, gem2=yellow), "
            f"found: {list(gems)}"
        )

    return Maze(
        width=width,
        height=height,
        walls=frozenset(walls),
        human_start=human_start,
        robot_start=robot_start,
        keys=keys,
        doors=doors,
        gems=gems,
        ascii_rows=rows,
        pddl_names=pddl_names,
    )


def initial_state(maze: Maze) -> State:
    return (maze.human_start, frozenset(), frozenset(), frozenset())


def is_goal(state: State, goal_gems: FrozenSet[str]) -> bool:
    """True iff every gem in `goal_gems` is held."""
    _, _, _, gems_held = state
    return goal_gems.issubset(gems_held)


def passable(maze: Maze, pos: Pos, unlocked: FrozenSet[str]) -> bool:
    if not maze.in_bounds(pos) or maze.is_wall(pos):
        return False
    door_id = maze.door_at(pos)
    if door_id is not None and door_id not in unlocked:
        return False
    return True


def successors(
    maze: Maze,
    state: State,
    unavailable_keys: FrozenSet[str] = frozenset(),
    wait_penalty: float = 0.1,
) -> Iterable[Tuple[str, State, float]]:
    """
    Yield (action, next_state, cost). Human-only; robot ignored.

    `unavailable_keys` restricts which key types can be picked up.
    `wait_penalty` is the cost of the human waiting in place.
    """
    (x, y), keys_held, unlocked, gems_held = state
    pos = (x, y)

    for name, (dx, dy) in MOVES.items():
        nxt = (x + dx, y + dy)
        if passable(maze, nxt, unlocked):
            yield name, (nxt, keys_held, unlocked, gems_held), 1.0

    key_id = None
    for kid, kpos in maze.keys.items():
        if (
            kpos == pos
            and kid not in keys_held
            and kid not in unavailable_keys
        ):
            key_id = kid
            break

    if key_id is not None:
        yield (
            f"pickup({key_id})",
            (pos, keys_held | {key_id}, unlocked, gems_held),
            1.0,
        )

    gem_id = None
    for gid, (gpos, _) in maze.gems.items():
        if gpos == pos and gid not in gems_held:
            gem_id = gid
            break

    if gem_id is not None:
        yield (
            f"pickup({gem_id})",
            (pos, keys_held, unlocked, gems_held | {gem_id}),
            1.0,
        )

    for door_id, (dpos, color) in maze.doors.items():
        if door_id in unlocked:
            continue

        dx = abs(dpos[0] - x)
        dy = abs(dpos[1] - y)
        if dx + dy != 1:
            continue

        key_id = f"key:{color}"
        if key_id not in keys_held:
            continue

        yield (
            f"unlock({key_id},{door_id})",
            (pos, keys_held - {key_id}, unlocked | {door_id}, gems_held),
            1.0,
        )

    # Wait: remain in the same state
    yield WAIT_ACTION, state, wait_penalty

def parse_goal_spec(spec: str) -> Tuple[str, FrozenSet[str]]:
    """
    Parse a goal name into (label, frozenset of logical gem ids).

    Examples:
      "red"            -> ("red", {gem:red})
      "yellow"         -> ("yellow", {gem:yellow})
      "both"           -> ("both", {gem:red, gem:yellow})
      "red+yellow"     -> ("red+yellow", {gem:red, gem:yellow})
    """
    s = spec.strip().lower()
    if s in ("both", "all", "red+yellow", "yellow+red"):
        return "both", frozenset({"gem:red", "gem:yellow"})
    if s in ("red", "yellow", "blue", "green"):
        return s, frozenset({f"gem:{s}"})
    parts = [p.strip() for p in s.replace("+", ",").split(",") if p.strip()]
    gems = frozenset(f"gem:{p}" if not p.startswith("gem:") else p for p in parts)
    if not gems:
        raise ValueError(f"Empty goal spec: {spec!r}")
    label = "+".join(sorted(g.split(":", 1)[1] for g in gems))
    return label, gems


def state_key(state: State, initial_keys: Optional[FrozenSet[str]] = None) -> str:
    """
    JSON key for a state. When `initial_keys` is given, an "initial_keys"
    field is included recording which key types existed/were obtainable
    in the scenario that produced this entry — this is what lets entries
    from different 2^|keys| combinations coexist without colliding even
    when (pos, keys_held, unlocked, gems_held) matches. When omitted, the
    output is identical to the original (pre-sweep) format.
    """
    pos, keys_held, unlocked, gems_held = state
    d: Dict[str, object] = {
        "pos": list(pos),
        "keys": sorted(keys_held),
    }
    if initial_keys is not None:
        d["initial_keys"] = sorted(initial_keys)
    d["unlocked"] = sorted(unlocked)
    d["gems"] = sorted(gems_held)
    return json.dumps(d, separators=(",", ":"))


def parse_state_key(s: str) -> State:
    d = json.loads(s)
    return (
        (d["pos"][0], d["pos"][1]),
        frozenset(d["keys"]),
        frozenset(d["unlocked"]),
        frozenset(d["gems"]),
    )


def explore_reachable(
    maze: Maze,
    start: State,
    unavailable_keys: FrozenSet[str] = frozenset(),
) -> List[State]:
    seen = {start}
    q = deque([start])
    while q:
        s = q.popleft()
        for _, sp, _ in successors(maze, s, unavailable_keys):
            if sp not in seen:
                seen.add(sp)
                q.append(sp)
    return list(seen)


def find_reachable_gems(
    maze: Maze,
    start: State,
    unavailable_keys: FrozenSet[str],
) -> FrozenSet[str]:
    """All gems that can ever be held in some reachable state, given that
    key types in `unavailable_keys` can never be picked up."""
    reachable_gems: set[str] = set()
    for s in explore_reachable(maze, start, unavailable_keys):
        reachable_gems.update(s[3])
    return frozenset(reachable_gems)


def all_key_combos(maze: Maze) -> List[Tuple[FrozenSet[str], FrozenSet[str]]]:
    """
    Enumerate all 2^|key types| combinations of which key types exist /
    are obtainable in the world.

    Returns a list of (available_keys, unavailable_keys) pairs where
    available_keys | unavailable_keys == the full set of key types in
    the maze, for every possible subset.
    """
    key_types = sorted(maze.keys.keys())
    n = len(key_types)
    all_keys = frozenset(key_types)
    combos: List[Tuple[FrozenSet[str], FrozenSet[str]]] = []
    for mask in range(2**n):
        available = frozenset(key_types[i] for i in range(n) if (mask >> i) & 1)
        combos.append((available, all_keys - available))
    return combos


def value_iteration(
    maze: Maze,
    start: State,
    goal_gems: FrozenSet[str],
    unavailable_keys: FrozenSet[str] = frozenset(),
    max_iters: int = 10_000,
) -> Tuple[Dict[State, float], QTable, Dict[State, str], int]:
    """
    Synchronous value iteration on reachable states.

    Reward is -1 per action; terminal (goal) states have V=0.
    Equivalently V(s) = -cost-to-go(s).

    Q(s,a) = -cost(s,a) + V(s')
    pi(s)   = argmax_a Q(s,a)

    `unavailable_keys` restricts which key types can ever be picked up
    (see `successors`), so this can be used to compute the value
    function under a specific key-availability scenario.
    """
    states = explore_reachable(maze, start, unavailable_keys)
    V: Dict[State, float] = {
        s: 0.0 if is_goal(s, goal_gems) else float("-inf") for s in states
    }
    wait_penalty = 0.1  # small penalty to avoid infinite loops of waiting
    n_iters = 0
    for n_iters in range(1, max_iters + 1):
        changed = False
        V_new = dict(V)
        for s in states:
            if is_goal(s, goal_gems):
                V_new[s] = 0.0
                continue
            best = float("-inf")
            for _, sp, cost in successors(maze, s, unavailable_keys, wait_penalty):
                if sp not in V:
                    continue
                cand = -cost + V[sp]
                if cand > best:
                    best = cand
            if best != V[s]:
                changed = True
            V_new[s] = best
        V = V_new
        if not changed:
            break
    else:
        raise RuntimeError(f"Value iteration did not converge in {max_iters} iterations")

    Q: QTable = {}
    policy: Dict[State, str] = {}
    for s in states:
        acts: Dict[str, float] = {}
        if is_goal(s, goal_gems):
            Q[s] = acts
            continue
        best_act = None
        best_q = float("-inf")
        for act, sp, cost in successors(maze, s, unavailable_keys, wait_penalty):
            if sp not in V:
                continue
            q = -cost + V[sp]
            acts[act] = q
            if q > best_q:
                best_q = q
                best_act = act
        Q[s] = acts
        if best_act is not None and best_q > float("-inf"):
            policy[s] = best_act

    return V, Q, policy, n_iters


def greedy_path(
    maze: Maze,
    start: State,
    goal_gems: FrozenSet[str],
    policy: Dict[State, str],
    V: Dict[State, float],
    unavailable_keys: FrozenSet[str] = frozenset(),
    max_steps: int = 10_000,
) -> Tuple[List[str], float]:
    """Follow pi from start until a goal state. Cost is -V(start) when finite."""
    path: List[str] = []
    s = start
    seen: set[State] = set()
    for _ in range(max_steps):
        if is_goal(s, goal_gems):
            v0 = V.get(start, float("-inf"))
            cost = -v0 if v0 > float("-inf") else float("inf")
            return path, cost
        if s in seen:
            break
        seen.add(s)
        act = policy.get(s)
        if act is None:
            break
        nxt = None
        for a, sp, _ in successors(maze, s, unavailable_keys):
            if a == act:
                nxt = sp
                break
        if nxt is None:
            break
        path.append(act)
        s = nxt
    return [], float("inf")


def to_pddl_action(maze: Maze, action: str) -> str:
    """Map an internal action string to a PDDL ground action for the human."""
    if action in MOVES:
        return f"({action} human)"
    if action.startswith("pickup(") and action.endswith(")"):
        logical = action[len("pickup(") : -1]
        return f"(pickup human {maze.pddl_names[logical]})"
    if action.startswith("unlock(") and action.endswith(")"):
        inner = action[len("unlock(") : -1]
        key_logical, door_logical = inner.split(",")
        return (
            f"(unlock human {maze.pddl_names[key_logical]} "
            f"{maze.pddl_names[door_logical]})"
        )
    if action == WAIT_ACTION:
        return "(wait human)"
    raise ValueError(f"Unknown action for PDDL export: {action}")


def build_trajectory(
    maze: Maze,
    start: State,
    path: List[str],
    V: Dict[State, float],
    Q: QTable,
    unavailable_keys: FrozenSet[str] = frozenset(),
) -> Tuple[List[dict], List[str]]:
    """
    Roll out `path` from `start`.

    Returns:
      trajectory: one entry per human state (including start), with V and Q
      pddl_plan: human actions interleaved with (wait robot) for PDDL turn-taking
    """
    trajectory: List[dict] = []
    pddl_plan: List[str] = []

    s = start
    (x, y), _, _, _ = s
    trajectory.append(
        {
            "t": 0,
            "x": x,
            "y": y,
            "utility": V.get(s, float("-inf")),
            "q_values": dict(Q.get(s, {})),
            "action": None,
            "pddl_action": None,
        }
    )

    for i, act in enumerate(path, start=1):
        nxt = None
        for a, sp, _ in successors(maze, s, unavailable_keys):
            if a == act:
                nxt = sp
                break
        if nxt is None:
            raise RuntimeError(f"Action {act!r} not applicable at state {s}")

        pddl_act = to_pddl_action(maze, act)
        pddl_plan.append(pddl_act)
        pddl_plan.append("(wait robot)")  # preserve human/robot turn order

        s = nxt
        (x, y), _, _, _ = s
        trajectory.append(
            {
                "t": i,
                "x": x,
                "y": y,
                "utility": V.get(s, float("-inf")),
                "q_values": dict(Q.get(s, {})),
                "action": act,
                "pddl_action": pddl_act,
            }
        )

    return trajectory, pddl_plan


def format_policy_on_grid(
    maze: Maze, policy: Dict[State, str], start: State, goal_gems: FrozenSet[str]
) -> str:
    """
    Show arrows for states that match start's inventory (keys/doors/gems),
    i.e. a 2D slice of the policy at the initial inventory.
    """
    _, keys0, unlocked0, gems0 = start
    arrow = {"up": "^", "down": "v", "left": "<", "right": ">"}
    grid = [list(row) for row in maze.ascii_rows]
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            if ch in ("h", "m"):
                grid[y][x] = "."

    for s, act in policy.items():
        (x, y), keys_held, unlocked, gems_held = s
        if keys_held != keys0 or unlocked != unlocked0 or gems_held != gems0:
            continue
        if act not in arrow:
            continue
        if maze.is_wall((x, y)):
            continue
        grid[y - 1][x - 1] = arrow[act]

    sx, sy = maze.human_start
    grid[sy - 1][sx - 1] = "H"
    for g in goal_gems:
        gx, gy = maze.gems[g][0]
        grid[gy - 1][gx - 1] = "G"
    return "\n".join("".join(r) for r in grid)


def _finite_or_none(v: float) -> Optional[float]:
    if v == float("inf") or v == float("-inf"):
        return None
    return float(v)


def serialize_v(
    V: Dict[State, float], initial_keys: Optional[FrozenSet[str]] = None
) -> Dict[str, float]:
    return {
        state_key(s, initial_keys): v for s, v in V.items() if v > float("-inf")
    }


def serialize_q(
    Q: QTable, initial_keys: Optional[FrozenSet[str]] = None
) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for s, acts in Q.items():
        out[state_key(s, initial_keys)] = {
            a: _finite_or_none(q) for a, q in acts.items()
        }
    return out


def solve_goal(maze: Maze, goal_spec: str) -> dict:
    """
    Solve for a goal specification, sweeping over all 2^|key types|
    key-availability combinations (see module docstring for details).

    `goal_spec` examples: "red", "yellow", "both" (red and yellow).
    """
    goal_label, goal_gems_requested = parse_goal_spec(goal_spec)
    for g in goal_gems_requested:
        if g not in maze.gems:
            raise ValueError(f"Goal gem {g} not present in maze (have {list(maze.gems)})")

    start = initial_state(maze)
    all_keys = frozenset(maze.keys.keys())

    V_out: Dict[str, float] = {}
    Q_out: Dict[str, Dict[str, Optional[float]]] = {}
    primary: Optional[dict] = None  # data from the all-keys-available combination

    for available_keys, unavailable_keys in all_key_combos(maze):
        reachable_gems = find_reachable_gems(maze, start, unavailable_keys)
        scenario_goal_gems = frozenset(
            g for g in goal_gems_requested if g in reachable_gems
        )
        is_baseline = available_keys == all_keys

        if not scenario_goal_gems:
            # None of the goal gems are reachable under this combination:
            # skip VI, fall back to a sentinel utility + a single "wait"
            # action for every state reached under it.
            states = explore_reachable(maze, start, unavailable_keys)
            V_s: Dict[State, float] = {s: UNREACHABLE_VALUE for s in states}
            Q_s: QTable = {s: {WAIT_ACTION: UNREACHABLE_VALUE} for s in states}
            policy_s: Dict[State, str] = {}
            n_iters_s = 0
        else:
            V_s, Q_s, policy_s, n_iters_s = value_iteration(
                maze, start, scenario_goal_gems, unavailable_keys=unavailable_keys
            )

        V_out.update(serialize_v(V_s, available_keys))
        Q_out.update(serialize_q(Q_s, available_keys))

        if is_baseline:
            primary = {
                "V": V_s,
                "Q": Q_s,
                "policy": policy_s,
                "n_iters": n_iters_s,
                "goal_gems": scenario_goal_gems,
                "unavailable_keys": unavailable_keys,
            }

    assert primary is not None, "the all-keys-available combination must always run"
    V, Q, policy = primary["V"], primary["Q"], primary["policy"]
    goal_gems = primary["goal_gems"]
    unavailable_keys = primary["unavailable_keys"]

    path, path_cost = greedy_path(
        maze, start, goal_gems, policy, V, unavailable_keys=unavailable_keys
    )

    trajectory: List[dict] = []
    pddl_plan: List[str] = []
    if path_cost != float("inf"):
        trajectory, pddl_plan = build_trajectory(
            maze, start, path, V, Q, unavailable_keys=unavailable_keys
        )
        for step in trajectory:
            step["utility"] = _finite_or_none(step["utility"])
            step["q_values"] = {
                a: _finite_or_none(q) for a, q in step["q_values"].items()
            }

    n_q = sum(len(acts) for acts in Q_out.values())
    return {
        "goal": goal_label,
        "goal_gems": sorted(goal_gems),
        "goal_pddl_objects": [maze.pddl_names[g] for g in sorted(goal_gems)],
        "solver": "value_iteration",
        "n_value_iteration_sweeps": primary["n_iters"],
        "path_cost": _finite_or_none(path_cost),
        "path": path,
        "trajectory": trajectory,
        "pddl_plan": pddl_plan,
        "n_states_with_finite_value": len(V_out),
        "n_policy_entries": len(policy),
        "n_q_entries": n_q,
        "utility": V_out,
        "q_values": Q_out,
        "policy": {state_key(s): a for s, a in policy.items()},
        "policy_grid_slice_initial_inventory": format_policy_on_grid(
            maze, policy, start, goal_gems
        ),
        "start_utility": _finite_or_none(V.get(start, float("-inf"))),
        "start_q_values": {
            a: _finite_or_none(q) for a, q in Q.get(start, {}).items()
        },
        "pddl_names": dict(maze.pddl_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Value iteration human policy and Q-values for maze gems"
    )
    parser.add_argument(
        "--maze",
        type=Path,
        default=Path(__file__).resolve().parent / "mazes" / "sample.txt",
        help="ASCII maze path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "mazes" / "value_iteration_results.json",
        help="JSON output path",
    )
    parser.add_argument(
        "--goals",
        nargs="+",
        default=["both"],
        help=(
            "Goal specs to solve. "
            "Use 'both' for red+yellow, or 'red' / 'yellow' for single gems. "
            "Default: both"
        ),
    )
    args = parser.parse_args()

    maze = load_maze(args.maze)
    print(f"Loaded maze {args.maze} ({maze.width}x{maze.height})")
    print("ASCII:")
    print("\n".join(maze.ascii_rows))
    print(f"Human start: {maze.human_start}")
    print(f"Keys: {maze.keys}")
    print(f"Doors: {maze.doors}")
    print(f"Gems: {maze.gems}")
    key_types = sorted(maze.keys.keys())
    print(
        f"Key types: {key_types} -> sweeping {2 ** len(key_types)} "
        "key-availability combinations per goal"
    )
    print()

    results = {
        "maze": str(args.maze),
        "solver": "value_iteration",
        "goals": {},
    }

    for spec in args.goals:
        label, goal_gems = parse_goal_spec(spec)
        print("=" * 60)
        print(f"Goal: collect {sorted(goal_gems)}")
        print("=" * 60)

        sol = solve_goal(maze, spec)
        results["goals"][label] = sol

        if sol["path_cost"] is None:
            print("No path found.")
            continue

        print(f"Value-iteration sweeps: {sol['n_value_iteration_sweeps']}")
        print(f"Optimal path cost: {sol['path_cost']}")
        print(f"Start utility V(s0): {sol['start_utility']}")
        print(f"Start Q(s0, a): {sol['start_q_values']}")
        print(f"Reachable states with finite V (all key combos): {sol['n_states_with_finite_value']}")
        print(f"Policy entries (baseline, all keys available): {sol['n_policy_entries']}")
        print(f"Q entries (state-action pairs, all key combos): {sol['n_q_entries']}")
        print("Optimal action sequence:")
        for i, a in enumerate(sol["path"], 1):
            print(f"  {i:3d}. {a}")
        print("Trajectory (t, x, y, V, Q, action):")
        for step in sol["trajectory"]:
            print(
                f"  t={step['t']:2d}  ({step['x']},{step['y']})  "
                f"V={step['utility']}  Q={step['q_values']}  {step['action']}"
            )
        print("\nPolicy arrows (slice at initial inventory; H=start, G=goal gem):")
        print(sol["policy_grid_slice_initial_inventory"])
        print()

    results["pddl_names"] = maze.pddl_names

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote V, Q, policy, and trajectories to {args.out}")


if __name__ == "__main__":
    main()