"""Belief-aware assistant planning with Real-Time Dynamic Programming (RTDP).

Two subcommands:

  train  Learn one belief-conditioned assistant policy with RTDP. Writes Q-values,
         utilities and the greedy policy. No rollouts (no path / trajectory).

  test   Load a trained policy and roll it out twice: once against an
         independent human, once against a collaborative human. Writes one JSON
         per human type with path / trajectory / pddl_plan. No Q-values or
         utilities.

Train inputs:
  --independent-human-q-values: single-human value_iteration.py output
  --joint-q-values: joint human+assistant value_iteration.py output
  --assistant-independent-q-values: fixed_human_value_iteration.py output

Test inputs:
  --policy: output of `train`
  --independent-human-q-values, --joint-q-values: used to simulate the human

The human type is fixed in the environment but unknown to the assistant.
The assistant maintains a Bayesian belief over human type. Human action
likelihoods come from the independent and collaborative human Q tables.

Place beside value_iteration.py and joint_value_iteration.py.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from value_iteration import load_maze, parse_goal_spec, _finite_or_none
from joint_value_iteration import (
    JointState,
    initial_joint_state,
    joint_successors,
    joint_state_key,
    is_goal as is_joint_goal,
    to_pddl_action,
)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_goal_solution(data: dict, goal_label: str) -> dict:
    """Accept goal keys such as 'both', 'H:both', or a single-goal file."""
    goals = data.get("goals", {})
    for key in (goal_label, f"H:{goal_label}", f"human:{goal_label}"):
        value = goals.get(key) if isinstance(goals, dict) else None
        if isinstance(value, dict) and isinstance(value.get("q_values"), dict):
            return value
    if isinstance(goals, dict):
        candidates = [v for v in goals.values() if isinstance(v, dict) and isinstance(v.get("q_values"), dict)]
        if len(candidates) == 1:
            return candidates[0]
    if isinstance(data.get("q_values"), dict):
        return data
    raise KeyError(f"Could not find Q table for goal {goal_label!r}; available goals: {list(goals) if isinstance(goals, dict) else []}")


def q_table(data: dict, goal_label: str) -> dict:
    return get_goal_solution(data, goal_label)["q_values"]


def state_key_dict(state: JointState) -> dict:
    return json.loads(joint_state_key(state))


def independent_human_key(state: JointState, maze) -> str:
    """Project the joint state into the single-human Q-table key format."""
    pos_h, keys_h, gems_h, _pos_a, keys_a, unlocked, _turn = state
    all_key_types = frozenset(maze.keys.keys())
    keys_from_unlocked = {f"key:{door.split(':', 1)[1]}" for door in unlocked}
    initial_keys = (all_key_types - keys_a) | keys_from_unlocked
    return json.dumps({
        "pos": list(pos_h),
        "keys": sorted(keys_h),
        "initial_keys": sorted(initial_keys),
        "unlocked": sorted(unlocked),
        "gems": sorted(gems_h),
    }, separators=(",", ":"))


def boltzmann(q_values: dict, beta: float) -> dict:
    finite = {a: float(q) for a, q in q_values.items() if q is not None and math.isfinite(float(q))}
    if not finite:
        return {}
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if beta == 0:
        return {a: 1.0 / len(finite) for a in finite}
    m = max(beta * q for q in finite.values())
    exp_q = {a: math.exp(beta * q - m) for a, q in finite.items()}
    z = sum(exp_q.values())
    return {a: v / z for a, v in exp_q.items()}


def normalize(dist: dict) -> dict:
    z = sum(max(0.0, float(v)) for v in dist.values())
    return {a: max(0.0, float(v)) / z for a, v in dist.items()} if z else {}


def legal_human_actions(maze, state: JointState) -> dict:
    return {
        action.split(":", 1)[1]: (next_state, human_cost, assistant_cost)
        for action, next_state, human_cost, assistant_cost in joint_successors(maze, state)
        if action.startswith("H:")
    }


def human_action_distributions(maze, state, independent_q, collaborative_q, beta):
    """Return P(action | independent) and P(action | collaborative)."""
    legal = legal_human_actions(maze, state)
    if not legal:
        return {}, {}
    qi = independent_q.get(independent_human_key(state, maze), {})
    qc = collaborative_q.get(joint_state_key(state), {})
    q_ind = {a: qi[a] for a in legal if a in qi}
    q_col = {a: qc[f"H:{a}"] for a in legal if f"H:{a}" in qc}
    pi, pc = boltzmann(q_ind, beta), boltzmann(q_col, beta)
    # Missing state entries should not make one hypothesis impossible.
    if not pi:
        pi = {a: 1.0 / len(legal) for a in legal}
    if not pc:
        pc = {a: 1.0 / len(legal) for a in legal}
    return pi, pc


def update_belief(p_i: float, p_action_i: float, p_action_c: float) -> float:
    denom = p_i * p_action_i + (1.0 - p_i) * p_action_c
    if denom <= 0.0:
        return p_i
    return min(1.0, max(0.0, p_i * p_action_i / denom))


def project_belief_on_grid(p_i: float, bins: int) -> float:
    """Project P(independent) to the nearest point on a uniform belief grid."""
    p_i = min(1.0, max(0.0, float(p_i)))
    return round(round(p_i * bins) / bins, 6)


def assistant_actions(maze, state):
    return [x for x in joint_successors(maze, state) if x[0].startswith("A:")]


def policy_key(state: JointState, pkey: float) -> str:
    """Key used for utility / q_values / policy entries in the train output."""
    key_obj = json.loads(joint_state_key(state))
    key_obj["p_independent"] = pkey
    key_obj["p_collaborative"] = round(1.0 - pkey, 6)
    return json.dumps(key_obj, separators=(",", ":"))


# --------------------------------------------------------------------------
# Training (RTDP)
# --------------------------------------------------------------------------

def load_assistant_q_tables(independent_assistant_data, joint_data, goal_label):
    q_i = q_table(independent_assistant_data, goal_label)
    q_c_all = q_table(joint_data, goal_label)
    # Keep assistant-turn states from the joint policy for the collaborative heuristic.
    q_c = {}
    for key, actions in q_c_all.items():
        try:
            state = json.loads(key)
        except (TypeError, json.JSONDecodeError):
            continue
        if state.get("turn") == "assistant":
            q_c[key] = actions
    return q_i, q_c


def heuristic_q(state, action, p_i, q_i, q_c):
    """Belief-weighted Q estimate for one assistant action, if available."""
    key = joint_state_key(state)
    vi = q_i.get(key, {}).get(action)
    vc = q_c.get(key, {}).get(action)
    if vi is None and vc is None:
        return None
    if vi is None:
        vi = vc
    if vc is None:
        vc = vi
    vi, vc = float(vi), float(vc)
    if not (math.isfinite(vi) and math.isfinite(vc)):
        return None
    return p_i * vi + (1.0 - p_i) * vc


def heuristic_v(state, p_i, q_i, q_c):
    """Initialize an unseen assistant-turn state using a shared action.

    For each action represented in either assistant Q-table, compute its
    belief-weighted value, then take the maximum across actions:

        V(s, p_i) = max_a [p_i * Q_i(s, a) + (1 - p_i) * Q_c(s, a)]

    This selects one common assistant action before combining the two
    hypotheses, rather than independently maximizing each table.

    If an action is missing from one table, ``heuristic_q`` uses the available
    table's value for both hypotheses. Return zero only if neither table has
    a usable assistant action value for this state.
    """
    key = joint_state_key(state)
    actions = set(q_i.get(key, {})) | set(q_c.get(key, {}))

    action_values = []
    for action in actions:
        if not action.startswith("A:"):
            continue
        q = heuristic_q(state, action, p_i, q_i, q_c)
        if q is not None and math.isfinite(q):
            action_values.append(q)

    return max(action_values) if action_values else 0.0


def rtdp(maze, start, goal_gems, q_h_ind, q_h_joint, q_a_ind, q_a_col,
         p_independent=0.5, beta=10.0, trials=2000, max_steps=500, seed=0,
         belief_bins=20, check_interval=100, residual_tolerance=1e-3,
         convergence_patience=3, min_trials=300):
    """Learn one assistant policy over (joint state, discretized belief).

    Training episodes are balanced across the two fixed human types for
    trajectory coverage. Bellman backups use the belief-mixture model.

    Convergence is checked every ``check_interval`` episodes using the maximum
    absolute update residual observed during that interval. Training stops
    after ``convergence_patience`` consecutive intervals below the tolerance,
    but never before ``min_trials`` episodes.
    """
    if not 0.0 <= p_independent <= 1.0:
        raise ValueError("p_independent must be in [0, 1]")
    if belief_bins < 1:
        raise ValueError("belief_bins must be >= 1")
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if check_interval < 1:
        raise ValueError("check_interval must be >= 1")
    if residual_tolerance < 0.0:
        raise ValueError("residual_tolerance must be non-negative")
    if convergence_patience < 1:
        raise ValueError("convergence_patience must be >= 1")
    if min_trials < 1:
        raise ValueError("min_trials must be >= 1")

    rng = random.Random(seed)
    values, q_values = {}, {}
    visits = defaultdict(int)
    backups = 0
    interval_max_residual = 0.0
    interval_backup_count = 0
    residual_history = []
    stable_intervals = 0
    converged = False
    episodes_run = 0

    def project_belief(p_i):
        return project_belief_on_grid(p_i, belief_bins)

    def bkey(p_i):
        return project_belief(p_i)

    def value(s, p_i):
        if is_joint_goal(s, goal_gems):
            return 0.0
        k = (s, bkey(p_i))
        if k not in values:
            values[k] = heuristic_v(s, p_i, q_a_ind, q_a_col)
        return values[k]

    initial_belief = project_belief(p_independent)
    # Balance the training episodes across both fixed types for coverage. This
    # sampling schedule is separate from the assistant's prior belief.
    episode_types = [True, False] * (trials // 2)
    if trials % 2:
        episode_types.append(rng.choice([True, False]))
    rng.shuffle(episode_types)

    for episode_idx, true_type_independent in enumerate(episode_types):
        episodes_run = episode_idx + 1
        state, p_i = start, initial_belief
        for _step in range(max_steps):
            if is_joint_goal(state, goal_gems):
                break
            k = (state, bkey(p_i))
            visits[k] += 1
            old_value = value(state, p_i)

            if state[-1] == "assistant":
                action_q = {}
                for action, sp, ch, ca in assistant_actions(maze, state):
                    action_q[action] = -(ch + ca) + value(sp, p_i)
                if not action_q:
                    break
                best_action = max(action_q, key=action_q.get)
                new_value = action_q[best_action]
                values[k] = new_value
                q_values[k] = action_q
                residual = abs(new_value - old_value)
                interval_max_residual = max(interval_max_residual, residual)
                interval_backup_count += 1
                backups += 1
                chosen = next((x for x in assistant_actions(maze, state) if x[0] == best_action), None)
                if chosen is None:
                    break
                _, state, _ch, _ca = chosen
            else:
                pi, pc = human_action_distributions(maze, state, q_h_ind, q_h_joint, beta)
                mixture = normalize({a: p_i * pi.get(a, 0.0) +
                                         (1.0 - p_i) * pc.get(a, 0.0)
                                     for a in (set(pi) | set(pc))})
                if not mixture:
                    break
                expected_value = 0.0
                successors = {}
                legal = legal_human_actions(maze, state)
                for raw_action, prob in mixture.items():
                    if raw_action not in legal:
                        continue
                    sp, ch, ca = legal[raw_action]
                    p_next = project_belief(update_belief(
                        p_i, pi.get(raw_action, 0.0), pc.get(raw_action, 0.0)))
                    successors[raw_action] = (sp, p_next)
                    expected_value += prob * (-(ch + ca) + value(sp, p_next))
                values[k] = expected_value
                q_values[k] = {"__expected__": expected_value}
                residual = abs(expected_value - old_value)
                interval_max_residual = max(interval_max_residual, residual)
                interval_backup_count += 1
                backups += 1

                # Simulate the actual fixed type, while updating belief using
                # both models because the assistant does not know the type.
                true_policy = pi if true_type_independent else pc
                true_dist = normalize(true_policy)
                if not true_dist:
                    true_dist = mixture
                actions = list(true_dist)
                raw_action = rng.choices(actions, weights=[true_dist[a] for a in actions], k=1)[0]
                if raw_action not in legal or raw_action not in successors:
                    break
                state, p_i = successors[raw_action]

        # Evaluate the maximum update residual over this interval. An interval
        # with no backups cannot count as converged.
        interval_complete = (episodes_run % check_interval == 0) or (episodes_run == trials)
        if interval_complete:
            max_residual = (interval_max_residual if interval_backup_count else None)
            residual_history.append({
                "episode": episodes_run,
                "max_update_residual": max_residual,
                "backups": interval_backup_count,
            })
            if (episodes_run >= min_trials and max_residual is not None
                    and max_residual < residual_tolerance):
                stable_intervals += 1
            else:
                stable_intervals = 0

            interval_max_residual = 0.0
            interval_backup_count = 0
            if stable_intervals >= convergence_patience:
                converged = True
                break

    training_stats = {
        "episodes_run": episodes_run,
        "independent_episodes": sum(1 for x in episode_types[:episodes_run] if x),
        "collaborative_episodes": sum(1 for x in episode_types[:episodes_run] if not x),
        "converged": converged,
        "stopping_reason": "residual_converged" if converged else "max_trials_reached",
        "check_interval": check_interval,
        "residual_tolerance": residual_tolerance,
        "convergence_patience": convergence_patience,
        "min_trials": min_trials,
        "stable_intervals_at_stop": stable_intervals,
        "residual_history": residual_history,
    }
    return values, q_values, dict(visits), backups, training_stats


def train(maze, goal_spec, independent_human_path, joint_path, assistant_independent_path,
          p_independent, beta, trials, max_steps, seed, belief_bins,
          check_interval, residual_tolerance, convergence_patience, min_trials):
    goal_label, goal_gems = parse_goal_spec(goal_spec)
    if maze.robot_start is None:
        raise ValueError("Maze must contain assistant/robot start 'm'")
    data_h_ind = read_json(independent_human_path)
    data_joint = read_json(joint_path)
    data_a_ind = read_json(assistant_independent_path)
    q_h_ind = q_table(data_h_ind, goal_label)
    q_h_joint = q_table(data_joint, goal_label)
    q_a_ind, q_a_col = load_assistant_q_tables(data_a_ind, data_joint, goal_label)
    start = initial_joint_state(maze)
    values, q_by_state, visits, backups, training_stats = rtdp(
        maze, start, goal_gems, q_h_ind, q_h_joint, q_a_ind, q_a_col,
        p_independent=p_independent, beta=beta, trials=trials,
        max_steps=max_steps, seed=seed, belief_bins=belief_bins,
        check_interval=check_interval, residual_tolerance=residual_tolerance,
        convergence_patience=convergence_patience, min_trials=min_trials)

    utility_out, q_out, policy_out = {}, {}, {}
    for (state, pkey), val in values.items():
        key = policy_key(state, pkey)
        utility_out[key] = _finite_or_none(val)
        qs = q_by_state.get((state, pkey), {})
        q_out[key] = {a: _finite_or_none(q) for a, q in qs.items()}
        if state[-1] == "assistant":
            acts = {a: q for a, q in qs.items() if a.startswith("A:")}
            if acts:
                policy_out[key] = max(acts, key=acts.get)

    start_belief = project_belief_on_grid(p_independent, belief_bins)
    start_value = values.get((start, start_belief),
                             heuristic_v(start, start_belief, q_a_ind, q_a_col))
    return {
        "goal": f"H:{goal_label}", "human_goal_gems": sorted(goal_gems),
        "solver": "belief_aware_assistant_rtdp",
        "input_files": {"independent_human_q_values": str(independent_human_path),
                        "joint_human_assistant_q_values": str(joint_path),
                        "assistant_independent_human_q_values": str(assistant_independent_path)},
        "human_type": "fixed per episode; hidden from the assistant",
        "initial_p_independent": p_independent,
        "initial_p_collaborative": 1.0 - p_independent,
        "belief_discretization": {"bins": belief_bins, "step_size": 1.0 / belief_bins,
                                   "grid_points": belief_bins + 1},
        "training_type_sampling": "balanced across independent and collaborative types; one type is sampled per episode and remains fixed",
        "training_episode_type_counts": {
            "independent": training_stats["independent_episodes"],
            "collaborative": training_stats["collaborative_episodes"],
        },
        "beta": beta, "rtdp_trials": trials,
        "rtdp_trials_requested": trials,
        "rtdp_trials_run": training_stats["episodes_run"], "seed": seed,
        "max_steps_per_trial": max_steps, "rtdp_backups": backups,
        "visited_belief_states": len(values),
        "training_stats": training_stats,
        "objective": "maximize negative expected combined human + assistant action cost",
        "heuristic": "belief-weighted assistant Q-values; RTDP backups learn one policy over (joint state, belief)",
        "start_utility": _finite_or_none(start_value), "utility": utility_out,
        "q_values": q_out, "policy": policy_out,
        "n_policy_entries": len(policy_out), "pddl_names": dict(maze.pddl_names),
    }


# --------------------------------------------------------------------------
# Testing (policy rollouts)
# --------------------------------------------------------------------------

class PolicyLookup:
    """Look up the trained assistant action for (joint state, belief).

    Exact (state, belief-grid-point) match first. If that pair was never
    visited in training, fall back to the same state at the nearest stored
    belief. If the state has no entries at all, returns None.
    """

    def __init__(self, policy: dict, belief_bins: int):
        self.policy = policy
        self.bins = belief_bins
        self.by_state = defaultdict(dict)  # state signature -> {belief: action}
        for key, action in policy.items():
            obj = json.loads(key)
            pkey = obj.pop("p_independent")
            obj.pop("p_collaborative", None)
            self.by_state[json.dumps(obj, sort_keys=True)][pkey] = action

    def act(self, state: JointState, p_i: float):
        pkey = project_belief_on_grid(p_i, self.bins)
        action = self.policy.get(policy_key(state, pkey))
        if action is not None:
            return action, "exact"
        sig = json.dumps(json.loads(joint_state_key(state)), sort_keys=True)
        options = self.by_state.get(sig)
        if not options:
            return None, "missing"
        nearest = min(options, key=lambda q: abs(q - pkey))
        return options[nearest], "nearest_belief"


def rollout(maze, start, goal_gems, lookup, q_h_ind, q_h_joint, true_type_independent,
            p_independent, beta, belief_bins, max_steps, seed):
    """Roll out the trained policy against a human of a fixed, given type."""
    rng = random.Random(seed)
    state = start
    p_i = project_belief_on_grid(p_independent, belief_bins)
    path, pddl_plan, trajectory = [], [], []
    total_cost = 0.0
    lookup_fallbacks = 0
    termination = "max_steps"

    for t in range(max_steps):
        if is_joint_goal(state, goal_gems):
            termination = "goal"
            break
        if state[-1] == "assistant":
            action, how = lookup.act(state, p_i)
            if action is None:
                termination = "no_policy_entry"
                break
            if how != "exact":
                lookup_fallbacks += 1
            chosen = next((x for x in assistant_actions(maze, state) if x[0] == action), None)
            if chosen is None:
                termination = "illegal_policy_action"
                break
            _, next_state, ch, ca = chosen
        else:
            pi, pc = human_action_distributions(maze, state, q_h_ind, q_h_joint, beta)
            true_dist = normalize(pi if true_type_independent else pc)
            if not true_dist:
                termination = "no_legal_human_action"
                break
            raw = rng.choices(list(true_dist), weights=list(true_dist.values()), k=1)[0]
            legal = legal_human_actions(maze, state).get(raw)
            if legal is None:
                termination = "illegal_human_action"
                break
            next_state, ch, ca = legal
            # The assistant does not know the true type, so its belief update
            # uses both models.
            p_i = project_belief_on_grid(
                update_belief(p_i, pi.get(raw, 0.0), pc.get(raw, 0.0)), belief_bins)
            action = f"H:{raw}"
        state = next_state
        total_cost += ch + ca
        agent, raw_action = action.split(":", 1)
        pddl = to_pddl_action(maze, "human" if agent == "H" else "robot", raw_action)
        path.append(action)
        pddl_plan.append(pddl)
        trajectory.append({
            "t": t + 1, "x": state[0][0], "y": state[0][1],
            "rx": state[3][0], "ry": state[3][1], "turn": state[-1],
            "p_independent": p_i, "p_collaborative": round(1.0 - p_i, 6),
            "action": action, "pddl_action": pddl, "step_cost": ch + ca,
        })
    else:
        # Loop ran out of steps; the last step may still have reached the goal.
        if is_joint_goal(state, goal_gems):
            termination = "goal"

    return {
        "reached_goal": termination == "goal",
        "termination": termination,
        "total_cost": total_cost,
        "steps": len(path),
        "final_p_independent": p_i,
        "policy_lookup_fallbacks": lookup_fallbacks,
        "path": path, "trajectory": trajectory, "pddl_plan": pddl_plan,
    }


def test(maze, goal_spec, policy_path, independent_human_path, joint_path,
         max_steps, seed):
    """Run the trained policy against each human type. Returns {type: result}."""
    goal_label, goal_gems = parse_goal_spec(goal_spec)
    if maze.robot_start is None:
        raise ValueError("Maze must contain assistant/robot start 'm'")
    sol = get_goal_solution(read_json(policy_path), goal_label)
    policy = sol.get("policy")
    if not isinstance(policy, dict) or not policy:
        raise ValueError(f"No 'policy' found in {policy_path}; was it produced by `train`?")
    beta = float(sol["beta"])
    belief_bins = int(sol["belief_discretization"]["bins"])
    p_independent = float(sol["initial_p_independent"])

    q_h_ind = q_table(read_json(independent_human_path), goal_label)
    q_h_joint = q_table(read_json(joint_path), goal_label)
    lookup = PolicyLookup(policy, belief_bins)
    start = initial_joint_state(maze)

    results = {}
    for label, is_independent in (("independent", True), ("collaborative", False)):
        res = rollout(maze, start, goal_gems, lookup, q_h_ind, q_h_joint,
                      true_type_independent=is_independent,
                      p_independent=p_independent, beta=beta,
                      belief_bins=belief_bins, max_steps=max_steps, seed=seed)
        results[label] = {
            "goal": f"H:{goal_label}", "human_goal_gems": sorted(goal_gems),
            "solver": "belief_aware_assistant_rtdp_test",
            "policy_file": str(policy_path),
            "true_human_type": label,
            "initial_p_independent": p_independent,
            "initial_p_collaborative": 1.0 - p_independent,
            "belief_bins": belief_bins, "beta": beta,
            "max_steps": max_steps, "seed": seed,
            **res,
            "pddl_names": dict(maze.pddl_names),
        }
    return results


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    base = Path(__file__).resolve().parent
    mazes = base / "mazes"
    default_policy = mazes / "belief_aware_assistant_rtdp_results.json"

    parser = argparse.ArgumentParser(description="Belief-aware assistant policy using RTDP")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--maze", type=Path, default=mazes / "sample.txt")
        p.add_argument("--human-goal", default="both")
        p.add_argument("--independent-human-q-values", type=Path, default=mazes / "value_iteration_results.json")
        p.add_argument("--joint-q-values", type=Path, default=mazes / "joint_value_iteration_results.json")
        p.add_argument("--max-steps", type=int, default=500)
        p.add_argument("--seed", type=int, default=0)

    p_train = sub.add_parser("train", help="Learn the belief-conditioned policy with RTDP")
    add_common(p_train)
    p_train.add_argument("--out", type=Path, default=default_policy)
    p_train.add_argument("--assistant-independent-q-values", type=Path, default=mazes / "fixed_human_value_iteration_results.json")
    p_train.add_argument("--p-independent", type=float, default=0.5,
                         help="Initial belief that the human is independent")
    p_train.add_argument("--belief-bins", type=int, default=20,
                         help="Number of intervals in the uniform belief grid (20 gives 0.00, 0.05, ..., 1.00)")
    p_train.add_argument("--beta", type=float, default=10.0, help="Boltzmann inverse-temperature for human models")
    p_train.add_argument("--trials", type=int, default=20000)
    p_train.add_argument("--check-interval", type=int, default=500,
                         help="Evaluate update residuals every N training episodes")
    p_train.add_argument("--residual-tolerance", type=float, default=2,
                         help="Maximum update residual required for a stable interval")
    p_train.add_argument("--convergence-patience", type=int, default=3,
                         help="Number of consecutive stable intervals required to stop")
    p_train.add_argument("--min-trials", type=int, default=300,
                         help="Minimum episodes before convergence-based early stopping")

    p_test = sub.add_parser("test", help="Roll out a trained policy against each human type")
    add_common(p_test)
    p_test.add_argument("--policy", type=Path, default=default_policy, help="Output JSON of `train`")
    p_test.add_argument("--out-dir", type=Path, default=mazes)
    p_test.add_argument("--out-prefix", default="belief_aware_assistant_rtdp_test")

    args = parser.parse_args()
    maze = load_maze(args.maze)

    if args.command == "train":
        result = train(maze, args.human_goal, args.independent_human_q_values, args.joint_q_values,
                       args.assistant_independent_q_values, args.p_independent, args.beta,
                       args.trials, args.max_steps, args.seed, args.belief_bins,
                       args.check_interval, args.residual_tolerance,
                       args.convergence_patience, args.min_trials)
        output = {"maze": str(args.maze), "solver": result["solver"],
                  "goals": {result["goal"]: result}, "pddl_names": maze.pddl_names}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Loaded maze: {args.maze} ({maze.width}x{maze.height})")
        print(f"Initial P(independent)={args.p_independent:.3f}; RTDP trials={args.trials}")
        print(f"Belief grid: {args.belief_bins + 1} points (step={1.0 / args.belief_bins:.4f})")
        print(f"Training episodes={result['rtdp_trials_run']}/{result['rtdp_trials_requested']}; "
              f"visited belief states={result['visited_belief_states']}; backups={result['rtdp_backups']}")
        stats = result["training_stats"]
        print(f"Converged={stats['converged']} ({stats['stopping_reason']}); "
              f"residual tolerance={stats['residual_tolerance']}; "
              f"stable intervals={stats['stable_intervals_at_stop']}")
        print(f"Output: {args.out}")
    else:
        results = test(maze, args.human_goal, args.policy, args.independent_human_q_values,
                       args.joint_q_values, args.max_steps, args.seed)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Loaded maze: {args.maze} ({maze.width}x{maze.height})")
        for label, res in results.items():
            out_path = args.out_dir / f"{args.out_prefix}_{label}.json"
            output = {"maze": str(args.maze), "solver": res["solver"],
                      "goals": {res["goal"]: res}, "pddl_names": maze.pddl_names}
            out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
            print(f"[{label}] goal={res['reached_goal']} ({res['termination']}), "
                  f"steps={res['steps']}, cost={res['total_cost']:.3f}, "
                  f"final P(independent)={res['final_p_independent']:.3f}, "
                  f"lookup fallbacks={res['policy_lookup_fallbacks']} -> {out_path}")


if __name__ == "__main__":
    main()