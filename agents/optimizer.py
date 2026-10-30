"""Optimization Agent (Reinforcement Learning & Rule Baselines).
Learns proactive scaling policies using Q-learning, balancing infrastructure cost and SLA targets.
"""
import os
import random
from typing import Dict, Any, Tuple, List, Callable
import numpy as np
import pandas as pd
from simulator.env import CloudEnv


def static_policy(s: Tuple[int, ...], env: CloudEnv) -> int:
    """Baseline policy: statically holds server count constant."""
    return 1  # Hold


def threshold_policy(s: Tuple[int, ...], env: CloudEnv) -> int:
    """Reactive baseline policy: scales up when usage > 70%, scales down when < 30%."""
    if env.t >= len(env.load):
        return 1
    total_cap = env.servers * env.capacity
    if total_cap <= 0:
        return 2
    usage = env.load[env.t] / total_cap
    if usage > 0.70:
        return 2  # Add server
    if usage < 0.30:
        return 0  # Remove server
    return 1  # Hold


def train_q_learning(
    env: CloudEnv,
    episodes: int = 400,
    alpha: float = 0.1,
    gamma: float = 0.95,
    eps_start: float = 1.0,
    eps_min: float = 0.05,
    decay: float = 0.992,
    seed: int = 42,
) -> np.ndarray:
    """Train tabular Q-learning agent on the simulation environment.

    State: (current_load_bucket, forecast_bucket, server_count)
    Action: 0 = scale down, 1 = hold, 2 = scale up
    """
    random.seed(seed)
    np.random.seed(seed)

    size = env.max_servers + 1
    # 4D Q-table: (load_bucket, forecast_bucket, servers, actions)
    Q = np.zeros((size, size, size, 3), dtype=float)
    eps = eps_start

    for ep in range(episodes):
        s = env.reset()
        done = False
        while not done:
            s_idx = (min(s[0], env.max_servers), min(s[1], env.max_servers), min(s[2], env.max_servers))
            if random.random() < eps:
                a = random.randint(0, 2)
            else:
                a = int(np.argmax(Q[s_idx]))

            s2, r, done, _ = env.step(a)
            s2_idx = (min(s2[0], env.max_servers), min(s2[1], env.max_servers), min(s2[2], env.max_servers))

            best_next = float(np.max(Q[s2_idx]))
            Q[s_idx + (a,)] += alpha * (r + gamma * best_next - Q[s_idx + (a,)])
            s = s2

        eps = max(eps_min, eps * decay)

    return Q


def q_policy(Q: np.ndarray, max_servers: int = 10) -> Callable:
    """Generate greedy policy function from learned Q-table with safe fallback for unvisited states."""
    def policy(s: Tuple[int, ...], env: CloudEnv) -> int:
        s_idx = (min(s[0], max_servers), min(s[1], max_servers), min(s[2], max_servers))
        q_vals = Q[s_idx]
        if np.all(q_vals == 0):
            # Safe proactive fallback for unvisited state tuples
            needed = max(s[0], s[1])
            if s[2] < needed:
                return 2  # add server
            elif s[2] > needed + 1:
                return 0  # remove server
            return 1  # hold
        return int(np.argmax(q_vals))
    return policy


def evaluate_policy(env: CloudEnv, policy: Callable) -> Dict[str, Any]:
    """Execute a policy over an environment episode and record full diagnostics."""
    s = env.reset()
    done = False
    total_cost = 0.0
    violations = 0
    utilizations = []
    rewards = []
    history = []

    while not done:
        a = policy(s, env)
        s, r, done, info = env.step(a)
        total_cost += info["cost"]
        violations += info["violation"]
        utilizations.append(info["utilization"])
        rewards.append(r)
        history.append(info)

    n_steps = max(1, len(history))
    avg_util = float(np.mean(utilizations)) * 100.0 if utilizations else 0.0
    sla_compliance = round(100.0 * (1.0 - (violations / n_steps)), 2)

    return {
        "total_cost": round(total_cost, 2),
        "violations": violations,
        "sla_compliance_pct": sla_compliance,
        "avg_utilization_pct": round(avg_util, 2),
        "total_reward": round(float(np.sum(rewards)), 2),
        "history": history,
    }


def compare_strategies(
    test_load: List[float],
    test_forecast: List[float],
    trained_Q: np.ndarray,
    capacity: float = 100.0,
    price: float = 1.0,
    sla_penalty: float = 100.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compare Static, Reactive Threshold, and Proactive RL Optimizer."""
    # 1. Static Policy Evaluation (Static provisioned for peak demand = 8 servers)
    env_static = CloudEnv(test_load, test_forecast, capacity=capacity, price=price, sla_penalty=sla_penalty, initial_servers=8)
    res_static = evaluate_policy(env_static, static_policy)

    # 2. Threshold Policy Evaluation
    env_thresh = CloudEnv(test_load, test_forecast, capacity=capacity, price=price, sla_penalty=sla_penalty, initial_servers=3)
    res_thresh = evaluate_policy(env_thresh, threshold_policy)

    # 3. RL Policy Evaluation
    env_rl = CloudEnv(test_load, test_forecast, capacity=capacity, price=price, sla_penalty=sla_penalty, initial_servers=3)
    rl_pol = q_policy(trained_Q, max_servers=env_rl.max_servers)
    res_rl = evaluate_policy(env_rl, rl_pol)

    baseline_cost = res_static["total_cost"]

    results = [
        {
            "strategy": "Static Allocation (Fixed 8)",
            "total_cost": res_static["total_cost"],
            "sla_violations": res_static["violations"],
            "sla_compliance_pct": res_static["sla_compliance_pct"],
            "avg_utilization_pct": res_static["avg_utilization_pct"],
            "cost_savings_pct": 0.0,
        },
        {
            "strategy": "Reactive Threshold (30/70%)",
            "total_cost": res_thresh["total_cost"],
            "sla_violations": res_thresh["violations"],
            "sla_compliance_pct": res_thresh["sla_compliance_pct"],
            "avg_utilization_pct": res_thresh["avg_utilization_pct"],
            "cost_savings_pct": round(((baseline_cost - res_thresh["total_cost"]) / baseline_cost) * 100, 2),
        },
        {
            "strategy": "Autonomous RL Optimizer (Q-Learning + Forecast)",
            "total_cost": res_rl["total_cost"],
            "sla_violations": res_rl["violations"],
            "sla_compliance_pct": res_rl["sla_compliance_pct"],
            "avg_utilization_pct": res_rl["avg_utilization_pct"],
            "cost_savings_pct": round(((baseline_cost - res_rl["total_cost"]) / baseline_cost) * 100, 2),
        },
    ]

    eval_df = pd.DataFrame(results)
    return eval_df, {"static": res_static, "threshold": res_thresh, "rl": res_rl}


def run_optimization_agent(
    train_load: List[float],
    train_forecast: List[float],
    test_load: List[float],
    test_forecast: List[float],
    output_dir: str = "data/results",
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """Train RL agent, benchmark against baselines, and save evaluation metrics."""
    os.makedirs(output_dir, exist_ok=True)

    train_env = CloudEnv(train_load, train_forecast, capacity=100.0, price=1.0, sla_penalty=100.0)
    Q = train_q_learning(train_env, episodes=650, alpha=0.15, gamma=0.95, eps_start=1.0, eps_min=0.03, decay=0.994)

    eval_df, detailed_runs = compare_strategies(test_load, test_forecast, Q, capacity=100.0, price=1.0, sla_penalty=100.0)

    eval_path = os.path.join(output_dir, "evaluation.csv")
    eval_df.to_csv(eval_path, index=False)

    # Save Q-table for deployment serving / quick inference
    np.save(os.path.join(output_dir, "q_table.npy"), Q)

    return eval_df, Q, detailed_runs
