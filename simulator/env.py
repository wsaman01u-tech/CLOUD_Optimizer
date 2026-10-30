"""Cloud scaling simulation environment.
Models server clusters, workload demand, operational costs, and SLA penalties.
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class CloudEnv:
    """Simulated Cloud Environment for Autonomous Scaling."""

    def __init__(
        self,
        load: List[float],
        forecast: Optional[List[float]] = None,
        capacity: float = 100.0,
        price: float = 1.0,
        sla_penalty: float = 5.0,
        min_servers: int = 1,
        max_servers: int = 10,
        initial_servers: int = 3,
    ):
        """Initialize cloud environment.

        Args:
            load: Hourly demand units over time
            forecast: Optional forecasted hourly demand units
            capacity: Fixed capacity handled by each server
            price: Cost per server per hour
            sla_penalty: Penalty factor incurred per SLA violation event
            min_servers: Minimum allowed active servers (default: 1)
            max_servers: Maximum allowed active servers (default: 10)
            initial_servers: Initial active server count (default: 3)
        """
        self.load = list(load)
        self.forecast = list(forecast) if forecast is not None else [float(x) for x in self.load]
        self.capacity = float(capacity)
        self.price = float(price)
        self.sla_penalty = float(sla_penalty)
        self.min_servers = min_servers
        self.max_servers = max_servers
        self.initial_servers = max(min_servers, min(initial_servers, max_servers))
        self.reset()

    def reset(self) -> Tuple[int, ...]:
        """Reset environment to initial timestep."""
        self.t = 0
        self.servers = self.initial_servers
        return self._state()

    def _bucket(self, value: float) -> int:
        """Discretize continuous workload into discrete state bucket [0, max_servers].
        Uses ceiling division so that the bucket represents the minimum server capacity required.
        """
        if self.capacity <= 0:
            return 0
        return min(max(0, int(np.ceil(value / self.capacity))), self.max_servers)

    def _state(self) -> Tuple[int, int, int]:
        """Return state tuple: (current_load_bucket, forecast_load_bucket, servers)."""
        idx = min(self.t, len(self.load) - 1)
        cur_b = self._bucket(self.load[idx])
        fc_val = self.forecast[idx] if idx < len(self.forecast) else self.load[idx]
        fc_b = self._bucket(fc_val)
        return (cur_b, fc_b, self.servers)

    def step(self, action: int) -> Tuple[Tuple[int, int, int], float, bool, Dict[str, Any]]:
        """Advance one simulation step.

        Actions:
            0: Remove 1 server (scale down)
            1: Hold current server count
            2: Add 1 server (scale up)
        """
        servers_before = self.servers
        if action == 0:
            self.servers = max(self.min_servers, self.servers - 1)
        elif action == 2:
            self.servers = min(self.max_servers, self.servers + 1)
        # action == 1 means hold

        demand = self.load[self.t]
        forecast_val = self.forecast[self.t] if self.t < len(self.forecast) else demand
        total_capacity = self.servers * self.capacity
        cost = self.servers * self.price
        violation = 1 if demand > total_capacity else 0
        utilization = demand / total_capacity if total_capacity > 0 else 1.0

        # Reward = negative cost, heavily penalized for any SLA violation
        # A breach must be significantly costlier than running extra instances
        reward = -cost - (self.sla_penalty * violation)

        self.t += 1
        done = self.t >= len(self.load)

        info = {
            "t": self.t - 1,
            "demand": round(float(demand), 2),
            "forecast": round(float(forecast_val), 2),
            "servers_before": servers_before,
            "servers_after": self.servers,
            "action": action,
            "cost": round(float(cost), 2),
            "violation": violation,
            "utilization": round(float(utilization), 3),
            "total_capacity": round(float(total_capacity), 2),
        }

        next_state = self._state() if not done else (0, 0, self.servers)
        return next_state, reward, done, info
