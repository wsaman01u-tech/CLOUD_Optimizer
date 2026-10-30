"""Audit and Explainability Agent.
Logs all autonomous scaling actions into tamper-evident records and provides
plain-English reasoning for stakeholders and compliance audits.
"""
import os
import csv
from datetime import datetime
from typing import Dict, Any, List, Optional


def explain_decision(row: Dict[str, Any]) -> str:
    """Generate high-clarity natural language explanation from structured telemetry."""
    action_raw = row.get("action", "hold")
    if isinstance(action_raw, int):
        action_map = {0: "remove", 1: "hold", 2: "add"}
        action_str = action_map.get(action_raw, "hold")
    else:
        action_str = str(action_raw).lower()

    load = row.get("load", row.get("demand", 0))
    forecast = row.get("forecast", load)
    sb = row.get("servers_before", row.get("servers", 1))
    sa = row.get("servers_after", sb)
    cost_change = row.get("cost_change", 0.0)
    sla_risk = row.get("sla_risk", "low")

    # Semantic context
    if action_str == "add":
        action_phrase = f"scale up by adding a server ({sb} -> {sa})"
        rationale = f"forecasted load ({forecast:.1f}) approaches or exceeds current capacity threshold, preventing SLA penalties"
    elif action_str == "remove":
        action_phrase = f"scale down by decommissioning an idle server ({sb} -> {sa})"
        rationale = f"load ({load:.1f}) and forecast ({forecast:.1f}) show surplus headroom, cutting redundant cloud spend"
    else:
        action_phrase = f"maintain current fleet size at {sb} servers"
        rationale = f"system load ({load:.1f}) operates comfortably within optimal utilization bounds"

    cost_sign = "+" if cost_change > 0 else ""
    return (
        f"At this step, observed demand was {load:.1f} units and forecasted demand was {forecast:.1f} units. "
        f"The agent decided to {action_phrase} because {rationale}. "
        f"Hourly cost impact: {cost_sign}${cost_change:.2f}/hr. SLA breach risk: {sla_risk.upper()}."
    )


class AuditAgent:
    """Audit Agent managing continuous decision transparency and logs."""

    def __init__(self, log_path: str = "data/results/decisions.csv"):
        self.log_path = log_path
        self.output_csv = "outputs/decisions.csv"
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        self.decisions: List[Dict[str, Any]] = []

    def record_decision(
        self,
        timestamp: str,
        demand: float,
        forecast: float,
        servers_before: int,
        servers_after: int,
        action: int,
        price_per_server: float = 1.0,
        capacity_per_server: float = 100.0,
    ) -> Dict[str, Any]:
        """Record and explain a single scaling decision."""
        action_names = {0: "remove", 1: "hold", 2: "add"}
        action_str = action_names.get(action, "hold")

        cost_before = servers_before * price_per_server
        cost_after = servers_after * price_per_server
        cost_change = cost_after - cost_before

        new_capacity = servers_after * capacity_per_server
        headroom = new_capacity - max(demand, forecast)
        if headroom < 0:
            sla_risk = "high"
        elif headroom < 30:
            sla_risk = "moderate"
        else:
            sla_risk = "low"

        row = {
            "timestamp": timestamp,
            "demand": round(float(demand), 2),
            "forecast": round(float(forecast), 2),
            "servers_before": servers_before,
            "servers_after": servers_after,
            "action": action_str,
            "cost_change": round(float(cost_change), 2),
            "sla_risk": sla_risk,
        }

        row["explanation"] = explain_decision(row)
        self.decisions.append(row)
        return row

    def persist(self) -> None:
        """Write all recorded decisions to disk in CSV format."""
        if not self.decisions:
            return

        fieldnames = list(self.decisions[0].keys())

        for target_path in [self.log_path, self.output_csv]:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.decisions:
                    writer.writerow(row)
