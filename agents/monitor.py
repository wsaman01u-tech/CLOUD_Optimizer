"""Usage and Monitoring Agent.
Analyzes VM telemetry to detect idle machines, over-provisioned instances,
and compute financial cloud waste.
"""
import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def generate_synthetic_vm_data(
    num_vms: int = 25,
    days: int = 14,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic multi-VM CPU workload telemetry.

    Generates heterogeneous profiles:
    - Normal production servers (diurnal cycle, 30-75% CPU)
    - Idle servers (dev/test left running, <10% CPU >90% time)
    - Over-provisioned servers (light tasks on heavy VMs, peak < 35%)
    - Spiky microservices (low baseline with periodic bursts)
    """
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2026-01-01", periods=24 * days, freq="h")
    records = []

    for vm_idx in range(1, num_vms + 1):
        vm_id = f"vm-{vm_idx:03d}"
        profile_type = vm_idx % 4

        # Diurnal base curve (peaks around 14:00, drops at night)
        diurnal = 0.5 + 0.5 * np.sin((hours.hour - 8) / 24 * 2 * np.pi)
        weekend_factor = np.where(hours.dayofweek >= 5, 0.65, 1.0)

        if profile_type == 0:
            # Idle VM: forgotten staging/dev box
            base_cpu = rng.uniform(1.0, 4.0, len(hours))
            noise = rng.normal(0, 1.2, len(hours))
            cpu = np.clip(base_cpu + noise, 0.5, 8.5)
        elif profile_type == 1:
            # Over-provisioned VM: provisioned for 10x capacity
            base_cpu = 12.0 + 10.0 * diurnal * weekend_factor
            noise = rng.normal(0, 2.5, len(hours))
            cpu = np.clip(base_cpu + noise, 4.0, 32.0)
        elif profile_type == 2:
            # Spiky batch VM
            base_cpu = 15.0 + 20.0 * diurnal
            spikes = (rng.uniform(0, 1, len(hours)) > 0.96) * rng.uniform(40, 70, len(hours))
            cpu = np.clip(base_cpu + spikes, 5.0, 95.0)
        else:
            # Healthy production VM
            base_cpu = 25.0 + 45.0 * diurnal * weekend_factor
            noise = rng.normal(0, 4.0, len(hours))
            cpu = np.clip(base_cpu + noise, 10.0, 92.0)

        for t_idx, ts in enumerate(hours):
            val = float(cpu[t_idx])
            records.append({
                "timestamp": int(ts.timestamp()),
                "time_iso": ts.isoformat(),
                "vm_id": vm_id,
                "min_cpu": round(max(0.0, val - rng.uniform(1, 4)), 2),
                "max_cpu": round(min(100.0, val + rng.uniform(2, 6)), 2),
                "avg_cpu": round(val, 2),
            })

    return pd.DataFrame(records)


def summarize_vms(
    df: pd.DataFrame,
    idle_limit: float = 10.0,
    low_peak_limit: float = 40.0,
    price_per_vm_hour: float = 0.96,
) -> pd.DataFrame:
    """Analyze VM CPU metrics and classify each VM's efficiency status."""
    stats = df.groupby("vm_id").agg(
        avg_cpu=("avg_cpu", "mean"),
        min_cpu=("min_cpu", "min"),
        peak_cpu=("max_cpu", "max"),
        reading_count=("avg_cpu", "count"),
        pct_low=("avg_cpu", lambda s: float(100.0 * (s < idle_limit).mean())),
    ).reset_index()

    # Classification rules:
    # 1. Idle: below idle_limit for > 90% of observed time
    stats["is_idle"] = stats["pct_low"] > 90.0

    # 2. Over-provisioned: peak CPU never exceeds low_peak_limit and not completely idle
    stats["is_over_provisioned"] = (stats["peak_cpu"] < low_peak_limit) & (~stats["is_idle"])

    # Status label
    def assign_status(row):
        if row["is_idle"]:
            return "IDLE"
        if row["is_over_provisioned"]:
            return "OVER_PROVISIONED"
        return "OPTIMAL"

    stats["status"] = stats.apply(assign_status, axis=1)

    # Compute financial waste
    # Idle VM: 100% of cost is wasted
    # Over-provisioned VM: ~50% waste assuming 2x rightsizing opportunity
    hours_observed = stats["reading_count"].max()
    stats["total_cost"] = stats["reading_count"] * price_per_vm_hour

    def calc_waste(row):
        if row["is_idle"]:
            return row["total_cost"]
        if row["is_over_provisioned"]:
            return row["total_cost"] * 0.50
        return 0.0

    stats["estimated_waste"] = stats.apply(calc_waste, axis=1)

    # Round numeric columns for clean reporting
    for col in ["avg_cpu", "min_cpu", "peak_cpu", "pct_low", "total_cost", "estimated_waste"]:
        stats[col] = stats[col].round(2)

    return stats


def compute_fleet_kpis(summary_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate aggregate cloud fleet monitoring KPIs."""
    total_vms = len(summary_df)
    idle_vms = int((summary_df["status"] == "IDLE").sum())
    overprov_vms = int((summary_df["status"] == "OVER_PROVISIONED").sum())
    optimal_vms = int((summary_df["status"] == "OPTIMAL").sum())

    total_spend = float(summary_df["total_cost"].sum())
    total_waste = float(summary_df["estimated_waste"].sum())
    waste_percentage = round((total_waste / total_spend * 100) if total_spend > 0 else 0, 1)

    return {
        "total_vms": total_vms,
        "idle_vms": idle_vms,
        "over_provisioned_vms": overprov_vms,
        "optimal_vms": optimal_vms,
        "fleet_avg_cpu": round(float(summary_df["avg_cpu"].mean()), 2),
        "total_spend_usd": round(total_spend, 2),
        "total_waste_usd": round(total_waste, 2),
        "waste_percentage": waste_percentage,
    }


def run_monitoring_agent(
    raw_path: str = "data/raw/vm_cpu_readings.csv",
    output_dir: str = "data/results",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run monitoring agent, generating or reading data and saving results."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
    else:
        df = generate_synthetic_vm_data(num_vms=25, days=14)
        df.to_csv(raw_path, index=False)

    summary_df = summarize_vms(df)
    kpis = compute_fleet_kpis(summary_df)

    summary_path = os.path.join(output_dir, "vm_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    return summary_df, kpis
