"""End-to-end execution pipeline for Autonomous Cloud Cost & Performance Optimizer.
Coordinates all 5 autonomous agents:
1. Data & Usage Monitoring Agent
2. Demand Forecasting Agent
3. RL Cloud Optimization Agent
4. Deployment & Infrastructure Agent
5. Audit & Explainability Agent
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from agents.monitor import run_monitoring_agent
from agents.forecast import run_forecasting_agent
from agents.optimizer import run_optimization_agent, q_policy, evaluate_policy
from agents.deploy import DeploymentAgent
from agents.audit import AuditAgent
from simulator.env import CloudEnv


def generate_visual_plots(vm_summary: pd.DataFrame, eval_df: pd.DataFrame, forecast_df: pd.DataFrame):
    """Generate Matplotlib plots required by the project specifications."""
    os.makedirs("outputs/plots", exist_ok=True)
    plt.style.use("ggplot")

    # Plot 1: VM Status Distribution
    plt.figure(figsize=(8, 5))
    status_counts = vm_summary["status"].value_counts()
    colors = ["#10b981" if s == "OPTIMAL" else "#ef4444" if s == "IDLE" else "#f59e0b" for s in status_counts.index]
    status_counts.plot(kind="bar", color=colors)
    plt.title("Cloud Fleet VM Efficiency Classification", fontsize=14, fontweight="bold")
    plt.xlabel("Status", fontsize=12)
    plt.ylabel("Number of VMs", fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("outputs/plots/vm_status_counts.png", dpi=150)
    plt.close()

    # Plot 2: Average CPU Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(vm_summary["avg_cpu"], bins=10, color="#3b82f6", edgecolor="black", alpha=0.8)
    plt.title("Distribution of Fleet Average CPU Utilization", fontsize=14, fontweight="bold")
    plt.xlabel("Average CPU %", fontsize=12)
    plt.ylabel("Count of VMs", fontsize=12)
    plt.axvline(10, color="red", linestyle="--", label="Idle Threshold (10%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/cpu_distribution.png", dpi=150)
    plt.close()

    # Plot 3: Strategy Comparison (Cost vs SLA Violations)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    strategies = [s.split(" ")[0] for s in eval_df["strategy"]]
    costs = eval_df["total_cost"]
    violations = eval_df["sla_violations"]

    x = np.arange(len(strategies))
    width = 0.35

    ax1.bar(x - width/2, costs, width, label="Total Cost ($)", color="#6366f1")
    ax1.set_ylabel("Total Cost ($)", color="#6366f1", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=11)

    ax2 = ax1.twinx()
    ax2.bar(x + width/2, violations, width, label="SLA Violations", color="#ef4444")
    ax2.set_ylabel("SLA Violations (count)", color="#ef4444", fontsize=12)

    plt.title("Autoscaling Strategies Benchmark (Cost vs SLA Violations)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig("outputs/plots/strategy_comparison.png", dpi=150)
    plt.close()

    # Plot 4: Forecast vs Actual Demand
    plt.figure(figsize=(10, 4.5))
    subset = forecast_df.head(72)
    plt.plot(range(len(subset)), subset["actual"], label="Actual Demand", color="#3b82f6", lw=2)
    plt.plot(range(len(subset)), subset["predicted"], label="ML Forecast (RF)", color="#10b981", lw=2, linestyle="--")
    plt.title("Workload Demand: Actual vs Machine Learning Forecast (72h)", fontsize=13, fontweight="bold")
    plt.xlabel("Timeline (Hours)", fontsize=12)
    plt.ylabel("Demand Units", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/forecast_comparison.png", dpi=150)
    plt.close()


def run_pipeline() -> None:
    """Run full cooperative multi-agent optimization pipeline."""
    print("=" * 70)
    print("STARTING AUTONOMOUS CLOUD COST & PERFORMANCE OPTIMIZER PIPELINE")
    print("=" * 70)

    # 1. Monitoring Agent
    print("\n[Stage 1/5] Executing Data & Usage Monitoring Agent...")
    vm_summary, kpis = run_monitoring_agent()
    print(f" -> Analyzed {kpis['total_vms']} VMs.")
    print(f" -> Found {kpis['idle_vms']} idle VMs, {kpis['over_provisioned_vms']} over-provisioned VMs.")
    print(f" -> Estimated financial waste: ${kpis['total_waste_usd']} ({kpis['waste_percentage']}% of spend).")

    # 2. Forecasting Agent
    print("\n[Stage 2/5] Executing Demand Forecasting Agent...")
    metrics_df, forecast_df, forecaster = run_forecasting_agent()
    print(" -> Forecasting Model Performance (RMSE / MAPE):")
    for _, row in metrics_df.iterrows():
        print(f"    * {row['model']}: RMSE = {row['rmse']}, MAPE = {row['mape_percent']}%")

    # 3. Optimization Agent (RL & Simulator)
    print("\n[Stage 3/5] Executing Optimization Agent (Q-Learning RL vs Baselines)...")
    # Load demand series
    raw_df = pd.read_csv("data/raw/vm_cpu_readings.csv")
    raw_df["dt"] = pd.to_datetime(raw_df["time_iso"])
    hourly = raw_df.groupby(pd.Grouper(key="dt", freq="1h"))["avg_cpu"].sum().dropna()

    # Scale series so peak is around 800 load units (fitting 1-10 servers of capacity 100)
    scale_factor = 800.0 / max(1.0, hourly.max())
    scaled_series = (hourly * scale_factor).round(1)

    n = len(scaled_series)
    train_split = int(n * 0.70)
    train_load = scaled_series.iloc[:train_split].tolist()
    test_load = scaled_series.iloc[train_split:].tolist()

    # Forecast values corresponding to train and test
    train_forecast = train_load  # Training baseline
    # Test forecast from trained forecaster
    test_forecast = [forecaster.predict_next(scaled_series.iloc[:train_split + i]) for i in range(len(test_load))]

    eval_df, Q, detailed_runs = run_optimization_agent(
        train_load=train_load,
        train_forecast=train_forecast,
        test_load=test_load,
        test_forecast=test_forecast,
    )
    print(" -> Autoscaling Benchmark Results:")
    for _, row in eval_df.iterrows():
        print(f"    * {row['strategy']}: Cost=${row['total_cost']}, Violations={row['sla_violations']}, SLA={row['sla_compliance_pct']}%, Savings={row['cost_savings_pct']}%")

    # 4. Deployment Agent & 5. Audit Agent
    print("\n[Stage 4 & 5/5] Executing Deployment & Audit Agents on RL scaling trajectory...")
    deployer = DeploymentAgent(min_servers=1, max_servers=10, dry_run=True)
    auditor = AuditAgent()

    rl_history = detailed_runs["rl"]["history"]
    test_dates = scaled_series.iloc[train_split:].index

    for step_info in rl_history:
        t_idx = step_info["t"]
        ts_str = test_dates[t_idx].isoformat() if t_idx < len(test_dates) else f"T+{t_idx}"

        # Deployment safe translation
        dep_rec = deployer.execute_scaling(
            servers_current=step_info["servers_before"],
            target_servers=step_info["servers_after"],
            reason=f"RL Agent action {step_info['action']}",
        )

        # Audit logging with natural language explanation
        auditor.record_decision(
            timestamp=ts_str,
            demand=step_info["demand"],
            forecast=step_info["forecast"],
            servers_before=step_info["servers_before"],
            servers_after=dep_rec["servers_deployed"],
            action=step_info["action"],
            price_per_server=1.0,
            capacity_per_server=100.0,
        )

    auditor.persist()
    print(f" -> Generated {len(auditor.decisions)} audited scaling decisions.")
    print(" -> Generated Terraform and Kubernetes configurations in outputs/terraform/.")

    # 6. Generate Plots
    print("\nGenerating charts in outputs/plots/...")
    generate_visual_plots(vm_summary, eval_df, forecast_df)

    # 7. Write aggregate overview JSON
    static_cost = float(eval_df.loc[eval_df["strategy"].str.startswith("Static"), "total_cost"].iloc[0])
    rl_cost = float(eval_df.loc[eval_df["strategy"].str.startswith("Autonomous"), "total_cost"].iloc[0])
    savings_pct = float(eval_df.loc[eval_df["strategy"].str.startswith("Autonomous"), "cost_savings_pct"].iloc[0])
    sla_pct = float(eval_df.loc[eval_df["strategy"].str.startswith("Autonomous"), "sla_compliance_pct"].iloc[0])

    overview_payload = {
        "static_cost": static_cost,
        "optimized_cost": rl_cost,
        "cost_saved_usd": round(static_cost - rl_cost, 2),
        "cost_savings_pct": savings_pct,
        "sla_compliance_pct": sla_pct,
        "sla_violations": int(eval_df.loc[eval_df["strategy"].str.startswith("Autonomous"), "sla_violations"].iloc[0]),
        "total_vms_monitored": kpis["total_vms"],
        "idle_vms_detected": kpis["idle_vms"],
        "overprovisioned_vms": kpis["over_provisioned_vms"],
        "fleet_waste_usd": kpis["total_waste_usd"],
        "latest_servers": rl_history[-1]["servers_after"] if rl_history else 3,
        "pipeline_status": "ONLINE",
    }

    with open("data/results/overview.json", "w") as f:
        json.dump(overview_payload, f, indent=2)

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Net Cloud Cost Reduction: {savings_pct}% with {sla_pct}% SLA compliance.")
    print("Results saved in data/results/ and outputs/.")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
