"""FastAPI Backend for Autonomous Cloud Cost & Performance Optimizer.
Serves stored multi-agent telemetry, forecasting results, scaling audit logs,
and interactive simulation endpoints.
"""
import os
import json
from typing import Dict, Any, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from simulator.env import CloudEnv
from agents.deploy import DeploymentAgent
from agents.audit import explain_decision

app = FastAPI(
    title="Autonomous Cloud Cost & Performance Optimizer API",
    description="REST API serving autonomous agent decisions, demand forecasts, and cloud metrics.",
    version="1.0.0",
)

# CORS configuration
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:5500,http://localhost:8000")
origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

RESULTS_DIR = "data/results"
OUTPUTS_DIR = "outputs"


def read_csv_safe(filename: str, limit: int = 2000) -> List[Dict[str, Any]]:
    """Safely read CSV from RESULTS_DIR, filling NaNs to avoid invalid JSON."""
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        # Fallback to outputs dir
        filepath = os.path.join(OUTPUTS_DIR, filename)
        if not os.path.exists(filepath):
            return []

    try:
        df = pd.read_csv(filepath)
        df = df.head(limit)
        # Convert NaN and Inf to friendly JSON values
        return df.fillna(0).to_dict(orient="records")
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint used by hosting platforms and container probes."""
    return {
        "status": "ok",
        "service": "Autonomous Cloud Cost Optimizer API",
        "version": "1.0.0",
    }


@app.get("/api/overview")
def get_overview() -> Dict[str, Any]:
    """Return top-level KPIs, cost savings, and fleet optimization status."""
    overview_path = os.path.join(RESULTS_DIR, "overview.json")
    if os.path.exists(overview_path):
        with open(overview_path, "r") as f:
            return json.load(f)

    # Fallback compute from evaluation and vm_summary if JSON not generated yet
    eval_data = read_csv_safe("evaluation.csv")
    vm_data = read_csv_safe("vm_summary.csv")

    static_cost = 0.0
    rl_cost = 0.0
    savings_pct = 0.0
    sla_pct = 100.0
    sla_violations = 0

    for row in eval_data:
        strat = str(row.get("strategy", ""))
        if "Static" in strat:
            static_cost = float(row.get("total_cost", 0))
        elif "Autonomous" in strat or "RL" in strat:
            rl_cost = float(row.get("total_cost", 0))
            savings_pct = float(row.get("cost_savings_pct", 0))
            sla_pct = float(row.get("sla_compliance_pct", 100))
            sla_violations = int(row.get("sla_violations", 0))

    idle_count = sum(1 for vm in vm_data if vm.get("status") == "IDLE")
    overprov_count = sum(1 for vm in vm_data if vm.get("status") == "OVER_PROVISIONED")
    total_waste = sum(float(vm.get("estimated_waste", 0)) for vm in vm_data)

    return {
        "static_cost": static_cost,
        "optimized_cost": rl_cost,
        "cost_saved_usd": round(static_cost - rl_cost, 2),
        "cost_savings_pct": savings_pct,
        "sla_compliance_pct": sla_pct,
        "sla_violations": sla_violations,
        "total_vms_monitored": len(vm_data),
        "idle_vms_detected": idle_count,
        "overprovisioned_vms": overprov_count,
        "fleet_waste_usd": round(total_waste, 2),
        "latest_servers": 3,
        "pipeline_status": "ONLINE",
    }


@app.get("/api/usage")
def get_usage(limit: int = 500) -> List[Dict[str, Any]]:
    """Return VM inventory telemetry and efficiency classification."""
    return read_csv_safe("vm_summary.csv", limit=limit)


@app.get("/api/forecast")
def get_forecast(limit: int = 1000) -> List[Dict[str, Any]]:
    """Return actual vs predicted demand time series."""
    return read_csv_safe("forecast.csv", limit=limit)


@app.get("/api/forecast/metrics")
def get_forecast_metrics() -> List[Dict[str, Any]]:
    """Return forecasting accuracy comparisons (RMSE, MAE, MAPE)."""
    return read_csv_safe("forecast_metrics.csv")


@app.get("/api/decisions")
def get_decisions(limit: int = 500) -> List[Dict[str, Any]]:
    """Return chronological audit log of scaling decisions with explanations."""
    return read_csv_safe("decisions.csv", limit=limit)


@app.get("/api/evaluation")
def get_evaluation() -> List[Dict[str, Any]]:
    """Return comparison table across Static, Reactive Threshold, and RL strategies."""
    return read_csv_safe("evaluation.csv")


@app.get("/api/config")
def get_config() -> Dict[str, str]:
    """Return latest generated Terraform and Kubernetes configuration files."""
    tfvars_path = os.path.join(OUTPUTS_DIR, "terraform", "terraform.tfvars")
    vars_path = os.path.join(OUTPUTS_DIR, "terraform", "variables.tf")
    k8s_path = os.path.join(OUTPUTS_DIR, "terraform", "deployment.yaml")

    def read_file(p: str) -> str:
        if os.path.exists(p):
            with open(p, "r") as f:
                return f.read()
        return "# File not generated yet. Run pipeline first."

    return {
        "terraform_tfvars": read_file(tfvars_path),
        "variables_tf": read_file(vars_path),
        "kubernetes_yaml": read_file(k8s_path),
    }


class SimulateStepRequest(BaseModel):
    current_servers: int = Field(default=3, ge=1, le=10)
    current_demand: float = Field(default=250.0, ge=0.0)
    forecast_demand: float = Field(default=320.0, ge=0.0)
    capacity_per_server: float = Field(default=100.0, gt=0.0)
    price_per_server: float = Field(default=1.0, gt=0.0)


@app.post("/api/simulate-step")
def simulate_single_step(req: SimulateStepRequest) -> Dict[str, Any]:
    """Interactive endpoint: compute real-time scaling action, safety checks, and explanation."""
    # Compute optimal action based on current load, forecast, and capacity
    total_capacity = req.current_servers * req.capacity_per_server
    effective_future_load = max(req.current_demand, req.forecast_demand)

    # Decision logic combining trained bounds & proactive headroom
    if effective_future_load > total_capacity * 0.85:
        target_action = 2  # Add server
        target_servers = req.current_servers + 1
    elif effective_future_load < (req.current_servers - 1) * req.capacity_per_server * 0.65:
        target_action = 0  # Remove server
        target_servers = req.current_servers - 1
    else:
        target_action = 1  # Hold
        target_servers = req.current_servers

    # Pass through DeploymentAgent safety validation
    deployer = DeploymentAgent(min_servers=1, max_servers=10, dry_run=True)
    safe_servers, safety_status = deployer.validate_action(req.current_servers, target_servers)

    action_map = {0: "remove", 1: "hold", 2: "add"}
    actual_action = 2 if safe_servers > req.current_servers else (0 if safe_servers < req.current_servers else 1)

    cost_change = (safe_servers - req.current_servers) * req.price_per_server
    new_capacity = safe_servers * req.capacity_per_server
    headroom = new_capacity - max(req.current_demand, req.forecast_demand)
    sla_risk = "high" if headroom < 0 else ("moderate" if headroom < 25 else "low")

    row = {
        "demand": req.current_demand,
        "forecast": req.forecast_demand,
        "servers_before": req.current_servers,
        "servers_after": safe_servers,
        "action": action_map[actual_action],
        "cost_change": cost_change,
        "sla_risk": sla_risk,
    }

    explanation = explain_decision(row)
    kubectl_cmd = deployer.generate_kubectl(safe_servers)

    return {
        "servers_before": req.current_servers,
        "servers_after": safe_servers,
        "action": action_map[actual_action],
        "safety_status": safety_status,
        "explanation": explanation,
        "kubectl_command": kubectl_cmd,
        "hourly_cost": safe_servers * req.price_per_server,
        "capacity": new_capacity,
        "utilization_pct": round((req.current_demand / new_capacity * 100) if new_capacity > 0 else 0, 1),
    }


# Mount frontend static directory if present
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
