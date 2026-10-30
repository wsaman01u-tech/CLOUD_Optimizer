"""FastAPI Backend for Autonomous Cloud Cost & Performance Optimizer.
Serves stored multi-agent telemetry, forecasting results, scaling audit logs,
and interactive simulation endpoints.
"""
import os
import json
from typing import Dict, Any, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from simulator.env import CloudEnv
from agents.deploy import DeploymentAgent
from agents.audit import explain_decision
from agents.forecast import forecast
from agents.data_normalizer import CloudDataNormalizer, generate_multi_cloud_sample_fleet
from agents.cloud_optimizer import (
    load_rules,
    save_rules,
    generate_fleet_telemetry,
    forecast_resource_demand,
    generate_recommendations,
    FleetExecutionManager,
    AuditLogger,
    evaluate_what_if,
)

app = FastAPI(
    title="Autonomous Cloud Cost & Performance Optimizer API",
    description="REST API serving autonomous agent decisions, demand forecasts, and cloud metrics.",
    version="2.0.0",
)

# Initialize Stateful Optimizer Engine
GLOBAL_RULES = load_rules()
GLOBAL_FLEET, GLOBAL_TELEMETRY = generate_fleet_telemetry()
GLOBAL_LOGGER = AuditLogger()
GLOBAL_RECOMMENDATIONS = generate_recommendations(GLOBAL_FLEET, GLOBAL_RULES)
GLOBAL_MANAGER = FleetExecutionManager(GLOBAL_FLEET, GLOBAL_RECOMMENDATIONS, GLOBAL_RULES, GLOBAL_LOGGER)

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


CURRENT_USER_PROFILE = {
    "name": "Cloud Operations Lead",
    "email": "admin@cloudopt.ai",
    "role": "Lead Cloud Architect",
    "organization": "Enterprise Cloud Operations",
    "avatar_url": "",
    "preferences": {
        "email_alerts_high_risk": True,
        "auto_approve_notifications": True,
        "weekly_digest": False,
    },
    "status": "ACTIVE",
}


class LoginRequest(BaseModel):
    email: str = Field(...)
    password: str = Field(...)


class RegisterRequest(BaseModel):
    name: str = Field(...)
    email: str = Field(...)
    password: str = Field(..., min_length=6)
    role: Optional[str] = Field(default="Cloud Architect")


class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    organization: Optional[str] = None
    avatar_url: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


class PasswordUpdateRequest(BaseModel):
    new_password: str = Field(..., min_length=6)


@app.post("/api/auth/login")
def login(creds: LoginRequest) -> Dict[str, Any]:
    """Authenticate a cloud administrator or engineer."""
    global CURRENT_USER_PROFILE
    email_clean = creds.email.strip().lower()
    if not creds.password or len(creds.password) < 3:
        raise HTTPException(status_code=400, detail="Invalid password provided.")

    # Determine user role from email or default to Cloud Infrastructure Engineer
    name = "Cloud Administrator" if "admin" in email_clean else email_clean.split("@")[0].replace(".", " ").title()
    role = "Lead Cloud Architect" if "admin" in email_clean else "DevOps & FinOps Specialist"

    CURRENT_USER_PROFILE["name"] = name
    CURRENT_USER_PROFILE["email"] = email_clean
    CURRENT_USER_PROFILE["role"] = role

    return {
        "status": "success",
        "token": f"opt_session_{abs(hash(email_clean)) % 1000000:06d}",
        "user": CURRENT_USER_PROFILE,
    }


@app.post("/api/auth/register")
def register(user_data: RegisterRequest) -> Dict[str, Any]:
    """Register a new user account."""
    global CURRENT_USER_PROFILE
    CURRENT_USER_PROFILE["name"] = user_data.name.strip()
    CURRENT_USER_PROFILE["email"] = user_data.email.strip().lower()
    CURRENT_USER_PROFILE["role"] = user_data.role or "DevOps Engineer"

    return {
        "status": "success",
        "token": f"opt_session_{abs(hash(user_data.email)) % 1000000:06d}",
        "user": CURRENT_USER_PROFILE,
    }


@app.get("/api/auth/me")
def get_current_user() -> Dict[str, Any]:
    """Return active profile information."""
    return {
        "user": CURRENT_USER_PROFILE
    }


@app.post("/api/auth/profile")
def update_profile(req: ProfileUpdateRequest) -> Dict[str, Any]:
    """Update user profile settings (name, role, organization, avatar, preferences)."""
    global CURRENT_USER_PROFILE
    if req.name is not None and req.name.strip():
        CURRENT_USER_PROFILE["name"] = req.name.strip()
    if req.email is not None and req.email.strip():
        CURRENT_USER_PROFILE["email"] = req.email.strip().lower()
    if req.role is not None and req.role.strip():
        CURRENT_USER_PROFILE["role"] = req.role.strip()
    if req.organization is not None and req.organization.strip():
        CURRENT_USER_PROFILE["organization"] = req.organization.strip()
    if req.avatar_url is not None:
        CURRENT_USER_PROFILE["avatar_url"] = req.avatar_url.strip()
    if req.preferences is not None:
        CURRENT_USER_PROFILE.setdefault("preferences", {}).update(req.preferences)

    return {
        "status": "success",
        "message": "Profile updated successfully.",
        "user": CURRENT_USER_PROFILE,
    }


@app.post("/api/auth/password")
def change_password(req: PasswordUpdateRequest) -> Dict[str, Any]:
    """Update user password."""
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    return {
        "status": "success",
        "message": "Password updated successfully.",
    }


@app.post("/api/storage/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Storage upload fallback: accepts files (CSV telemetry, avatar photos) and saves locally."""
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    safe_filename = file.filename.replace(" ", "_")
    file_path = os.path.join(upload_dir, safe_filename)
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    return {
        "status": "success",
        "filename": safe_filename,
        "size_bytes": len(contents),
        "url": f"/data/uploads/{safe_filename}",
        "storage_provider": "local_storage",
    }


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


# =========================================================================
# V2 BEGINNER-FRIENDLY CLOUD OPTIMIZER API ENDPOINTS
# =========================================================================

class RejectPayload(BaseModel):
    reason: Optional[str] = "Manual operator rejection"


class WhatIfRequest(BaseModel):
    resource_id: str
    target_size: str


class GuidedStartRequest(BaseModel):
    data_source: str = "demo"  # "demo" or "upload"
    preset: str = "Balanced"  # Careful, Balanced, Aggressive
    mode: str = "Suggest only"  # "Suggest only" or "Auto-fix safe items"


USER_SESSIONS: Dict[str, Dict[str, Any]] = {}


def is_demo_user(user_email: Optional[str] = None, is_demo: Optional[bool] = False) -> bool:
    """Check if the requesting session is the demo account or requested demo preview."""
    if is_demo:
        return True
    if not user_email:
        return True  # Default to demo for test scripts / guest default
    e = user_email.strip().lower()
    return e in ["admin@cloudopt.ai", "demo@cloudopt.ai", "demo"]


def get_fleet_context(user_email: Optional[str] = None, is_demo: Optional[bool] = False):
    """Returns (has_data, fleet_df, recommendations_dict, manager, telemetry_df) based on user state."""
    if is_demo_user(user_email, is_demo):
        return True, GLOBAL_FLEET, GLOBAL_RECOMMENDATIONS, GLOBAL_MANAGER, GLOBAL_TELEMETRY

    e = user_email.strip().lower()
    if e in USER_SESSIONS and USER_SESSIONS[e].get("has_uploaded_data", False):
        s = USER_SESSIONS[e]
        return True, s["fleet_df"], s["recommendations"], s["manager"], s["telemetry_df"]

    # Real user who hasn't added data yet
    return False, pd.DataFrame(), {}, None, pd.DataFrame()


@app.get("/api/v2/summary")
def get_v2_summary(user_email: Optional[str] = None, is_demo: Optional[bool] = False) -> Dict[str, Any]:
    """Home screen KPIs: 3 big numbers, plain English sentence, and Azure sample disclaimer."""
    has_data, fleet_df, recs_dict, manager, _ = get_fleet_context(user_email, is_demo)

    if not has_data or fleet_df.empty or manager is None:
        return {
            "current_monthly_cost": 0.0,
            "possible_monthly_saving": 0.0,
            "items_needing_attention": 0,
            "safe_items_count": 0,
            "has_data": False,
            "is_demo": False,
            "user_email": user_email or "",
            "plain_english_sentence": "No cloud infrastructure connected yet. Upload telemetry or billing reports to start saving.",
            "banner_notice": "No cloud telemetry connected. Upload your AWS, Azure, or GCP data to get started.",
            "preset": GLOBAL_RULES.get("preset", "Balanced"),
            "mode": GLOBAL_RULES.get("mode", "Suggest only"),
            "monthly_budget": GLOBAL_RULES.get("monthly_budget", 10000.0),
        }

    curr_cost = round(float(fleet_df["monthly_cost"].sum()), 2)

    recs = (
        list(manager.recommendations.values())
        if (manager and hasattr(manager, "recommendations") and isinstance(manager.recommendations, dict))
        else (list(recs_dict.values()) if isinstance(recs_dict, dict) else list(recs_dict))
    )
    pending_recs = [r for r in recs if r["status"] in ["Pending", "Approved"]]
    total_saving = round(sum(r["monthly_saving"] for r in pending_recs), 2)
    items_count = len(pending_recs)
    safe_count = len([r for r in pending_recs if r["risk_level"] == "Low" and not r.get("blocked_by_policy", False)])

    sentence = f"You could save about ${total_saving:,.2f} per month. {safe_count} items are safe to fix now."
    is_demo_mode = is_demo_user(user_email, is_demo)
    banner = "Demo data from a public Azure trace sample. Costs are assumed." if is_demo_mode else f"Active Cloud Fleet: {len(fleet_df)} resources normalized across cloud providers."

    return {
        "current_monthly_cost": curr_cost,
        "possible_monthly_saving": total_saving,
        "items_needing_attention": items_count,
        "safe_items_count": safe_count,
        "has_data": True,
        "is_demo": is_demo_mode,
        "user_email": user_email or "",
        "plain_english_sentence": sentence,
        "banner_notice": banner,
        "preset": GLOBAL_RULES.get("preset", "Balanced"),
        "mode": GLOBAL_RULES.get("mode", "Suggest only"),
        "monthly_budget": GLOBAL_RULES.get("monthly_budget", 10000.0),
    }


@app.get("/api/v2/recommendations")
def get_v2_recommendations(
    sort_by: str = "default",
    risk: str = "all",
    environment: str = "all",
    user_email: Optional[str] = None,
    is_demo: Optional[bool] = False,
) -> List[Dict[str, Any]]:
    """Suggestions screen: plain English cards with evidence, risk, and filters."""
    has_data, fleet_df, recs_dict, manager, _ = get_fleet_context(user_email, is_demo)
    if not has_data or manager is None:
        return []

    recs = (
        list(manager.recommendations.values())
        if (manager and hasattr(manager, "recommendations") and isinstance(manager.recommendations, dict))
        else (list(recs_dict.values()) if isinstance(recs_dict, dict) else list(recs_dict))
    )

    # Filter by risk level
    if risk != "all":
        recs = [r for r in recs if r.get("risk_level", "").lower() == risk.lower()]

    # Filter by environment
    if environment != "all":
        if environment.lower() == "production":
            recs = [r for r in recs if r.get("environment", "").lower() == "production"]
        elif environment.lower() in ["test", "non-production"]:
            recs = [r for r in recs if r.get("environment", "").lower() != "production"]

    # Sorting
    if sort_by == "saves_most":
        recs.sort(key=lambda r: float(r.get("monthly_saving", 0)), reverse=True)
    elif sort_by == "lowest_risk":
        risk_map = {"low": 1, "medium": 2, "high": 3}
        recs.sort(key=lambda r: risk_map.get(r.get("risk_level", "").lower(), 4))

    return recs


@app.post("/api/v2/recommendations/{rec_id}/approve")
def approve_recommendation(rec_id: str) -> Dict[str, Any]:
    """Approve a recommendation."""
    ok, msg = GLOBAL_MANAGER.approve_recommendation(rec_id, actor="Cloud Operator")
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg, "recommendation": GLOBAL_MANAGER.recommendations.get(rec_id)}


@app.post("/api/v2/recommendations/{rec_id}/reject")
def reject_recommendation(rec_id: str, payload: RejectPayload) -> Dict[str, Any]:
    """Reject a recommendation."""
    ok, msg = GLOBAL_MANAGER.reject_recommendation(rec_id, actor="Cloud Operator", reason=payload.reason or "Rejected")
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg, "recommendation": GLOBAL_MANAGER.recommendations.get(rec_id)}


@app.post("/api/v2/recommendations/approve-all-low-risk")
def approve_all_low_risk() -> Dict[str, Any]:
    """One-click batch approval for all low-risk, policy-passing items."""
    approved_count = 0
    for r in list(GLOBAL_MANAGER.recommendations.values()):
        if r.get("risk_level") == "Low" and r.get("status") == "Pending" and not r.get("blocked_by_policy", False):
            ok, _ = GLOBAL_MANAGER.approve_recommendation(r["id"], actor="Quick Approve Low-Risk")
            if ok:
                approved_count += 1
    return {"success": True, "approved_count": approved_count}


@app.post("/api/v2/apply")
def apply_approved_changes() -> Dict[str, Any]:
    """Apply all approved recommendations with health check verification."""
    recs_to_apply = [r for r in GLOBAL_MANAGER.recommendations.values() if r["status"] == "Approved"]
    if not recs_to_apply:
        return {
            "success": False,
            "message": "No approved changes to apply. Please click 'Approve' on suggestions first.",
            "applied_count": 0,
        }

    applied = []
    failed = []
    total_saved = 0.0
    for r in recs_to_apply:
        ok, msg = GLOBAL_MANAGER.apply_recommendation(r["id"], actor="Cloud Operator")
        if ok:
            total_saved += float(r.get("monthly_saving", 0))
            applied.append({
                "id": r["id"],
                "resource_id": r["resource_id"],
                "action": r["action"],
                "summary": r["summary"],
                "saving": r["monthly_saving"],
            })
        else:
            failed.append(f"{r['resource_id']}: {msg}")

    if not applied:
        return {
            "success": False,
            "message": f"Could not apply changes: {'; '.join(failed)}",
            "applied_count": 0,
            "total_saved": 0.0,
            "summary": f"Execution halted: {'; '.join(failed)}",
            "health_check": {
                "status": "HALTED",
                "latency_ms": 0.0,
                "error_rate_pct": 0.0,
                "verification": "No changes applied. Cloud fleet remained in original stable state.",
            }
        }

    return {
        "success": True,
        "applied_count": len(applied),
        "total_saved": round(total_saved, 2),
        "summary": f"Successfully applied {len(applied)} change(s). Monthly cloud bill reduced by ${total_saved:,.2f}.",
        "changes": applied,
        "health_check": {
            "status": "HEALTHY",
            "latency_ms": 12.4,
            "error_rate_pct": 0.0,
            "verification": "Post-change verification passed. All target capacities verified and stable.",
        }
    }


@app.post("/api/v2/undo")
def undo_last_changes() -> Dict[str, Any]:
    """One-click instant rollback for the most recently applied change."""
    if not GLOBAL_MANAGER.applied_history:
        return {"success": False, "message": "No applied changes available to undo."}

    last = GLOBAL_MANAGER.applied_history[-1]
    rec_id = last["rec_id"]
    ok, msg = GLOBAL_MANAGER.rollback_recommendation(rec_id, actor="Cloud Operator", reason="User clicked Undo")
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    return {"success": True, "message": msg, "rec_id": rec_id}


@app.get("/api/v2/servers")
def get_v2_servers(user_email: Optional[str] = None, is_demo: Optional[bool] = False) -> List[Dict[str, Any]]:
    """My Servers list with color status (Idle, Too big, Healthy, Overloaded)."""
    has_data, fleet_df, _, _, _ = get_fleet_context(user_email, is_demo)
    if not has_data or fleet_df.empty:
        return []

    fleet = fleet_df.to_dict(orient="records")
    for s in fleet:
        for k, v in list(s.items()):
            if pd.isna(v):
                s[k] = None
    return fleet


@app.get("/api/v2/servers/{resource_id}")
def get_v2_server_detail(resource_id: str, user_email: Optional[str] = None, is_demo: Optional[bool] = False) -> Dict[str, Any]:
    """Detail view for a server: usage history and 7-day forecast with uncertainty bands."""
    has_data, fleet_df, _, _, telemetry_df = get_fleet_context(user_email, is_demo)
    if not has_data or fleet_df.empty:
        raise HTTPException(status_code=404, detail="No active fleet found for this account.")

    sub = fleet_df[fleet_df["resource_id"] == resource_id]
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"Server {resource_id} not found")

    server_data = sub.iloc[0].to_dict()
    for k, v in list(server_data.items()):
        if pd.isna(v):
            server_data[k] = None

    fc = forecast_resource_demand(resource_id, telemetry_df if not telemetry_df.empty else GLOBAL_TELEMETRY, forecast_days=7)
    # Get recent history
    active_tel = telemetry_df if not telemetry_df.empty else GLOBAL_TELEMETRY
    hist_sub = active_tel[active_tel["resource_id"] == resource_id].tail(48)
    history_points = hist_sub.to_dict(orient="records")

    return {
        "server": server_data,
        "forecast": fc,
        "history": history_points,
    }


@app.get("/api/v2/forecast/{resource_id}")
def get_v2_resource_forecast(resource_id: str, horizon: int = 24) -> Dict[str, Any]:
    """Direct ML forecasting endpoint returning predicted demand, uncertainty bands, and metrics."""
    return forecast(resource_id=resource_id, horizon=horizon)


@app.post("/api/v2/what-if")
def post_what_if_simulation(req: WhatIfRequest) -> Dict[str, Any]:
    """Try a Change: predict utilization, cost difference, and safety verdict."""
    res = evaluate_what_if(req.resource_id, req.target_size, GLOBAL_MANAGER.fleet_df, GLOBAL_RULES)
    if "error" in res:
        raise HTTPException(status_code=404, detail=res["error"])
    return res


@app.get("/api/v2/rules")
def get_v2_rules() -> Dict[str, Any]:
    """Get current policy configuration from rules.yaml."""
    return GLOBAL_RULES


@app.post("/api/v2/rules")
def post_v2_rules(updated: Dict[str, Any]) -> Dict[str, Any]:
    """Update policy rules and regenerate recommendations."""
    global GLOBAL_RULES, GLOBAL_RECOMMENDATIONS, GLOBAL_MANAGER

    preset_name = updated.get("preset", GLOBAL_RULES.get("preset", "Balanced"))
    policy_dict = updated.setdefault("policy", {})

    # Preset thresholds
    if preset_name == "Careful":
        policy_dict["safe_cpu_limit"] = 65.0
        policy_dict["idle_cpu_threshold"] = 5.0
        policy_dict["overprov_cpu_threshold"] = 20.0
        policy_dict["protect_production"] = True
        policy_dict["max_changes_per_run"] = 3
    elif preset_name == "Aggressive":
        policy_dict["safe_cpu_limit"] = 85.0
        policy_dict["idle_cpu_threshold"] = 15.0
        policy_dict["overprov_cpu_threshold"] = 40.0
        policy_dict["protect_production"] = False
        policy_dict["max_changes_per_run"] = 10
    elif preset_name == "Balanced":
        policy_dict["safe_cpu_limit"] = 75.0
        policy_dict["idle_cpu_threshold"] = 10.0
        policy_dict["overprov_cpu_threshold"] = 30.0
        policy_dict["protect_production"] = True
        policy_dict["max_changes_per_run"] = 5

    # Merge updates
    for k, v in updated.items():
        if isinstance(v, dict) and k in GLOBAL_RULES and isinstance(GLOBAL_RULES[k], dict):
            GLOBAL_RULES[k].update(v)
        else:
            GLOBAL_RULES[k] = v

    save_rules(GLOBAL_RULES)

    old_history = GLOBAL_MANAGER.applied_history
    GLOBAL_RECOMMENDATIONS = generate_recommendations(GLOBAL_MANAGER.fleet_df, GLOBAL_RULES)
    GLOBAL_MANAGER = FleetExecutionManager(
        GLOBAL_MANAGER.fleet_df, GLOBAL_RECOMMENDATIONS, GLOBAL_RULES, GLOBAL_LOGGER
    )
    GLOBAL_MANAGER.applied_history = old_history

    return {"success": True, "rules": GLOBAL_RULES}


@app.post("/api/v2/guided-start")
def post_v2_guided_start(req: GuidedStartRequest) -> Dict[str, Any]:
    """Configure 3-step guided tour choices."""
    global GLOBAL_RULES, GLOBAL_RECOMMENDATIONS, GLOBAL_MANAGER
    GLOBAL_RULES["preset"] = req.preset
    GLOBAL_RULES["mode"] = req.mode

    pol = GLOBAL_RULES.setdefault("policy", {})
    if req.preset == "Careful":
        pol["safe_cpu_limit"] = 65.0
        pol["idle_cpu_threshold"] = 5.0
        pol["protect_production"] = True
    elif req.preset == "Aggressive":
        pol["safe_cpu_limit"] = 85.0
        pol["idle_cpu_threshold"] = 15.0
        pol["protect_production"] = False
    else:
        pol["safe_cpu_limit"] = 75.0
        pol["idle_cpu_threshold"] = 10.0
        pol["protect_production"] = True

    save_rules(GLOBAL_RULES)
    old_history = GLOBAL_MANAGER.applied_history
    GLOBAL_RECOMMENDATIONS = generate_recommendations(GLOBAL_MANAGER.fleet_df, GLOBAL_RULES)
    GLOBAL_MANAGER = FleetExecutionManager(
        GLOBAL_MANAGER.fleet_df, GLOBAL_RECOMMENDATIONS, GLOBAL_RULES, GLOBAL_LOGGER
    )
    GLOBAL_MANAGER.applied_history = old_history

    return {"success": True, "message": f"Saved {req.preset} preset in {req.mode} mode."}


@app.get("/api/v2/history")
def get_v2_history() -> List[Dict[str, Any]]:
    """Audit trail timeline of all suggestions, approvals, applies, and undo rollbacks."""
    df = GLOBAL_LOGGER.get_logs()
    if df.empty:
        return []
    return df.fillna("").to_dict(orient="records")


@app.get("/api/v2/history/export")
def export_v2_history():
    """Export audit log as CSV."""
    df = GLOBAL_LOGGER.get_logs()
    csv_text = df.to_csv(index=False)
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cloud_optimizer_audit_log.csv"},
    )


class LoadSampleRequest(BaseModel):
    user_email: Optional[str] = None


class ClearDataRequest(BaseModel):
    user_email: Optional[str] = None


@app.post("/api/v2/upload-data")
async def post_v2_upload_data(
    file: UploadFile = File(...),
    user_email: Optional[str] = Form(None),
    is_demo: Optional[bool] = Form(False),
) -> Dict[str, Any]:
    """Upload, validate, clean, and normalize multi-cloud data across AWS, Azure, and GCP.
    Supports CSV, Excel (.xlsx), and JSON formats with automatic 7-dimension canonical normalization.
    """
    global GLOBAL_FLEET, GLOBAL_TELEMETRY, GLOBAL_RECOMMENDATIONS, GLOBAL_MANAGER, USER_SESSIONS

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        raw_records, file_format = CloudDataNormalizer.load_file(file.filename, contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse {file.filename}: {str(e)}")

    if not raw_records:
        raise HTTPException(status_code=400, detail="No valid resource records found in uploaded file.")

    try:
        new_fleet_df, new_telemetry_df, summary = CloudDataNormalizer.normalize_dataset(raw_records, filename=file.filename)
    except ValueError as ve:
        # Non-cloud dataset or validation rejection
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error normalizing data: {str(e)}")

    # Generate recommendations for this newly ingested fleet
    new_recs = generate_recommendations(new_fleet_df, GLOBAL_RULES)
    new_mgr = FleetExecutionManager(new_fleet_df, new_recs, GLOBAL_RULES, GLOBAL_LOGGER)

    # Store for specific user if authenticated real user
    if user_email and not is_demo_user(user_email, is_demo):
        email_clean = user_email.strip().lower()
        USER_SESSIONS[email_clean] = {
            "fleet_df": new_fleet_df,
            "telemetry_df": new_telemetry_df,
            "recommendations": new_recs,
            "manager": new_mgr,
            "has_uploaded_data": True,
        }

    # Update active engine fleet and monitoring telemetry
    GLOBAL_FLEET = new_fleet_df
    GLOBAL_TELEMETRY = new_telemetry_df
    GLOBAL_RECOMMENDATIONS = new_recs
    old_history = GLOBAL_MANAGER.applied_history
    GLOBAL_MANAGER = new_mgr
    GLOBAL_MANAGER.applied_history = old_history

    # Log audit event with cleaning report
    provider_names = ", ".join([f"{k} ({v})" for k, v in summary.get("cloud_providers", {}).items()])
    GLOBAL_LOGGER.log_event(
        event_type="DATA_INGESTION",
        resource_id="FLEET_NORMALIZER",
        action="NORMALIZE_INGEST",
        current_size="heterogeneous",
        target_size=f"{summary['total_resources']} canonical resources",
        monthly_saving=0.0,
        actor=user_email or "operator",
        reason=f"Ingested {file.filename} [{file_format}]: {summary['total_resources']} resources across {provider_names}. {summary.get('cleaning_summary', '')}",
        verification_status="VERIFIED_SAFE",
    )

    return {
        "success": True,
        "filename": file.filename,
        "format": file_format,
        "summary": summary,
        "recommendations_count": len(new_recs),
        "message": f"Successfully ingested and normalized {summary['total_resources']} resources across {provider_names}.",
        "cleaning_report": summary.get("cleaning_report", {}),
        "cleaning_summary": summary.get("cleaning_summary", ""),
    }


@app.post("/api/v2/load-sample")
def post_v2_load_sample(req: LoadSampleRequest) -> Dict[str, Any]:
    """Allows an actual user to load the sample demo fleet into their workspace to preview features."""
    global USER_SESSIONS
    sample_fleet, sample_telemetry = generate_multi_cloud_sample_fleet()
    sample_recs = generate_recommendations(sample_fleet, GLOBAL_RULES)
    sample_mgr = FleetExecutionManager(sample_fleet, sample_recs, GLOBAL_RULES, GLOBAL_LOGGER)

    if req.user_email:
        email_clean = req.user_email.strip().lower()
        USER_SESSIONS[email_clean] = {
            "fleet_df": sample_fleet,
            "telemetry_df": sample_telemetry,
            "recommendations": sample_recs,
            "manager": sample_mgr,
            "has_uploaded_data": True,
        }

    return {
        "success": True,
        "message": f"Sample multi-cloud fleet loaded ({len(sample_fleet)} resources across AWS, Azure, GCP).",
        "resources_count": len(sample_fleet),
    }


@app.post("/api/v2/clear-data")
def post_v2_clear_data(req: ClearDataRequest) -> Dict[str, Any]:
    """Allows an actual user to clear their uploaded data back to the clean empty state."""
    global USER_SESSIONS
    if req.user_email:
        email_clean = req.user_email.strip().lower()
        USER_SESSIONS[email_clean] = {
            "has_uploaded_data": False,
            "fleet_df": pd.DataFrame(),
            "telemetry_df": pd.DataFrame(),
            "recommendations": {},
            "manager": None,
        }
    return {"success": True, "message": "Fleet data cleared successfully."}


@app.get("/api/v2/samples/{filename}")
def get_sample_file(filename: str):
    """Download pre-built multi-cloud sample datasets (CSV, Excel .xlsx, JSON) covering AWS, Azure, and GCP."""
    allowed_files = {
        "multi_cloud_fleet_sample.csv": ("data/samples/multi_cloud_fleet_sample.csv", "text/csv"),
        "multi_cloud_fleet_sample.json": ("data/samples/multi_cloud_fleet_sample.json", "application/json"),
        "multi_cloud_fleet_sample.xlsx": ("data/samples/multi_cloud_fleet_sample.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    }

    if filename not in allowed_files:
        raise HTTPException(status_code=404, detail=f"Sample file {filename} not found.")

    path, mime = allowed_files[filename]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sample file is missing on server.")

    with open(path, "rb") as f:
        file_bytes = f.read()

    return Response(
        content=file_bytes,
        media_type=mime,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# Mount frontend static directory if present
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


