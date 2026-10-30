"""Unit and integration tests for Autonomous Cloud Cost & Performance Optimizer.
Tests all five autonomous agents, the cloud simulator, safety bounds, and API endpoints.
"""
import os
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from simulator.env import CloudEnv
from agents.monitor import summarize_vms, compute_fleet_kpis
from agents.forecast import rmse, mape, mae, Forecaster, forecast
from agents.optimizer import static_policy, threshold_policy, train_q_learning, q_policy, evaluate_policy
from agents.deploy import DeploymentAgent
from agents.audit import explain_decision, AuditAgent
from api.main import app


# -------------------------------------------------------------
# 1. Simulator Tests
# -------------------------------------------------------------
def test_cost_is_servers_times_price():
    """Verify that hourly cost directly equals active server count * price."""
    env = CloudEnv([100, 100, 100], capacity=100, price=1.0)
    _, _, _, info = env.step(1)  # Hold action
    assert info["cost"] == 3.0  # Initial servers = 3


def test_violation_when_demand_exceeds_capacity():
    """Verify that an SLA violation flag is raised when demand > total capacity."""
    env = CloudEnv([1000, 1000], capacity=100, max_servers=10, initial_servers=3)
    _, _, _, info = env.step(1)
    # 3 servers * 100 capacity = 300 capacity < 1000 demand => violation
    assert info["violation"] == 1


def test_servers_never_below_min():
    """Verify that scaling down cannot breach the minimum server boundary."""
    env = CloudEnv([100] * 10, min_servers=1, initial_servers=3)
    for _ in range(8):
        env.step(0)  # Continuous scale down
    assert env.servers == 1


def test_servers_never_above_max():
    """Verify that scaling up cannot breach the maximum server boundary."""
    env = CloudEnv([100] * 15, max_servers=10, initial_servers=3)
    for _ in range(15):
        env.step(2)  # Continuous scale up
    assert env.servers == 10


# -------------------------------------------------------------
# 2. Monitoring Agent Tests
# -------------------------------------------------------------
def test_monitoring_idle_detection():
    """Verify that a machine with 2% CPU utilization is correctly flagged as IDLE."""
    df = pd.DataFrame({
        "vm_id": ["vm-test-idle"] * 100,
        "avg_cpu": [2.0] * 100,
        "min_cpu": [1.0] * 100,
        "max_cpu": [3.0] * 100,
    })
    summary = summarize_vms(df, idle_limit=10.0, low_peak_limit=40.0)
    assert len(summary) == 1
    assert summary.iloc[0]["status"] == "IDLE"
    assert summary.iloc[0]["is_idle"] == True
    assert summary.iloc[0]["estimated_waste"] > 0


def test_monitoring_overprovisioned_detection():
    """Verify that a machine with peak CPU under 40% but not idle is flagged as OVER_PROVISIONED."""
    df = pd.DataFrame({
        "vm_id": ["vm-test-overprov"] * 100,
        "avg_cpu": [20.0] * 100,
        "min_cpu": [15.0] * 100,
        "max_cpu": [28.0] * 100,
    })
    summary = summarize_vms(df, idle_limit=10.0, low_peak_limit=40.0)
    assert len(summary) == 1
    assert summary.iloc[0]["status"] == "OVER_PROVISIONED"
    assert summary.iloc[0]["is_over_provisioned"] == True


# -------------------------------------------------------------
# 3. Demand Forecasting Agent Tests
# -------------------------------------------------------------
def test_forecasting_error_metrics():
    """Verify RMSE, MAE, and MAPE calculations."""
    actual = np.array([100.0, 200.0, 300.0])
    pred = np.array([110.0, 190.0, 310.0])
    assert rmse(actual, pred) == pytest.approx(10.0, 0.01)
    assert mae(actual, pred) == pytest.approx(10.0, 0.01)
    assert mape(actual, pred) > 0


def test_forecaster_fit_and_predict():
    """Verify Forecaster class trains and outputs valid future predictions."""
    history = pd.Series([100 + 50 * np.sin(i / 12 * np.pi) for i in range(100)])
    fc = Forecaster(n_lags=12)
    fc.fit(history)
    next_demand = fc.predict_next(history)
    assert isinstance(next_demand, float)
    assert next_demand > 0


def test_production_forecast_function():
    """Verify new forecast(resource_id, horizon) returns required structure and metrics."""
    res = forecast(resource_id="ec2-prod-web-01", horizon=24)
    assert "predicted_value" in res
    assert "lower_band" in res
    assert "upper_band" in res
    assert "model_name" in res
    assert "model_version" in res
    assert "error_metrics" in res
    assert res["model_name"] == "LightGBM_Demand_Forecaster"
    assert res["model_version"] == "v2.0.0"
    assert "MAE" in res["error_metrics"]
    assert "sMAPE" in res["error_metrics"]
    assert len(res["points"]) == 24
    assert res["upper_band"] >= res["predicted_value"] >= res["lower_band"]


def test_forecast_api_endpoints():
    """Verify backend forecast endpoints return correct data."""
    client = TestClient(app)
    # Direct forecast endpoint
    resp = client.get("/api/v2/forecast/ec2-prod-web-01?horizon=24")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "LightGBM_Demand_Forecaster"
    assert len(data["points"]) == 24

    # Server detail view with 7-day forecast
    resp_server = client.get("/api/v2/servers/ec2-prod-web-01")
    assert resp_server.status_code == 200
    s_data = resp_server.json()
    assert "forecast" in s_data
    assert "points" in s_data["forecast"]


# -------------------------------------------------------------
# 4. Deployment Agent Tests
# -------------------------------------------------------------
def test_deployment_safety_rate_limiter():
    """Verify rate limiter permits at most +1 / -1 replica change per step."""
    deployer = DeploymentAgent(min_servers=1, max_servers=10, dry_run=True)
    # Requested jump from 3 to 8
    safe_count, status = deployer.validate_action(servers_current=3, target_servers=8)
    assert safe_count == 4
    assert "Rate limited" in status

    # Requested jump from 7 to 1
    safe_count, status = deployer.validate_action(servers_current=7, target_servers=1)
    assert safe_count == 6
    assert "Rate limited" in status


def test_deployment_generates_terraform_and_kubectl():
    """Verify deployment agent outputs valid Terraform files and kubectl command."""
    deployer = DeploymentAgent(output_dir="outputs/test_tf")
    record = deployer.execute_scaling(servers_current=3, target_servers=4)
    assert record["servers_deployed"] == 4
    assert record["kubectl_command"] == "kubectl scale deployment cloud-worker-deployment --replicas=4"
    assert os.path.exists("outputs/test_tf/terraform.tfvars")
    assert os.path.exists("outputs/test_tf/variables.tf")


# -------------------------------------------------------------
# 5. Audit & Explainability Agent Tests
# -------------------------------------------------------------
def test_explain_decision_produces_rationale():
    """Verify explainability engine produces readable justification for scaling."""
    row = {
        "demand": 450,
        "forecast": 520,
        "servers_before": 4,
        "servers_after": 5,
        "action": "add",
        "cost_change": 1.0,
        "sla_risk": "low",
    }
    explanation = explain_decision(row)
    assert "scale up" in explanation
    assert "520.0" in explanation
    assert "$1.00" in explanation


# -------------------------------------------------------------
# 6. FastAPI Backend Tests
# -------------------------------------------------------------
def test_fastapi_health_endpoint():
    """Verify /health endpoint returns HTTP 200 and ok status."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_fastapi_overview_endpoint():
    """Verify /api/overview returns required KPI metrics."""
    client = TestClient(app)
    response = client.get("/api/overview")
    assert response.status_code == 200
    data = response.json()
    assert "cost_savings_pct" in data
    assert "sla_compliance_pct" in data


def test_fastapi_simulate_step_endpoint():
    """Verify POST /api/simulate-step computes real-time decision."""
    client = TestClient(app)
    payload = {
        "current_servers": 3,
        "current_demand": 280,
        "forecast_demand": 380,
    }
    response = client.post("/api/simulate-step", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "action" in data
    assert "servers_after" in data
    assert "explanation" in data
    assert "kubectl_command" in data


def test_fastapi_auth_login_and_me():
    """Verify login and me endpoints."""
    client = TestClient(app)
    # Login
    res = client.post("/api/auth/login", json={"email": "admin@cloudopt.ai", "password": "demo123"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "token" in data
    assert "user" in data

    # Me
    res_me = client.get("/api/auth/me")
    assert res_me.status_code == 200
    assert "user" in res_me.json()


def test_fastapi_profile_update_and_storage_upload():
    """Verify profile updates, password change, and file storage uploads."""
    client = TestClient(app)

    # 1. Profile Update
    update_payload = {
        "name": "Sarah Connor",
        "email": "sarah.connor@cyberdyne.cloud",
        "role": "Principal FinOps Architect",
        "organization": "Skynet Defense Systems",
        "preferences": {
            "email_alerts_high_risk": True,
            "weekly_digest": True,
        }
    }
    res_prof = client.post("/api/auth/profile", json=update_payload)
    assert res_prof.status_code == 200
    user_data = res_prof.json()["user"]
    assert user_data["name"] == "Sarah Connor"
    assert user_data["email"] == "sarah.connor@cyberdyne.cloud"
    assert user_data["role"] == "Principal FinOps Architect"
    assert user_data["preferences"]["weekly_digest"] is True

    # 2. Password Change
    res_pass = client.post("/api/auth/password", json={"new_password": "secureNewPassword123!"})
    assert res_pass.status_code == 200
    assert res_pass.json()["status"] == "success"

    # 3. Storage Upload
    file_content = b"timestamp,resource_id,cpu_utilization\n1700000000,server-1,45.2\n"
    res_upload = client.post(
        "/api/storage/upload",
        files={"file": ("test_trace.csv", file_content, "text/csv")}
    )
    assert res_upload.status_code == 200
    up_data = res_upload.json()
    assert up_data["status"] == "success"
    assert up_data["filename"] == "test_trace.csv"
    assert up_data["size_bytes"] == len(file_content)

