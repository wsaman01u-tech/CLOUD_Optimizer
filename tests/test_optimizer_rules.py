"""Tests for Autonomous Cloud Cost & Performance Optimizer Rules & Guardrails.
Covers:
1. Missing data handling
2. Insufficient history (no auto action)
3. Policy-blocked change (protect production resources)
4. Unapproved change not applied in Suggest only mode
5. Duplicate run does not repeat an action (idempotency)
6. Audit entry created for every action
7. One-click Undo works and restores previous server state
"""
import os
import pytest
import pandas as pd
import numpy as np

from agents.cloud_optimizer import (
    load_rules,
    generate_fleet_telemetry,
    generate_recommendations,
    FleetExecutionManager,
    AuditLogger,
    evaluate_what_if,
)


@pytest.fixture
def base_rules():
    """Load default rules dictionary for testing."""
    return load_rules("rules.yaml")


@pytest.fixture
def mock_fleet():
    """Create a controlled mini-fleet dataframe for test coverage."""
    return pd.DataFrame([
        {
            "resource_id": "ec2-prod-web-01",
            "resource_type": "compute",
            "environment": "production",
            "current_size": "t3.2xlarge",
            "vcpu": 8,
            "monthly_cost": 239.62,
            "avg_cpu": 3.5,
            "peak_cpu": 6.8,
            "pct_low": 98.0,
            "status": "idle",
            "history_points": 336,
            "is_protected": True,
            "has_missing_data": False,
        },
        {
            "resource_id": "ec2-dev-sandbox-01",
            "resource_type": "compute",
            "environment": "development",
            "current_size": "t3.xlarge",
            "vcpu": 4,
            "monthly_cost": 119.81,
            "avg_cpu": 2.8,
            "peak_cpu": 5.4,
            "pct_low": 99.0,
            "status": "idle",
            "history_points": 336,
            "is_protected": False,
            "has_missing_data": False,
        },
        {
            "resource_id": "ec2-dev-new-box",
            "resource_type": "compute",
            "environment": "development",
            "current_size": "t3.large",
            "vcpu": 2,
            "monthly_cost": 59.90,
            "avg_cpu": 4.1,
            "peak_cpu": 7.2,
            "pct_low": 95.0,
            "status": "idle",
            "history_points": 8,  # Insufficient history points (< 24)
            "is_protected": False,
            "has_missing_data": False,
        },
        {
            "resource_id": "ec2-corrupted-data",
            "resource_type": "compute",
            "environment": "staging",
            "current_size": "t3.medium",
            "vcpu": 2,
            "monthly_cost": 29.95,
            "avg_cpu": np.nan,  # Missing/corrupted data
            "peak_cpu": np.nan,
            "pct_low": np.nan,
            "status": "unknown",
            "history_points": 0,
            "is_protected": False,
            "has_missing_data": True,
        },
    ])


def test_missing_data_handled_gracefully(base_rules, mock_fleet):
    """Verify that records with missing/NaN metrics are safely skipped without crash."""
    recs = generate_recommendations(mock_fleet, base_rules)
    rec_ids = [r["resource_id"] for r in recs]
    assert "ec2-corrupted-data" not in rec_ids


def test_insufficient_history_prevents_action(base_rules, mock_fleet):
    """Verify that resources with insufficient observations (< min_data_history_points) produce no recommendations."""
    recs = generate_recommendations(mock_fleet, base_rules)
    rec_ids = [r["resource_id"] for r in recs]
    assert "ec2-dev-new-box" not in rec_ids


def test_policy_blocks_production_termination_without_approval(base_rules, mock_fleet):
    """Verify that protect_production guardrail blocks production terminations from auto-applying."""
    base_rules["policy"]["protect_production"] = True
    base_rules["mode"] = "Auto-fix safe items"

    recs = generate_recommendations(mock_fleet, base_rules)
    prod_rec = next((r for r in recs if r["resource_id"] == "ec2-prod-web-01"), None)

    assert prod_rec is not None
    assert prod_rec["blocked_by_policy"] is True
    assert prod_rec["risk_level"] == "High"

    logger = AuditLogger("outputs/test_audit.csv")
    mgr = FleetExecutionManager(mock_fleet, recs, base_rules, logger)
    success, msg = mgr.apply_recommendation(prod_rec["id"], actor="system")

    assert success is False
    assert "blocked" in msg.lower()


def test_unapproved_change_not_applied_in_suggest_mode(base_rules, mock_fleet):
    """Verify that in 'Suggest only' mode, pending recommendations cannot be applied until approved."""
    base_rules["mode"] = "Suggest only"
    recs = generate_recommendations(mock_fleet, base_rules)
    dev_rec = next((r for r in recs if r["resource_id"] == "ec2-dev-sandbox-01"), None)

    assert dev_rec is not None
    assert dev_rec["status"] == "Pending"

    logger = AuditLogger("outputs/test_audit.csv")
    mgr = FleetExecutionManager(mock_fleet, recs, base_rules, logger)

    # Attempt to apply unapproved change
    success, msg = mgr.apply_recommendation(dev_rec["id"], actor="system")
    assert success is False
    assert "requires manual approval" in msg.lower() or "blocked" in msg.lower()

    # Now explicitly approve it
    approve_ok, _ = mgr.approve_recommendation(dev_rec["id"], actor="cloud_admin")
    assert approve_ok is True
    assert mgr.recommendations[dev_rec["id"]]["status"] == "Approved"

    # Now apply should succeed
    success_after, _ = mgr.apply_recommendation(dev_rec["id"], actor="cloud_admin")
    assert success_after is True
    assert mgr.recommendations[dev_rec["id"]]["status"] == "Applied"


def test_idempotent_application_no_duplicate_runs(base_rules, mock_fleet):
    """Verify that re-applying an already applied recommendation returns idempotently without repeating cost deduction."""
    recs = generate_recommendations(mock_fleet, base_rules)
    dev_rec = next((r for r in recs if r["resource_id"] == "ec2-dev-sandbox-01"), None)

    logger = AuditLogger("outputs/test_audit.csv")
    mgr = FleetExecutionManager(mock_fleet, recs, base_rules, logger)

    mgr.approve_recommendation(dev_rec["id"], actor="admin")
    success1, _ = mgr.apply_recommendation(dev_rec["id"], actor="admin")
    assert success1 is True

    cost_after_first = mgr.fleet_df.loc[
        mgr.fleet_df["resource_id"] == "ec2-dev-sandbox-01", "monthly_cost"
    ].values[0]
    assert cost_after_first == 0.0  # terminated

    # Run second time
    success2, msg2 = mgr.apply_recommendation(dev_rec["id"], actor="admin")
    assert success2 is True
    assert "idempotent" in msg2.lower()

    cost_after_second = mgr.fleet_df.loc[
        mgr.fleet_df["resource_id"] == "ec2-dev-sandbox-01", "monthly_cost"
    ].values[0]
    assert cost_after_first == cost_after_second


def test_audit_entry_created_for_actions():
    """Verify that every lifecycle event (approval, apply, rollback) creates an audit record."""
    test_path = "outputs/test_audit_entry.csv"
    if os.path.exists(test_path):
        os.remove(test_path)

    logger = AuditLogger(test_path)
    logger.log_event(
        event_type="RECOMMENDED",
        resource_id="ec2-test-01",
        action="downsize",
        current_size="t3.xlarge",
        target_size="t3.large",
        monthly_saving=59.91,
        actor="CloudOptimizer-Engine",
        reason="Peak CPU < 30%",
    )
    logger.log_event(
        event_type="APPROVED",
        resource_id="ec2-test-01",
        action="downsize",
        current_size="t3.xlarge",
        target_size="t3.large",
        monthly_saving=59.91,
        actor="Alice DevOps",
        reason="Approved after review",
    )
    logger.log_event(
        event_type="APPLIED",
        resource_id="ec2-test-01",
        action="downsize",
        current_size="t3.xlarge",
        target_size="t3.large",
        monthly_saving=59.91,
        actor="CloudOptimizer-Deployer",
        reason="Terraform applied",
        verification_status="VERIFIED_SAFE",
    )

    df_logs = logger.get_logs()
    assert len(df_logs) == 3
    assert "RECOMMENDED" in df_logs["event_type"].values
    assert "APPROVED" in df_logs["event_type"].values
    assert "APPLIED" in df_logs["event_type"].values
    assert "Alice DevOps" in df_logs["actor"].values


def test_undo_works_and_reverts_state(base_rules, mock_fleet):
    """Verify that one-click undo reverts an applied change back to its original server size and cost."""
    recs = generate_recommendations(mock_fleet, base_rules)
    dev_rec = next((r for r in recs if r["resource_id"] == "ec2-dev-sandbox-01"), None)

    logger = AuditLogger("outputs/test_undo_audit.csv")
    mgr = FleetExecutionManager(mock_fleet, recs, base_rules, logger)

    # 1. Approve and Apply
    mgr.approve_recommendation(dev_rec["id"], actor="admin")
    apply_ok, _ = mgr.apply_recommendation(dev_rec["id"], actor="admin")
    assert apply_ok is True
    assert mgr.recommendations[dev_rec["id"]]["status"] == "Applied"

    applied_cost = mgr.fleet_df.loc[
        mgr.fleet_df["resource_id"] == "ec2-dev-sandbox-01", "monthly_cost"
    ].values[0]
    assert applied_cost == 0.0

    # 2. Undo / Rollback
    undo_ok, undo_msg = mgr.rollback_recommendation(dev_rec["id"], actor="admin", reason="Mistake, revert")
    assert undo_ok is True
    assert "restored" in undo_msg.lower()

    # Verify original state is restored
    reverted_cost = mgr.fleet_df.loc[
        mgr.fleet_df["resource_id"] == "ec2-dev-sandbox-01", "monthly_cost"
    ].values[0]
    assert reverted_cost == pytest.approx(119.81, 0.01)

    # Verify audit log captures rollback
    logs = logger.get_logs()
    assert "ROLLED_BACK" in logs["event_type"].values

