"""Tests for Multi-Cloud Data Normalizer and cross-cloud upload ingestion."""
import os
import pytest
from fastapi.testclient import TestClient
from api.main import app
from agents.data_normalizer import CloudDataNormalizer, generate_multi_cloud_sample_fleet


def test_multi_cloud_sample_fleet_generation():
    """Verify generated sample fleet contains all 3 providers and 7 required dimensions."""
    fleet_df, telemetry_df = generate_multi_cloud_sample_fleet()
    assert len(fleet_df) >= 10
    providers = set(fleet_df["cloud_provider"].unique())
    assert "AWS" in providers
    assert "Azure" in providers
    assert "GCP" in providers

    # Check 7 core data dimensions exist
    required_cols = [
        "resource_id", "resource_name", "resource_type", "cloud_provider", "region", "current_size", "status",
        "avg_cpu", "peak_cpu",
        "memory_utilization_pct", "available_memory_gb",
        "total_storage_gb", "storage_utilization_pct",
        "network_in_mb", "network_out_mb",
        "monthly_cost", "currency", "billing_period",
        "uptime_hours", "daily_usage_hours"
    ]
    for col in required_cols:
        assert col in fleet_df.columns, f"Missing required column: {col}"


def test_load_and_normalize_samples():
    """Test loading and normalizing the sample CSV, JSON, and XLSX files."""
    for filename in ["multi_cloud_fleet_sample.csv", "multi_cloud_fleet_sample.json", "multi_cloud_fleet_sample.xlsx"]:
        path = os.path.join("data/samples", filename)
        assert os.path.exists(path), f"Sample file missing: {path}"
        with open(path, "rb") as f:
            content = f.read()
        records, fmt = CloudDataNormalizer.load_file(filename, content)
        assert len(records) > 0
        fleet_df, telemetry_df, summary = CloudDataNormalizer.normalize_dataset(records)
        assert summary["total_resources"] == len(records)
        assert "AWS" in summary["cloud_providers"]
        assert "Azure" in summary["cloud_providers"]
        assert "GCP" in summary["cloud_providers"]


def test_api_upload_data_and_samples():
    """Verify /api/v2/upload-data and /api/v2/samples/{filename} endpoints."""
    client = TestClient(app)

    # 1. Download sample CSV
    resp = client.get("/api/v2/samples/multi_cloud_fleet_sample.csv")
    assert resp.status_code == 200
    csv_bytes = resp.content
    assert len(csv_bytes) > 0

    # 2. Upload CSV to ingestion endpoint
    upload_resp = client.post(
        "/api/v2/upload-data",
        files={"file": ("test_multi_cloud.csv", csv_bytes, "text/csv")}
    )
    assert upload_resp.status_code == 200
    data = upload_resp.json()
    assert data["success"] is True
    assert data["format"] == "CSV"
    assert data["summary"]["total_resources"] >= 10
    assert "AWS" in data["summary"]["cloud_providers"]

    # 3. Check /api/v2/servers reflects multi-cloud fleet
    servers_resp = client.get("/api/v2/servers")
    assert servers_resp.status_code == 200
    servers = servers_resp.json()
    providers = set(s.get("cloud_provider") for s in servers)
    assert "AWS" in providers
    assert "Azure" in providers
    assert "GCP" in providers


def test_reject_non_cloud_data():
    """Verify that non-cloud datasets (student grades, sales orders, HR payroll) are identified and rejected with HTTP 400."""
    client = TestClient(app)

    # 1. Academic / Student Grades CSV
    student_csv = "student_id,student_name,grade,subject\n101,Alice Johnson,A,Mathematics\n102,Bob Smith,B,Physics\n"
    resp = client.post(
        "/api/v2/upload-data",
        files={"file": ("grades.csv", student_csv.encode("utf-8"), "text/csv")},
        data={"user_email": "engineer@cloudopt.ai", "is_demo": "false"}
    )
    assert resp.status_code == 400
    assert "Invalid Cloud Dataset" in resp.json()["detail"]
    assert "Academic / Student" in resp.json()["detail"]

    # 2. E-commerce / Retail Sales CSV
    sales_csv = "order_id,product_name,customer_id,shipping_address\nORD-991,Blue Denim Jeans,CUST-44,123 Main St\n"
    resp = client.post(
        "/api/v2/upload-data",
        files={"file": ("sales_orders.csv", sales_csv.encode("utf-8"), "text/csv")},
        data={"user_email": "engineer@cloudopt.ai", "is_demo": "false"}
    )
    assert resp.status_code == 400
    assert "Invalid Cloud Dataset" in resp.json()["detail"]
    assert "E-Commerce / Retail" in resp.json()["detail"]

    # 3. Arbitrary Unrelated Garbage Columns
    garbage_csv = "foo_col,bar_metric,baz_val\n1,2,3\n4,5,6\n"
    resp = client.post(
        "/api/v2/upload-data",
        files={"file": ("random_data.csv", garbage_csv.encode("utf-8"), "text/csv")},
        data={"user_email": "engineer@cloudopt.ai", "is_demo": "false"}
    )
    assert resp.status_code == 400
    assert "Invalid Cloud Dataset" in resp.json()["detail"]
    assert "foo_col" in resp.json()["detail"]


def test_clean_unstructured_and_noisy_cloud_data():
    """Verify that messy, noisy cloud telemetry (dirty currency, unit suffixes, fraction scales, outliers) is cleaned and sanitized."""
    noisy_records = [
        {
            "resource_id": "  i-0987654321fedcba0   ",
            "instance_type": "  t3.2xlarge  ",
            "cloud_provider": "  amzn aws  ",
            "monthly_cost": " $ 1,450.75 ",
            "cpu_utilization": "0.14",  # Fractional scale (should become 14.0%)
            "peak_cpu": " 125.0 % ",     # Outlier (> 100%, should clamp to 100.0%)
            "total_memory": "16384 MB",  # Unit conversion (should become 16.0 GB)
            "total_storage": " 2 TB ",   # Unit conversion (should become 2048.0 GB)
            "network_in": " 1.5 GB ",    # Unit conversion (should become 1536.0 MB)
            "status": "idle",
        },
        {
            "resource_id": "Standard_D8s_v3-azure-vm",
            "instance_type": "Standard_D8s_v3",
            "monthly_cost": "€ 890.50",
            "cpu_utilization": "-5.0",   # Outlier (< 0%, should clamp to 0.0%)
            "peak_cpu": "22.5",
            "total_memory": " 32 GB ",
            "total_storage": "500 GB",
            "network_in": " 20480 KB ",  # Unit conversion (should become 20.0 MB)
            "status": "over_provisioned",
        }
    ]

    fleet_df, telemetry_df, summary = CloudDataNormalizer.normalize_dataset(noisy_records, filename="noisy_test.json")
    assert len(fleet_df) == 2

    row1 = fleet_df.iloc[0]
    # Check whitespace stripped
    assert row1["resource_id"] == "i-0987654321fedcba0"
    assert row1["current_size"] == "t3.2xlarge"
    assert row1["cloud_provider"] == "AWS"
    # Check currency parsed and cleaned
    assert row1["monthly_cost"] == 1450.75
    # Check fractional CPU scaled
    assert row1["avg_cpu"] == 14.0
    # Check outlier clamped
    assert row1["peak_cpu"] == 100.0
    # Check unit conversion
    assert row1["total_memory_gb"] == 16.0
    assert row1["total_storage_gb"] == 2048.0
    assert row1["network_in_mb"] == 1536.0

    row2 = fleet_df.iloc[1]
    assert row2["cloud_provider"] == "Azure"
    assert row2["monthly_cost"] == 890.50
    assert row2["avg_cpu"] == 0.0  # Clamped from -5.0
    assert row2["network_in_mb"] == 20.0  # Converted from 20480 KB

    # Check cleaning report
    cleaning_report = summary["cleaning_report"]
    assert cleaning_report["currency_cleaned"] >= 2
    assert cleaning_report["units_converted"] >= 3
    assert cleaning_report["outliers_clamped"] >= 2
    assert cleaning_report["fractions_scaled"] >= 1
    assert "Cleaned and sanitized" in summary["cleaning_summary"]


def test_real_user_account_empty_state_and_upload():
    """Verify that an actual registered user starts with 0 data and populates upon upload."""
    client = TestClient(app)
    user_email = "alex.cloud.eng@enterprise.io"

    # 1. Real user with no data uploaded yet -> should return empty state
    summary_resp = client.get(f"/api/v2/summary?user_email={user_email}&is_demo=false")
    assert summary_resp.status_code == 200
    data = summary_resp.json()
    assert data["has_data"] is False
    assert data["is_demo"] is False
    assert data["current_monthly_cost"] == 0.0
    assert data["possible_monthly_saving"] == 0.0
    assert data["items_needing_attention"] == 0
    assert "No cloud infrastructure connected yet" in data["plain_english_sentence"]

    # Servers endpoint returns empty for real user without data
    servers_resp = client.get(f"/api/v2/servers?user_email={user_email}&is_demo=false")
    assert servers_resp.status_code == 200
    assert len(servers_resp.json()) == 0

    # Recommendations endpoint returns empty for real user without data
    recs_resp = client.get(f"/api/v2/recommendations?user_email={user_email}&is_demo=false")
    assert recs_resp.status_code == 200
    assert len(recs_resp.json()) == 0

    # 2. Demo account still returns demo data
    demo_resp = client.get("/api/v2/summary?user_email=admin@cloudopt.ai&is_demo=true")
    assert demo_resp.status_code == 200
    demo_data = demo_resp.json()
    assert demo_data["has_data"] is True
    assert demo_data["is_demo"] is True
    assert demo_data["current_monthly_cost"] > 0
    assert demo_data["items_needing_attention"] > 0

    # 3. User loads sample fleet via /api/v2/load-sample
    load_resp = client.post("/api/v2/load-sample", json={"user_email": user_email})
    assert load_resp.status_code == 200
    assert load_resp.json()["success"] is True

    # User now has data!
    summary_after = client.get(f"/api/v2/summary?user_email={user_email}&is_demo=false").json()
    assert summary_after["has_data"] is True
    assert summary_after["current_monthly_cost"] > 0

    servers_after = client.get(f"/api/v2/servers?user_email={user_email}&is_demo=false").json()
    assert len(servers_after) >= 10

    # 4. User clears data via /api/v2/clear-data
    clear_resp = client.post("/api/v2/clear-data", json={"user_email": user_email})
    assert clear_resp.status_code == 200
    summary_cleared = client.get(f"/api/v2/summary?user_email={user_email}&is_demo=false").json()
    assert summary_cleared["has_data"] is False
    assert summary_cleared["current_monthly_cost"] == 0.0


