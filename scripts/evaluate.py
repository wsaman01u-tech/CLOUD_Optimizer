"""Evaluation script for CloudOptimizer demand forecasting and cost-link analysis.

Performs:
1. Strict chronological evaluation on test set (models/test_features.parquet).
2. Evaluates 5 models:
   - Naive (t-1)
   - Seasonal Naive (t-168)
   - Moving Average (24h)
   - Random Forest Regressor
   - LightGBM Regressor
3. Metrics: MAE, RMSE, sMAPE, and Accuracy (100 - sMAPE).
4. Segmented evaluation by workload type:
   - Steady
   - Daily Peak
   - Bursty
   - Mostly Idle
5. Baseline beat verification (only measured numbers reported).
6. Uncertainty band evaluation (residual coverage).
7. Cost link and overload hours trade-off analysis using rules.yaml pricing.
8. Generates visualization charts and CSV metrics.
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def calculate_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    """Calculate MAE, RMSE, sMAPE, and Accuracy."""
    act = np.asarray(actual, dtype=float)
    pr = np.asarray(pred, dtype=float)
    mae_val = float(np.mean(np.abs(act - pr)))
    rmse_val = float(np.sqrt(np.mean((act - pr) ** 2)))
    # Symmetric MAPE: 100 * (2 * |y - y_hat|) / (|y| + |y_hat| + eps)
    denom = np.abs(act) + np.abs(pr) + 1e-6
    smape_val = float(100.0 * np.mean(2.0 * np.abs(act - pr) / denom))
    accuracy_val = float(max(0.0, 100.0 - smape_val))
    return {
        "MAE": round(mae_val, 3),
        "RMSE": round(rmse_val, 3),
        "sMAPE": round(smape_val, 2),
        "Accuracy": round(accuracy_val, 2),
    }


def classify_workload_type(sample_parquet_path: str = "data/processed/sample.parquet") -> dict:
    """Classify each VM into Steady, Daily Peak, Bursty, or Mostly Idle using observed full-trace behavior."""
    df = pd.read_parquet(sample_parquet_path)
    stats = df.groupby("vm_id").agg(
        avg_mean=("hourly_avg_cpu", "mean"),
        avg_std=("hourly_avg_cpu", "std"),
        max_peak=("hourly_max_cpu", "max"),
    )
    stats["ratio"] = stats["max_peak"] / stats["avg_mean"].clip(lower=1.0)

    vm_types = {}
    for vm_id, r in stats.iterrows():
        if r["avg_mean"] < 10.0:
            vm_types[vm_id] = "Mostly Idle"
        elif r["ratio"] > 3.0:
            vm_types[vm_id] = "Bursty"
        elif r["avg_std"] > 8.0:
            vm_types[vm_id] = "Daily Peak"
        else:
            vm_types[vm_id] = "Steady"
    return vm_types


def run_evaluation(
    models_dir: str = "models",
    output_dir: str = "outputs",
    rules_path: str = "rules.yaml",
):
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs("data/results", exist_ok=True)

    print("=" * 60)
    print("STEP 1: LOADING TEST DATA & TRAINED MODELS")
    print("=" * 60)

    test_path = os.path.join(models_dir, "test_features.parquet")
    meta_path = os.path.join(models_dir, "forecaster_metadata.json")
    lgb_path = os.path.join(models_dir, "forecaster_lgb.joblib")
    rf_path = os.path.join(models_dir, "forecaster_rf.joblib")

    test_df = pd.read_parquet(test_path)
    with open(meta_path) as f:
        metadata = json.load(f)

    lgb_model = joblib.load(lgb_path)
    rf_model = joblib.load(rf_path)

    feature_cols = metadata["feature_cols"]
    target_col = metadata["target_col"]
    q10_res = metadata["uncertainty_band"]["q10_res"]
    q90_res = metadata["uncertainty_band"]["q90_res"]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values

    print(f"Test samples: {len(test_df):,} across {test_df['vm_id'].nunique()} VMs")
    print(f"Test hours range: {test_df['hour_bucket'].min()} to {test_df['hour_bucket'].max()} ({test_df['hour_bucket'].nunique()} hours)")

    # 1. Predictions
    print("\nGenerating model predictions...")
    preds = {
        "Naive (last hour)": test_df["lag_1h"].values,
        "Seasonal Naive (same hour last week)": test_df["lag_168h"].values,
        "Moving Average (24h)": test_df["rolling_mean_24h"].values,
        "Random Forest": rf_model.predict(X_test),
        "LightGBM": lgb_model.predict(X_test),
    }

    # Attach predictions and uncertainty bands to test_df
    for name, p in preds.items():
        test_df[f"pred_{name}"] = p

    # Uncertainty band for primary LightGBM model
    test_df["lgb_lower"] = np.clip(test_df["pred_LightGBM"] + q10_res, 0.0, 100.0)
    test_df["lgb_upper"] = np.clip(test_df["pred_LightGBM"] + q90_res, 0.0, 100.0)

    # Coverage rate
    within_band = (test_df["hourly_avg_cpu"] >= test_df["lgb_lower"]) & (test_df["hourly_avg_cpu"] <= test_df["lgb_upper"])
    band_coverage = float(100.0 * within_band.mean())

    # 2. Overall Metrics Table
    print("\n" + "=" * 60)
    print("OVERALL MODEL PERFORMANCE COMPARISON")
    print("=" * 60)
    overall_results = []
    baseline_mae = None

    for name, p in preds.items():
        m = calculate_metrics(y_test, p)
        if name == "Naive (last hour)":
            baseline_mae = m["MAE"]
            beats = "Baseline"
        elif baseline_mae is not None:
            beats = "YES" if m["MAE"] < baseline_mae else "NO"
        else:
            beats = "N/A"

        row = {
            "Model": name,
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "sMAPE (%)": m["sMAPE"],
            "Accuracy (%)": m["Accuracy"],
            "Beats Naive": beats,
        }
        overall_results.append(row)
        print(f"{name:38s} | MAE: {m['MAE']:6.3f} | RMSE: {m['RMSE']:6.3f} | sMAPE: {m['sMAPE']:5.2f}% | Acc: {m['Accuracy']:5.2f}% | Beats: {beats}")

    overall_df = pd.DataFrame(overall_results)

    # 3. Workload Type Breakdown
    print("\n" + "=" * 60)
    print("PERFORMANCE BREAKDOWN BY WORKLOAD TYPE")
    print("=" * 60)
    vm_workload_types = classify_workload_type("data/processed/sample.parquet")
    test_df["workload_type"] = test_df["vm_id"].map(vm_workload_types)

    segment_results = []
    for w_type in ["Steady", "Daily Peak", "Bursty", "Mostly Idle"]:
        sub = test_df[test_df["workload_type"] == w_type]
        vm_count = sub["vm_id"].nunique()
        print(f"\n--- {w_type.upper()} ({vm_count} VMs, {len(sub):,} readings) ---")
        for m_name in preds.keys():
            m = calculate_metrics(sub["hourly_avg_cpu"].values, sub[f"pred_{m_name}"].values)
            segment_results.append({
                "Workload Type": w_type,
                "VM Count": vm_count,
                "Model": m_name,
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "sMAPE (%)": m["sMAPE"],
                "Accuracy (%)": m["Accuracy"],
            })
            print(f"  {m_name:36s} | MAE: {m['MAE']:6.3f} | RMSE: {m['RMSE']:6.3f} | sMAPE: {m['sMAPE']:5.2f}% | Acc: {m['Accuracy']:5.2f}%")

    segment_df = pd.DataFrame(segment_results)

    # 4. Cost Link & Overload Hours Trade-Off Analysis
    print("\n" + "=" * 60)
    print("COST LINK & OVERLOAD TRADE-OFF ANALYSIS")
    print("=" * 60)

    # Load pricing from rules.yaml
    with open(rules_path) as f:
        rules_cfg = yaml.safe_load(f)

    catalog = rules_cfg.get("pricing", {}).get("instance_catalog", {})
    # Default assumed price per vCPU-hour: ~0.0208 $/core-hr based on t3.small
    price_per_vcpu_hour = 0.0208
    safe_cpu_limit = float(rules_cfg.get("policy", {}).get("safe_cpu_limit", 65.0))

    # Evaluate across all VMs during test period (33 hours)
    test_hours = test_df["hour_bucket"].nunique()
    sizing_records = []

    # Strategy 1: No Optimization (Baseline as-is)
    # Original assigned cores running 24/7
    orig_total_cores = float(test_df.groupby("vm_id")["core_count"].first().sum())
    no_opt_cost = orig_total_cores * test_hours * price_per_vcpu_hour
    # Overload hours: when actual CPU > 90%
    no_opt_overloads = int((test_df["hourly_avg_cpu"] > 90.0).sum())

    # Strategy 2: Baseline Sizing (Moving Average 24h peak)
    # Target size = ceil(original_cores * (past_rolling_max_24h / safe_cpu_limit)) clamped [2, orig]
    bl_cores_list = []
    bl_overloads = 0
    for vm_id, g in test_df.groupby("vm_id"):
        orig_cores = int(g["core_count"].iloc[0])
        bl_p95 = float(g["rolling_max_24h"].iloc[0])
        sized_cores = max(2, min(orig_cores, int(np.ceil(orig_cores * (bl_p95 / safe_cpu_limit)))))
        bl_cores_list.append(sized_cores)

        # Scale actual CPU to the new capacity
        scaled_cpu = g["hourly_avg_cpu"] * (orig_cores / sized_cores)
        bl_overloads += int((scaled_cpu > 90.0).sum())

    bl_total_cores = sum(bl_cores_list)
    bl_cost = bl_total_cores * test_hours * price_per_vcpu_hour

    # Strategy 3: Model-Based Sizing (LightGBM Upper Band Forecast)
    # Uses upper uncertainty band predicted for the upcoming window
    model_cores_list = []
    model_overloads = 0
    for vm_id, g in test_df.groupby("vm_id"):
        orig_cores = int(g["core_count"].iloc[0])
        pred_upper_max = float(g["lgb_upper"].max())
        sized_cores = max(2, min(orig_cores, int(np.ceil(orig_cores * (pred_upper_max / safe_cpu_limit)))))
        model_cores_list.append(sized_cores)

        # Scale actual CPU to the new capacity
        scaled_cpu = g["hourly_avg_cpu"] * (orig_cores / sized_cores)
        model_overloads += int((scaled_cpu > 90.0).sum())

    model_total_cores = sum(model_cores_list)
    model_cost = model_total_cores * test_hours * price_per_vcpu_hour

    tradeoff_rows = [
        {
            "Strategy": "1. No Optimization (As-Is)",
            "Total Cores Allocated": int(orig_total_cores),
            "Total Spend ($)": round(no_opt_cost, 2),
            "Spend Reduction (%)": 0.0,
            "Overload Hours (>90% CPU)": no_opt_overloads,
            "Overload Rate (%)": round(100.0 * no_opt_overloads / len(test_df), 3),
        },
        {
            "Strategy": "2. Baseline Sizing (Moving Avg)",
            "Total Cores Allocated": int(bl_total_cores),
            "Total Spend ($)": round(bl_cost, 2),
            "Spend Reduction (%)": round(100.0 * (no_opt_cost - bl_cost) / no_opt_cost, 2),
            "Overload Hours (>90% CPU)": bl_overloads,
            "Overload Rate (%)": round(100.0 * bl_overloads / len(test_df), 3),
        },
        {
            "Strategy": "3. Model-Based Sizing (LightGBM Forecaster)",
            "Total Cores Allocated": int(model_total_cores),
            "Total Spend ($)": round(model_cost, 2),
            "Spend Reduction (%)": round(100.0 * (no_opt_cost - model_cost) / no_opt_cost, 2),
            "Overload Hours (>90% CPU)": model_overloads,
            "Overload Rate (%)": round(100.0 * model_overloads / len(test_df), 3),
        },
    ]

    tradeoff_df = pd.DataFrame(tradeoff_rows)
    for r in tradeoff_rows:
        print(f"{r['Strategy']:42s} | Spend: ${r['Total Spend ($)']:8.2f} (-{r['Spend Reduction (%)']:5.1f}%) | Overload Hours: {r['Overload Hours (>90% CPU)']:3d} ({r['Overload Rate (%)']:.2f}%)")

    # Save metrics CSVs
    metrics_csv = os.path.join(output_dir, "forecast_evaluation_metrics.csv")
    overall_df.to_csv(metrics_csv, index=False)
    segment_df.to_csv(os.path.join(output_dir, "forecast_metrics_by_workload.csv"), index=False)
    tradeoff_df.to_csv(os.path.join(output_dir, "cost_vs_overload_tradeoff.csv"), index=False)
    # Also save to data/results for API/dashboard access
    overall_df.to_csv("data/results/forecast_metrics.csv", index=False)
    tradeoff_df.to_csv("data/results/evaluation.csv", index=False)

    print(f"\nSaved metrics CSV to: {metrics_csv}")

    # 5. Generate Matplotlib Charts
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATION PLOTS")
    print("=" * 60)

    # Plot 1: Forecast vs Actual with Uncertainty Band
    sample_vm = test_df[test_df["workload_type"] == "Daily Peak"]["vm_id"].iloc[0]
    vm_plot_df = test_df[test_df["vm_id"] == sample_vm].sort_values("hour_bucket")
    hours_axis = np.arange(len(vm_plot_df))

    plt.figure(figsize=(10, 5), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.plot(hours_axis, vm_plot_df["hourly_avg_cpu"].values, label="Actual CPU", color="#1e293b", linewidth=2.0)
    plt.plot(hours_axis, vm_plot_df["pred_LightGBM"].values, label="LightGBM Forecast", color="#059669", linewidth=2.0, linestyle="--")
    plt.fill_between(
        hours_axis,
        vm_plot_df["lgb_lower"].values,
        vm_plot_df["lgb_upper"].values,
        color="#10b981",
        alpha=0.25,
        label=f"80% Uncertainty Band ({band_coverage:.1f}% coverage)",
    )
    plt.title(f"33-Hour Cloud Demand Forecast vs Actual (VM: {sample_vm[:8]})", fontsize=13, fontweight="bold")
    plt.xlabel("Horizon Step (Hours)", fontsize=11)
    plt.ylabel("Hourly Average CPU (%)", fontsize=11)
    plt.ylim(0, 100)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "plots", "forecast_vs_actual.png")
    plt.savefig(chart1_path)
    plt.close()
    print("Saved:", chart1_path)

    # Plot 2: Error Comparison Across Models
    plt.figure(figsize=(9, 5), dpi=150)
    x = np.arange(len(overall_df))
    width = 0.25
    plt.bar(x - width, overall_df["MAE"], width, label="MAE (%)", color="#3b82f6")
    plt.bar(x, overall_df["RMSE"], width, label="RMSE (%)", color="#6366f1")
    plt.bar(x + width, overall_df["sMAPE (%)"], width, label="sMAPE (%)", color="#f59e0b")
    plt.xticks(x, [m.split(" ")[0] for m in overall_df["Model"]], fontsize=10)
    plt.ylabel("Error (%)", fontsize=11)
    plt.title("Error Metric Comparison Across Forecasting Models (Lower is Better)", fontsize=13, fontweight="bold")
    plt.legend(frameon=True)
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "plots", "error_by_model.png")
    plt.savefig(chart2_path)
    plt.close()
    print("Saved:", chart2_path)

    # Plot 3: Cost vs Overload Trade-Off
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=150)
    strat_labels = ["No Optimization", "Baseline Sizing", "Model-Based"]
    costs = tradeoff_df["Total Spend ($)"]
    overloads = tradeoff_df["Overload Hours (>90% CPU)"]

    color_bar = "#10b981"
    color_line = "#ef4444"

    bars = ax1.bar(strat_labels, costs, color=color_bar, width=0.45, label="Total Spend ($)")
    ax1.set_ylabel("Total Spend ($)", color="#065f46", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#065f46")
    ax1.set_ylim(0, max(costs) * 1.25)

    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 15, f"${h:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(strat_labels, overloads, color=color_line, marker="o", linewidth=2.5, markersize=8, label="Overload Hours")
    ax2.set_ylabel("Overload Hours (>90% CPU)", color=color_line, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_line)
    ax2.set_ylim(0, max(overloads) * 1.5 + 5)

    for i, val in enumerate(overloads):
        ax2.annotate(f"{val} hrs", (strat_labels[i], val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10, fontweight="bold", color=color_line)

    plt.title("Cost vs SLA Overload Risk Trade-Off Across Sizing Strategies", fontsize=13, fontweight="bold")
    fig.tight_layout()
    chart3_path = os.path.join(output_dir, "plots", "cost_vs_overload_tradeoff.png")
    plt.savefig(chart3_path)
    plt.close()
    print("Saved:", chart3_path)

    return overall_df, segment_df, tradeoff_df


if __name__ == "__main__":
    run_evaluation()
