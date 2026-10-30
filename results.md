# Forecasting Model Evaluation & Cloud Cost Link Report

Autonomous Cloud Cost & Performance Optimizer  
Model Version: `v2.0.0` | Primary Model: `LightGBM Regressor`

---

## 1. Dataset & Preprocessing

### 1.1 Source & Characteristics
- **Dataset Trace:** Azure Public Dataset V1 (`schema.csv`, `trace_data_vmtable_vmtable.csv.gz`, `trace_data_vm_cpu_readings_vm_cpu_readings-file-1-of-125.csv.gz`).
- **Trace Period:** 16.00 consecutive days ($1,382,100$ seconds).
- **Sampling Interval:** Measured at exactly $300.0$ seconds ($5.0$ minutes).
- **Raw Streamed Records:** $1,371,194$ five-minute readings processed in memory-safe chunks of $100,000$ rows.
- **VMs Filtered:** $300$ virtual machines with at least $14$ days of continuous readings.
- **Hourly Resampling:** Aggregated to mean CPU (`hourly_avg_cpu`) and maximum CPU (`hourly_max_cpu`) per hour ($384$ continuous hours per VM, total $115,200$ rows).
- **Processed Artifact:** Saved to `data/processed/sample.parquet` ($0.68$ MB).

### 1.2 Missing Data Handling
- Missing readings in the raw trace ($0.8\%$ sensor dropout) were audited during chunk streaming.
- Hourly aggregation handles sparse intervals, followed by forward-fill and backward-fill per VM series to guarantee an unbroken, contiguous timeline.
- Final missing values in processed dataset: **$0$** ($0.0\%$).

---

## 2. Feature Engineering & Split

### 2.1 Feature Definitions
- **Autoregressive Lags:** `lag_1h`, `lag_2h`, `lag_24h`, `lag_168h` (1-week lag).
- **Rolling Statistics (Shift 1 to eliminate future leakage):**
  - $6$-hour window: `rolling_mean_6h`, `rolling_max_6h`, `rolling_std_6h`
  - $24$-hour window: `rolling_mean_24h`, `rolling_max_24h`, `rolling_std_24h`
  - $168$-hour window: `rolling_mean_168h`, `rolling_max_168h`, `rolling_std_168h`
- **Weekly Trend:** `trend_7d` ($24$h rolling mean minus $168$h rolling mean).
- **Diurnal & Calendar:** `hour_sin`, `hour_cos`, `day_of_week`, `is_weekend`.
- **VM Metadata (from `vmtable`):** `core_count`, `memory_gb`, and one-hot encoded `category` (`Interactive`, `Delay-insensitive`, `Unknown`).

### 2.2 Chronological Split (No Shuffling, No Data Leakage)
- Warm-up period ($168$ hours) reserved for lag and rolling calculations.
- Active sequence ($216$ hours per VM, hours $168$ to $383$, $64,800$ total samples):
  - **Training Set (70%):** Hours $168$ to $318$ ($151$ hours, $45,300$ samples)
  - **Validation Set (15%):** Hours $319$ to $350$ ($32$ hours, $9,600$ samples)
  - **Test Set (15%):** Hours $351$ to $383$ ($33$ hours, $9,900$ samples)

---

## 3. Measured Forecasting Performance

All numbers below were measured directly on the unseen $9,900$-sample chronological test set.

### 3.1 Overall Model Comparison

$$\text{Accuracy} = 100 - \text{sMAPE}$$

| Model | MAE (%) | RMSE (%) | sMAPE (%) | Accuracy (%) | Beats Naive Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive ($t-1$)** | 1.992 | 3.014 | 11.26% | 88.74% | *Baseline* |
| **Seasonal Naive ($t-168$)** | 1.211 | 1.904 | 9.14% | 90.86% | **YES** |
| **Moving Average (24h)** | 4.620 | 8.527 | 16.03% | 83.97% | **NO** |
| **Random Forest** | 0.993 | 1.466 | 7.25% | 92.75% | **YES** |
| **LightGBM (Production)** | **0.967** | **1.449** | **7.04%** | **92.96%** | **YES** |

### 3.2 Train vs. Validation vs. Test Set Comparison (Overfitting Audit)

To verify model fit and ensure zero data leakage or overfitting, both LightGBM and Random Forest were evaluated across the chronological splits:

| Split | Sample Count | Model | MAE (%) | RMSE (%) | sMAPE (%) | Accuracy (%) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Training Set** | 45,300 | **LightGBM** | 0.920% | 1.387% | 6.83% | **93.17%** |
| | | Random Forest | 0.766% | 1.176% | 5.79% | 94.21% |
| **Validation Set** | 9,600 | **LightGBM** | 0.974% | 1.456% | 7.01% | **92.99%** |
| | | Random Forest | 1.004% | 1.488% | 7.20% | 92.80% |
| **Test Set (Unseen)** | 9,900 | **LightGBM** | 0.967% | 1.449% | 7.04% | **92.96%** |
| | | Random Forest | 0.993% | 1.466% | 7.25% | 92.75% |

> **Audit Verdict:** LightGBM demonstrates remarkable generalization stability with test accuracy ($92.96\%$) closely matching training accuracy ($93.17\%$). There is zero overfitting, and the residual distribution confirms robust calibration across all operational ranges.

> **Key Finding:** LightGBM achieves the lowest MAE ($0.967\%$) and lowest RMSE ($1.449\%$) with $92.96\%$ measured accuracy, outperforming both naive and statistical baselines across all metrics.

---

### 3.2 Breakdown by Workload Type ($75$ VMs / $2,475$ test readings each)

#### 1. Steady Workloads ($75$ VMs)
*Persistent load, modest variance, average CPU $\ge 10\%$*
- **Naive ($t-1$):** MAE $1.036\%$ | RMSE $1.311\%$ | sMAPE $2.47\%$ | Accuracy $97.53\%$
- **Seasonal Naive ($t-168$):** MAE $0.971\%$ | RMSE $1.222\%$ | sMAPE $2.31\%$ | Accuracy $97.69\%$
- **Moving Average (24h):** MAE $1.608\%$ | RMSE $1.891\%$ | sMAPE $3.82\%$ | Accuracy $96.18\%$
- **Random Forest:** MAE $0.784\%$ | RMSE $0.992\%$ | sMAPE $1.87\%$ | Accuracy $98.13\%$
- **LightGBM:** **MAE $0.793\%$** | **RMSE $0.999\%$** | **sMAPE $1.89\%$** | **Accuracy $98.11\%$**

#### 2. Daily Peak Workloads ($75$ VMs)
*Strong diurnal cycle with peak business-hour spikes*
- **Naive ($t-1$):** MAE $4.321\%$ | RMSE $4.898\%$ | sMAPE $11.58\%$ | Accuracy $88.42\%$
- **Seasonal Naive ($t-168$):** MAE $1.318\%$ | RMSE $1.647\%$ | sMAPE $3.81\%$ | Accuracy $96.19\%$
- **Moving Average (24h):** MAE $14.961\%$ | RMSE $16.789\%$ | sMAPE $37.43\%$ | Accuracy $62.57\%$
- **Random Forest:** MAE $1.118\%$ | RMSE $1.415\%$ | sMAPE $3.18\%$ | Accuracy $96.82\%$
- **LightGBM:** **MAE $1.129\%$** | **RMSE $1.428\%$** | **sMAPE $3.22\%$** | **Accuracy $96.78\%$**

#### 3. Bursty Workloads ($75$ VMs)
*Low baseline with intermittent high utilization spikes*
- **Naive ($t-1$):** MAE $2.134\%$ | RMSE $3.206\%$ | sMAPE $13.70\%$ | Accuracy $86.30\%$
- **Seasonal Naive ($t-168$):** MAE $2.085\%$ | RMSE $3.153\%$ | sMAPE $13.36\%$ | Accuracy $86.64\%$
- **Moving Average (24h):** MAE $1.568\%$ | RMSE $2.281\%$ | sMAPE $10.34\%$ | Accuracy $89.66\%$
- **Random Forest:** MAE $1.727\%$ | RMSE $2.331\%$ | sMAPE $11.48\%$ | Accuracy $88.52\%$
- **LightGBM:** **MAE $1.606\%$** | **RMSE $2.275\%$** | **sMAPE $10.65\%$** | **Accuracy $89.35\%$**

#### 4. Mostly Idle Workloads ($75$ VMs)
*Mean CPU $< 10\%$*
- **Naive ($t-1$):** MAE $0.476\%$ | RMSE $0.595\%$ | sMAPE $17.30\%$ | Accuracy $82.70\%$
- **Seasonal Naive ($t-168$):** MAE $0.469\%$ | RMSE $0.590\%$ | sMAPE $17.06\%$ | Accuracy $82.94\%$
- **Moving Average (24h):** MAE $0.344\%$ | RMSE $0.430\%$ | sMAPE $12.53\%$ | Accuracy $87.47\%$
- **Random Forest:** MAE $0.343\%$ | RMSE $0.427\%$ | sMAPE $12.45\%$ | Accuracy $87.55\%$
- **LightGBM:** **MAE $0.340\%$** | **RMSE $0.425\%$** | **sMAPE $12.39\%$** | **Accuracy $87.61\%$**

---

## 4. Uncertainty Quantification

Residual distribution measured on validation data:
- **10th Percentile Residual ($q_{10}$):** $-1.46\%$
- **90th Percentile Residual ($q_{90}$):** $+1.58\%$
- **Standard Deviation ($\sigma$):** $1.46\%$
- **Empirical Test Coverage:** $79.8\%$ of actual CPU points fall within the $[ \hat{y} + q_{10}, \hat{y} + q_{90} ]$ prediction band.

---

## 5. Cost Link & Overload Trade-Off Analysis

Evaluated over the $33$-hour test window across all $300$ virtual machines using the assumed pricing of $\$0.0208$ per vCPU-hour from `rules.yaml` with a safe capacity threshold of $65\%$:

| Strategy | Total Allocated Cores | Total Spend ($) | Spend Reduction (%) | Overload Hours (>90% CPU) | Overload Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. No Optimization (As-Is)** | 1,454 | $998.03 | 0.00% | 0 | 0.00% |
| **2. Baseline Sizing (Moving Avg)** | 1,063 | $729.64 | 26.89% | 0 | 0.00% |
| **3. Model-Based Sizing (LightGBM)** | **1,062** | **$728.96** | **26.96%** | **1** | **0.01%** |

- **Cost Reduction:** Model-based sizing reduces cloud compute spend by **$26.96\%$** ($-\$269.07$ in 33 hours across 300 VMs).
- **SLA Reliability:** Overload hours remained at **$1$ hour out of $9,900$ VM-hours** ($0.01\%$), well within the $99.9\%$ enterprise SLA target.

---

## 6. Generated Visualizations & Artifacts

1. **`outputs/plots/forecast_vs_actual.png`**: Visual comparison of predicted hourly demand vs. real telemetry with the $80\%$ uncertainty band.
2. **`outputs/plots/error_by_model.png`**: Side-by-side error metrics (MAE, RMSE, sMAPE) comparing all five models.
3. **`outputs/plots/cost_vs_overload_tradeoff.png`**: Total spend vs. overload hours trade-off comparing unoptimized, baseline, and model-based policies.
4. **`outputs/forecast_evaluation_metrics.csv`**: Full numeric table of overall error metrics.
5. **`outputs/forecast_metrics_by_workload.csv`**: Segmented metrics by workload classification.
6. **`outputs/cost_vs_overload_tradeoff.csv`**: Quantitative cost and SLA impact breakdown.
7. **`models/forecaster_lgb.joblib` & `forecaster_metadata.json`**: Production forecaster model and inference configuration.

