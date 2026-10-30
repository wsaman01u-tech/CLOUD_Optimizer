"""Demand Forecasting Agent.
Predicts near-future cloud workload demand and evaluates against baselines.
"""
import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    act = np.asarray(actual, dtype=float)
    pr = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((act - pr) ** 2)))


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    act = np.asarray(actual, dtype=float)
    pr = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(act - pr)))


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    act = np.asarray(actual, dtype=float)
    pr = np.asarray(pred, dtype=float)
    eps = 1e-6
    return float(100.0 * np.mean(np.abs((act - pr) / np.maximum(act, eps))))


def make_lags(series: pd.Series, n_lags: int = 24) -> pd.DataFrame:
    """Generate lagged features for supervised regression."""
    df = pd.DataFrame({"y": series})
    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = series.shift(i)
    # Add hour of day and day of week features
    if isinstance(series.index, pd.DatetimeIndex):
        df["hour"] = series.index.hour
        df["dayofweek"] = series.index.dayofweek
    return df.dropna()


class Forecaster:
    """Production Cloud Demand Forecaster with ML & Fallbacks."""

    def __init__(self, n_lags: int = 24, random_state: int = 42):
        self.n_lags = n_lags
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=80,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_names = None
        self.last_known_history = None

    def fit(self, history: pd.Series) -> "Forecaster":
        """Train the forecaster on historical hourly demand."""
        self.last_known_history = history.copy()
        lag_df = make_lags(history, self.n_lags)
        X = lag_df.drop(columns=["y"])
        y = lag_df["y"]
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict_next(self, history: pd.Series) -> float:
        """Predict workload demand for the next hour."""
        if len(history) < self.n_lags:
            # Fallback to rolling average if insufficient points
            return float(history.mean())

        recent = history.iloc[-self.n_lags:].values[::-1]
        feature_dict = {f"lag_{i+1}": recent[i] for i in range(self.n_lags)}
        if isinstance(history.index, pd.DatetimeIndex):
            next_ts = history.index[-1] + pd.Timedelta(hours=1)
            feature_dict["hour"] = next_ts.hour
            feature_dict["dayofweek"] = next_ts.dayofweek

        X_row = pd.DataFrame([feature_dict])
        if self.feature_names:
            X_row = X_row.reindex(columns=self.feature_names, fill_value=0)

        pred_val = float(self.model.predict(X_row)[0])
        return max(10.0, pred_val)


def evaluate_forecasting_models(
    series: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, Forecaster]:
    """Train and evaluate baseline, statistical, and ML forecasting models.

    Returns:
        metrics_df: Comparison of RMSE, MAE, MAPE across models
        predictions_df: Timestamped actual vs predicted test values
        best_model: Trained Forecaster instance
    """
    n = len(series)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = series.iloc[:train_end]
    val = series.iloc[train_end:val_end]
    test = series.iloc[val_end:]

    # Model 1: Rolling 3-hr Moving Average Baseline
    ma_preds = series.rolling(3).mean().shift(1).iloc[val_end:]

    # Model 2: Seasonal Naive (24h shift)
    sn_preds = series.shift(24).iloc[val_end:]

    # Model 3: Holt-Winters Exponential Smoothing
    hw_preds = []
    try:
        hw_model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=24,
            initialization_method="estimated",
        ).fit()
        # Forecast test length
        hw_forecast = hw_model.forecast(len(test))
        hw_preds = hw_forecast.values
    except Exception:
        # Fallback if convergence issue
        hw_preds = sn_preds.fillna(train.mean()).values

    # Model 4: Random Forest with Lag Features (ML)
    lag_df = make_lags(series, n_lags=24)
    # Align split indices
    test_indices = test.index.intersection(lag_df.index)
    X_all = lag_df.drop(columns=["y"])
    y_all = lag_df["y"]

    train_mask = lag_df.index < test.index[0]
    test_mask = lag_df.index >= test.index[0]

    rf_model = Forecaster(n_lags=24)
    rf_model.fit(series.loc[series.index < test.index[0]])
    rf_preds = rf_model.model.predict(X_all.loc[test_mask])

    # Slice actual test to aligned test_indices
    actual_test = series.loc[test_indices].values
    ma_aligned = ma_preds.loc[test_indices].bfill().values
    sn_aligned = sn_preds.loc[test_indices].bfill().values
    hw_aligned = hw_preds[:len(test_indices)]
    rf_aligned = rf_preds[:len(test_indices)]

    models = {
        "Moving Average (3h)": ma_aligned,
        "Seasonal Naive (24h)": sn_aligned,
        "Holt-Winters ExpSmoothing": hw_aligned,
        "Random Forest Regressor (ML)": rf_aligned,
    }

    metrics = []
    for name, pred in models.items():
        metrics.append({
            "model": name,
            "rmse": round(rmse(actual_test, pred), 2),
            "mae": round(mae(actual_test, pred), 2),
            "mape_percent": round(mape(actual_test, pred), 2),
            "beats_baseline": bool(rmse(actual_test, pred) < rmse(actual_test, ma_aligned)),
        })

    metrics_df = pd.DataFrame(metrics).sort_values("rmse")

    # Build predictions dataframe for web API
    preds_df = pd.DataFrame({
        "time": [ts.isoformat() if hasattr(ts, "isoformat") else str(ts) for ts in test_indices],
        "actual": [round(float(x), 2) for x in actual_test],
        "predicted": [round(float(x), 2) for x in rf_aligned],
        "baseline_ma": [round(float(x), 2) for x in ma_aligned],
        "residual": [round(float(a - p), 2) for a, p in zip(actual_test, rf_aligned)],
    })

    return metrics_df, preds_df, rf_model


def run_forecasting_agent(
    raw_path: str = "data/raw/vm_cpu_readings.csv",
    output_dir: str = "data/results",
) -> Tuple[pd.DataFrame, pd.DataFrame, Forecaster]:
    """Run forecasting agent, producing forecast.csv and forecast_metrics.csv."""
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(raw_path):
        from agents.monitor import generate_synthetic_vm_data
        df = generate_synthetic_vm_data(num_vms=25, days=14)
        df.to_csv(raw_path, index=False)
    else:
        df = pd.read_csv(raw_path)

    # Hourly aggregate demand across fleet
    if "time_iso" in df.columns:
        df["dt"] = pd.to_datetime(df["time_iso"])
    elif "timestamp" in df.columns:
        df["dt"] = pd.to_datetime(df["timestamp"], unit="s")
    else:
        df["dt"] = pd.date_range("2026-01-01", periods=len(df), freq="h")

    hourly_demand = df.groupby(pd.Grouper(key="dt", freq="1h"))["avg_cpu"].sum().dropna()

    metrics_df, preds_df, model = evaluate_forecasting_models(hourly_demand)

    preds_path = os.path.join(output_dir, "forecast.csv")
    metrics_path = os.path.join(output_dir, "forecast_metrics.csv")

    preds_df.to_csv(preds_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    return metrics_df, preds_df, model
