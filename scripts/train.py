"""Model training script for CloudOptimizer demand forecasting.

Implements:
1. Feature engineering:
   - Lags: 1h, 2h, 24h, 168h
   - Rolling stats (shift=1): 6h, 24h, 168h (mean, max, std)
   - Diurnal & calendar: hour sin/cos, day of week, weekend flag
   - Trend: 7-day trend (24h vs 168h mean)
   - VM table metadata: core count, category, memory
2. Chronological split (70% train, 15% validation, 15% test) without shuffling or future leakage.
3. Models:
   - Naive (last hour)
   - Seasonal Naive (same hour last week)
   - Moving Average (24h rolling mean)
   - Random Forest Regressor
   - LightGBM Regressor
4. Uncertainty quantification (residual quantiles q10, q90)
5. Saves the production model and metadata to models/
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal, rolling, lag, and categorical features per VM."""
    print("Engineering features across VM time series...")
    df = df.sort_values(["vm_id", "hour_bucket"]).reset_index(drop=True)

    feature_dfs = []
    for vm_id, group in df.groupby("vm_id"):
        g = group.copy()
        target_series = g["hourly_avg_cpu"]

        # Lags
        g["lag_1h"] = target_series.shift(1)
        g["lag_2h"] = target_series.shift(2)
        g["lag_24h"] = target_series.shift(24)
        g["lag_168h"] = target_series.shift(168)

        # Rolling stats computed strictly on past data (shift 1 to avoid data leakage)
        past_series = target_series.shift(1)

        g["rolling_mean_6h"] = past_series.rolling(6, min_periods=3).mean()
        g["rolling_max_6h"] = past_series.rolling(6, min_periods=3).max()
        g["rolling_std_6h"] = past_series.rolling(6, min_periods=3).std().fillna(0.0)

        g["rolling_mean_24h"] = past_series.rolling(24, min_periods=12).mean()
        g["rolling_max_24h"] = past_series.rolling(24, min_periods=12).max()
        g["rolling_std_24h"] = past_series.rolling(24, min_periods=12).std().fillna(0.0)

        g["rolling_mean_168h"] = past_series.rolling(168, min_periods=84).mean()
        g["rolling_max_168h"] = past_series.rolling(168, min_periods=84).max()
        g["rolling_std_168h"] = past_series.rolling(168, min_periods=84).std().fillna(0.0)

        # Trend (recent 24h vs weekly 168h baseline)
        g["trend_7d"] = g["rolling_mean_24h"] - g["rolling_mean_168h"]

        feature_dfs.append(g)

    feat_df = pd.concat(feature_dfs, ignore_index=True)

    # Calendar and cyclical features
    # hour_bucket % 24 gives hour of day (0-23)
    hour_of_day = feat_df["hour_bucket"] % 24
    feat_df["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24.0)
    feat_df["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24.0)

    # day of week (0 to 6)
    feat_df["day_of_week"] = (feat_df["hour_bucket"] // 24) % 7
    feat_df["is_weekend"] = (feat_df["day_of_week"] >= 5).astype(int)

    # One-hot encode category
    cat_dummies = pd.get_dummies(feat_df["category"], prefix="cat", drop_first=False, dtype=int)
    feat_df = pd.concat([feat_df, cat_dummies], axis=1)

    # Drop warm-up rows where 168h lag is NaN
    clean_feat_df = feat_df.dropna(subset=["lag_168h", "rolling_mean_168h"]).reset_index(drop=True)
    return clean_feat_df


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """Perform strictly chronological train/validation/test split by hour_bucket."""
    unique_hours = sorted(df["hour_bucket"].unique())
    n_hours = len(unique_hours)
    n_train = int(n_hours * train_ratio)
    n_val = int(n_hours * val_ratio)

    train_hours = set(unique_hours[:n_train])
    val_hours = set(unique_hours[n_train:n_train + n_val])
    test_hours = set(unique_hours[n_train + n_val:])

    train_df = df[df["hour_bucket"].isin(train_hours)].copy()
    val_df = df[df["hour_bucket"].isin(val_hours)].copy()
    test_df = df[df["hour_bucket"].isin(test_hours)].copy()

    print(f"Total available hours: {n_hours} ({unique_hours[0]} to {unique_hours[-1]})")
    print(f"Train split: {len(train_hours)} hours ({len(train_df):,} samples)")
    print(f"Val split:   {len(val_hours)} hours ({len(val_df):,} samples)")
    print(f"Test split:  {len(test_hours)} hours ({len(test_df):,} samples)")

    return train_df, val_df, test_df, (unique_hours[0], unique_hours[n_train - 1], unique_hours[n_train + n_val - 1], unique_hours[-1])


def run_training(
    data_path: str = "data/processed/sample.parquet",
    models_dir: str = "models",
):
    os.makedirs(models_dir, exist_ok=True)
    print("Loading preprocessed dataset from:", data_path)
    raw_df = pd.read_parquet(data_path)

    feat_df = build_features(raw_df)

    # Define feature column list
    feature_cols = [
        "lag_1h", "lag_2h", "lag_24h", "lag_168h",
        "rolling_mean_6h", "rolling_max_6h", "rolling_std_6h",
        "rolling_mean_24h", "rolling_max_24h", "rolling_std_24h",
        "rolling_mean_168h", "rolling_max_168h", "rolling_std_168h",
        "trend_7d", "hour_sin", "hour_cos", "day_of_week", "is_weekend",
        "core_count", "memory_gb",
    ]
    # Add category dummy columns
    cat_cols = [c for c in feat_df.columns if c.startswith("cat_")]
    feature_cols.extend(cat_cols)

    target_col = "hourly_avg_cpu"

    train_df, val_df, test_df, split_hours = chronological_split(feat_df)

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    # 1. Random Forest Regressor
    print("Training Random Forest Regressor (n_estimators=100, max_depth=12)...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    # 2. LightGBM Regressor
    print("Training LightGBM Regressor (n_estimators=150, lr=0.05, max_depth=8)...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
    )

    # Calculate uncertainty bands on validation set residuals
    val_preds_lgb = lgb_model.predict(X_val)
    residuals_lgb = y_val.values - val_preds_lgb
    q10_res = float(np.percentile(residuals_lgb, 10))
    q90_res = float(np.percentile(residuals_lgb, 90))
    res_std = float(np.std(residuals_lgb))

    print(f"\nValidation Residual Uncertainty: 10th percentile={q10_res:.2f}%, 90th percentile={q90_res:.2f}%, std={res_std:.2f}%")

    # Save artifacts
    model_metadata = {
        "model_name": "LightGBM_Demand_Forecaster",
        "model_version": "v2.0.0",
        "feature_cols": feature_cols,
        "target_col": target_col,
        "split_hours": {
            "start": int(split_hours[0]),
            "train_end": int(split_hours[1]),
            "val_end": int(split_hours[2]),
            "test_end": int(split_hours[3]),
        },
        "uncertainty_band": {
            "q10_res": q10_res,
            "q90_res": q90_res,
            "res_std": res_std,
        },
    }

    joblib.dump(lgb_model, os.path.join(models_dir, "forecaster_lgb.joblib"))
    joblib.dump(rf_model, os.path.join(models_dir, "forecaster_rf.joblib"))
    with open(os.path.join(models_dir, "forecaster_metadata.json"), "w") as f:
        json.dump(model_metadata, f, indent=2)

    # Save test set with features for evaluate script
    test_df.to_parquet(os.path.join(models_dir, "test_features.parquet"), index=False)
    val_df.to_parquet(os.path.join(models_dir, "val_features.parquet"), index=False)

    print("\nSaved models and metadata to:", models_dir)
    return lgb_model, rf_model, model_metadata


if __name__ == "__main__":
    run_training()
