"""Data preparation script for Cloud Cost & Performance Optimizer forecasting.

Reads raw Azure Public Dataset V1 files in chunks according to schema.csv:
- data/raw/schema.csv
- data/raw/trace_data_vmtable_vmtable.csv.gz
- data/raw/trace_data_vm_cpu_readings_vm_cpu_readings-file-1-of-125.csv.gz

Performs:
1. Dynamic schema discovery and column name extraction from schema.csv
2. Chunked processing (never loading whole raw file into memory)
3. Sampling interval verification, time range, and missing value audit
4. Filtering VMs with at least 14 days of data (>= 336 hourly buckets)
5. Hourly resampling (mean CPU and max CPU)
6. Merging VM table metadata (vm_core_count, vm_category, vm_memory)
7. Saving clean dataset to data/processed/sample.parquet
"""
import os
import glob
import pandas as pd
import numpy as np


def parse_schema(schema_path: str = "data/raw/schema.csv") -> dict:
    """Parse schema.csv to extract real column definitions dynamically."""
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found at {schema_path}")

    schema_df = pd.read_csv(schema_path)
    # Expected columns: filepattern, field_number, content, format
    # Normalize content names into valid python identifiers
    schemas = {}
    for pattern, group in schema_df.groupby("filepattern"):
        group = group.sort_values("field_number")
        clean_cols = [
            c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            for c in group["content"]
        ]
        schemas[pattern] = clean_cols
    return schemas


def run_data_prep(
    raw_dir: str = "data/raw",
    processed_output: str = "data/processed/sample.parquet",
    min_days: int = 14,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    print("=" * 60)
    print("STEP 1: INSPECTING SCHEMA & RAW DATA CHARACTERISTICS")
    print("=" * 60)

    schema_path = os.path.join(raw_dir, "schema.csv")
    schemas = parse_schema(schema_path)

    vmtable_cols = None
    cpu_cols = None
    for pattern, cols in schemas.items():
        if "vmtable" in pattern:
            vmtable_cols = cols
        elif "vm_cpu_readings" in pattern or "cpu_readings" in pattern:
            cpu_cols = cols

    if not vmtable_cols:
        vmtable_cols = [
            "vm_id", "subscription_id", "deployment_id", "timestamp_vm_created",
            "timestamp_vm_deleted", "max_cpu", "avg_cpu", "p95_max_cpu",
            "vm_category", "vm_virtual_core_count", "vm_memory_gb"
        ]
    if not cpu_cols:
        cpu_cols = ["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]

    print(f"Parsed vmtable columns ({len(vmtable_cols)}): {vmtable_cols}")
    print(f"Parsed cpu_readings columns ({len(cpu_cols)}): {cpu_cols}")

    # Find raw files
    vmtable_files = glob.glob(os.path.join(raw_dir, "*vmtable*.csv.gz"))
    if not vmtable_files:
        raise FileNotFoundError(f"No vmtable file found in {raw_dir}")
    vmtable_file = vmtable_files[0]

    cpu_files = sorted(glob.glob(os.path.join(raw_dir, "*vm_cpu_readings*.csv.gz")))
    if not cpu_files:
        raise FileNotFoundError(f"No vm_cpu_readings file found in {raw_dir}")

    print(f"Loading metadata from: {vmtable_file}")
    vmtable_df = pd.read_csv(vmtable_file, header=None, names=vmtable_cols)
    print(f"Total VMs registered in vmtable: {len(vmtable_df):,}")

    # Standardize metadata columns of interest
    vmtable_df = vmtable_df.rename(columns={
        "vm_virtual_core_count": "core_count",
        "vm_category": "category",
        "vm_memory_gb": "memory_gb",
    })
    meta_subset = vmtable_df[["vm_id", "core_count", "category", "memory_gb"]].drop_duplicates("vm_id")

    print("\n" + "=" * 60)
    print("STEP 2: CHUNKED PROCESSING OF CPU READINGS")
    print("=" * 60)

    # Accumulate hourly metrics per VM across chunks to avoid loading full file
    # We aggregate into hourly buckets: hour_idx = timestamp // 3600
    hourly_accum = {}
    total_raw_rows = 0
    sample_intervals = []
    min_timestamp = float("inf")
    max_timestamp = float("-inf")
    nan_count = 0

    for cpu_file in cpu_files:
        print(f"Processing chunked stream from: {cpu_file} (chunksize={chunk_size:,})")
        for chunk in pd.read_csv(cpu_file, header=None, names=cpu_cols, chunksize=chunk_size):
            total_raw_rows += len(chunk)
            nan_count += int(chunk.isna().sum().sum())

            # Audit sample intervals from first chunk
            if len(sample_intervals) < 1000 and len(chunk) > 1:
                chunk_sorted = chunk.sort_values(["vm_id", "timestamp"])
                diffs = chunk_sorted.groupby("vm_id")["timestamp"].diff().dropna()
                valid_diffs = diffs[diffs > 0].head(500)
                sample_intervals.extend(valid_diffs.tolist())

            t_min = chunk["timestamp"].min()
            t_max = chunk["timestamp"].max()
            if t_min < min_timestamp:
                min_timestamp = t_min
            if t_max > max_timestamp:
                max_timestamp = t_max

            # Hourly aggregation within chunk
            chunk["hour_bucket"] = chunk["timestamp"] // 3600
            grouped = chunk.groupby(["vm_id", "hour_bucket"]).agg(
                sum_avg_cpu=("avg_cpu", "sum"),
                count_readings=("avg_cpu", "count"),
                max_cpu=("max_cpu", "max"),
                min_cpu=("min_cpu", "min"),
            ).reset_index()

            for _, row in grouped.iterrows():
                key = (row["vm_id"], int(row["hour_bucket"]))
                if key not in hourly_accum:
                    hourly_accum[key] = {
                        "sum_avg": row["sum_avg_cpu"],
                        "count": row["count_readings"],
                        "max_cpu": row["max_cpu"],
                        "min_cpu": row["min_cpu"],
                    }
                else:
                    curr = hourly_accum[key]
                    curr["sum_avg"] += row["sum_avg_cpu"]
                    curr["count"] += row["count_readings"]
                    curr["max_cpu"] = max(curr["max_cpu"], row["max_cpu"])
                    curr["min_cpu"] = min(curr["min_cpu"], row["min_cpu"])

    # Calculate sampling interval
    median_interval = np.median(sample_intervals) if sample_intervals else 300.0
    total_days = (max_timestamp - min_timestamp) / 86400.0

    print("\n" + "=" * 60)
    print("DATA AUDIT REPORT")
    print("=" * 60)
    print(f"Total raw readings streamed: {total_raw_rows:,}")
    print(f"Sampling interval detected:  {median_interval:.1f} seconds ({median_interval / 60:.1f} minutes)")
    print(f"Time range:                  {min_timestamp} to {max_timestamp} seconds ({total_days:.2f} days)")
    print(f"Missing/NaN raw values:      {nan_count}")
    print(f"Unique VM-hour buckets:      {len(hourly_accum):,}")

    # Build final hourly DataFrame
    records = []
    for (vm_id, hour_bucket), stats in hourly_accum.items():
        hourly_mean = stats["sum_avg"] / max(stats["count"], 1)
        records.append({
            "vm_id": vm_id,
            "hour_bucket": hour_bucket,
            "hourly_avg_cpu": round(hourly_mean, 2),
            "hourly_max_cpu": round(stats["max_cpu"], 2),
            "hourly_min_cpu": round(stats["min_cpu"], 2),
            "readings_in_hour": stats["count"],
        })

    hourly_df = pd.DataFrame(records)
    del hourly_accum  # Free memory immediately

    # Filter VMs with at least 14 days of data (14 * 24 = 336 hours)
    min_required_hours = min_days * 24
    vm_hour_counts = hourly_df.groupby("vm_id")["hour_bucket"].nunique()
    qualifying_vms = vm_hour_counts[vm_hour_counts >= min_required_hours].index.tolist()

    print(f"\nVMs with >= {min_days} days ({min_required_hours} hours): {len(qualifying_vms)} of {len(vm_hour_counts)}")
    if len(qualifying_vms) < 200:
        print(f"Warning: Only {len(qualifying_vms)} qualify for {min_days} days threshold. Falling back to top 300 by hours.")
        qualifying_vms = vm_hour_counts.nlargest(300).index.tolist()

    hourly_df = hourly_df[hourly_df["vm_id"].isin(qualifying_vms)].copy()

    # Re-index to ensure a complete continuous hourly time series per VM
    # Fill any sparse/missing hourly intervals with forward-fill then backward-fill
    resampled_records = []
    base_timestamp = pd.Timestamp("2026-01-01 00:00:00")

    for vm_id, group in hourly_df.groupby("vm_id"):
        min_h = group["hour_bucket"].min()
        max_h = group["hour_bucket"].max()
        all_hours = pd.DataFrame({"hour_bucket": np.arange(min_h, max_h + 1)})
        merged = pd.merge(all_hours, group, on="hour_bucket", how="left")
        merged["vm_id"] = vm_id
        merged["hourly_avg_cpu"] = merged["hourly_avg_cpu"].ffill().bfill()
        merged["hourly_max_cpu"] = merged["hourly_max_cpu"].ffill().bfill()
        merged["hourly_min_cpu"] = merged["hourly_min_cpu"].ffill().bfill()
        merged["readings_in_hour"] = merged["readings_in_hour"].fillna(0).astype(int)
        merged["timestamp"] = (merged["hour_bucket"] * 3600).astype(int)
        merged["datetime"] = base_timestamp + pd.to_timedelta(merged["hour_bucket"], unit="h")
        resampled_records.append(merged)

    clean_df = pd.concat(resampled_records, ignore_index=True)

    # Merge VM table metadata
    clean_df = pd.merge(clean_df, meta_subset, on="vm_id", how="left")
    clean_df["core_count"] = clean_df["core_count"].fillna(4).astype(int)
    clean_df["category"] = clean_df["category"].fillna("Interactive").astype(str)
    clean_df["memory_gb"] = clean_df["memory_gb"].fillna(8.0).astype(float)

    # Sort deterministically
    clean_df = clean_df.sort_values(["vm_id", "hour_bucket"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(processed_output), exist_ok=True)
    clean_df.to_parquet(processed_output, index=False)
    file_size_mb = os.path.getsize(processed_output) / (1024 * 1024)

    print("\n" + "=" * 60)
    print("DATA PREP COMPLETE")
    print("=" * 60)
    print(f"Saved: {processed_output}")
    print(f"Rows: {len(clean_df):,}")
    print(f"VMs: {clean_df['vm_id'].nunique()}")
    print(f"Average hours per VM: {clean_df.groupby('vm_id').size().mean():.1f}")
    print(f"Parquet file size: {file_size_mb:.2f} MB")
    print(f"Columns: {clean_df.columns.tolist()}")

    return clean_df


if __name__ == "__main__":
    run_data_prep()

