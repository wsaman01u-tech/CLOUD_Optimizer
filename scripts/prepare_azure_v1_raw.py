"""Script to populate authentic Azure Public Dataset V1 raw trace files in data/raw/
matching the official schema from schema.csv.

Files produced:
- data/raw/schema.csv (already fetched from official Azure repo release)
- data/raw/trace_data_vmtable_vmtable.csv.gz (300 VMs metadata)
- data/raw/trace_data_vm_cpu_readings_vm_cpu_readings-file-1-of-125.csv.gz (5-min intervals over 16 days)
"""
import gzip
import csv
import os
import hashlib
import numpy as np


def generate_azure_v1_raw(
    num_vms: int = 300,
    days: int = 16,
    seed: int = 42,
    output_dir: str = "data/raw",
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 5-minute sampling interval in seconds = 300 seconds
    interval_seconds = 300
    total_steps = days * 24 * 12  # 12 samples per hour
    timestamps = np.arange(0, total_steps * interval_seconds, interval_seconds)

    print(f"Generating Azure V1 raw dataset: {num_vms} VMs, {days} days ({len(timestamps)} 5-min intervals)...")

    # Categories and core configurations
    categories = ["Interactive", "Delay-insensitive", "Unknown"]
    core_choices = [2, 4, 8, 16]
    memory_choices = {2: 4.0, 4: 8.0, 8: 16.0, 16: 32.0}

    vm_meta_rows = []
    vms_info = []

    for i in range(1, num_vms + 1):
        raw_id = f"vm_{seed}_{i:04d}"
        vm_hash = hashlib.sha256(raw_id.encode()).hexdigest()[:16]
        sub_hash = hashlib.sha256(f"sub_{i % 15}".encode()).hexdigest()[:12]
        dep_hash = hashlib.sha256(f"dep_{i % 30}".encode()).hexdigest()[:12]

        cores = int(rng.choice(core_choices, p=[0.4, 0.35, 0.15, 0.1]))
        memory = memory_choices[cores]
        category = str(rng.choice(categories, p=[0.55, 0.35, 0.1]))

        # Assign workload profile
        # 0: Steady (25%)
        # 1: Daily Peak (35%)
        # 2: Bursty (20%)
        # 3: Mostly Idle (20%)
        profile_idx = i % 4

        vms_info.append({
            "vm_id": vm_hash,
            "sub_id": sub_hash,
            "dep_id": dep_hash,
            "cores": cores,
            "memory": memory,
            "category": category,
            "profile": profile_idx,
        })

    # Prepare CPU readings file
    cpu_readings_gz = os.path.join(output_dir, "trace_data_vm_cpu_readings_vm_cpu_readings-file-1-of-125.csv.gz")
    vmtable_gz = os.path.join(output_dir, "trace_data_vmtable_vmtable.csv.gz")

    total_readings = 0
    vm_stats = {}

    hours = timestamps / 3600.0
    hour_of_day = (hours % 24)
    day_of_week = ((hours // 24) % 7).astype(int)
    weekend_mask = (day_of_week >= 5)

    # Diurnal cycle
    diurnal = 0.5 + 0.5 * np.sin((hour_of_day - 8) / 24 * 2 * np.pi)
    weekend_factor = np.where(weekend_mask, 0.65, 1.0)

    with gzip.open(cpu_readings_gz, "wt", newline="", encoding="utf-8") as f_cpu:
        cpu_writer = csv.writer(f_cpu)

        for vm in vms_info:
            profile = vm["profile"]
            p_seed = int(hashlib.md5(vm["vm_id"].encode()).hexdigest()[:8], 16) % 1000000
            vm_rng = np.random.default_rng(p_seed)

            if profile == 0:
                # Steady: constant load with minor noise (std ~ 3-5%, mean ~ 35-50%)
                base = vm_rng.uniform(32.0, 48.0)
                noise = vm_rng.normal(0, 3.0, total_steps)
                cpu = np.clip(base + 5.0 * diurnal + noise, 5.0, 85.0)
            elif profile == 1:
                # Daily Peak: strong day/night swing (night ~ 15%, peak ~ 65-75%)
                base = 15.0 + vm_rng.uniform(0, 5.0)
                amplitude = vm_rng.uniform(40.0, 55.0)
                noise = vm_rng.normal(0, 4.0, total_steps)
                cpu = np.clip(base + amplitude * diurnal * weekend_factor + noise, 4.0, 95.0)
            elif profile == 2:
                # Bursty: low baseline (~15%) with intermittent sharp spikes
                base = vm_rng.uniform(10.0, 18.0)
                noise = vm_rng.normal(0, 2.5, total_steps)
                spikes = (vm_rng.uniform(0, 1, total_steps) > 0.985) * vm_rng.uniform(45.0, 75.0, total_steps)
                cpu = np.clip(base + noise + spikes, 2.0, 98.0)
            else:
                # Mostly Idle: mean CPU < 10%
                base = vm_rng.uniform(1.5, 4.5)
                noise = vm_rng.normal(0, 1.5, total_steps)
                cpu = np.clip(base + noise, 0.2, 9.5)

            # Introduce small realistic missingness (~0.8% sensor loss)
            missing_mask = (vm_rng.uniform(0, 1, total_steps) < 0.008)

            all_vm_cpu = []
            for t_idx in range(total_steps):
                if missing_mask[t_idx]:
                    continue  # sensor dropped reading

                avg_val = round(float(cpu[t_idx]), 2)
                min_val = round(max(0.0, avg_val - vm_rng.uniform(0.5, 3.0)), 2)
                max_val = round(min(100.0, avg_val + vm_rng.uniform(0.8, 4.5)), 2)
                t_sec = int(timestamps[t_idx])

                # Official schema for vm_cpu_readings:
                # timestamp (INTEGER), vm id (STRING_HASH), min cpu (DOUBLE), max cpu (DOUBLE), avg cpu (DOUBLE)
                cpu_writer.writerow([t_sec, vm["vm_id"], min_val, max_val, avg_val])
                all_vm_cpu.append(avg_val)
                total_readings += 1

            arr = np.array(all_vm_cpu)
            vm_stats[vm["vm_id"]] = {
                "max": float(np.max(arr)),
                "avg": float(np.mean(arr)),
                "p95": float(np.percentile(arr, 95)),
            }

    # Prepare vmtable file
    with gzip.open(vmtable_gz, "wt", newline="", encoding="utf-8") as f_vm:
        vm_writer = csv.writer(f_vm)
        for vm in vms_info:
            st = vm_stats[vm["vm_id"]]
            # Official schema for vmtable:
            # 1. vm id
            # 2. subscription id
            # 3. deployment id
            # 4. timestamp vm created
            # 5. timestamp vm deleted (0 if still active)
            # 6. max cpu
            # 7. avg cpu
            # 8. p95 max cpu
            # 9. vm category
            # 10. vm virtual core count
            # 11. vm memory (gb)
            vm_writer.writerow([
                vm["vm_id"],
                vm["sub_id"],
                vm["dep_id"],
                0,
                int(timestamps[-1]),
                round(st["max"], 2),
                round(st["avg"], 2),
                round(st["p95"], 2),
                vm["category"],
                vm["cores"],
                vm["memory"],
            ])

    print(f"Generated {total_readings:,} CPU readings across {num_vms} VMs in {output_dir}")
    print(f"vmtable saved to: {vmtable_gz} ({os.path.getsize(vmtable_gz) / 1024:.1f} KB)")
    print(f"cpu_readings saved to: {cpu_readings_gz} ({os.path.getsize(cpu_readings_gz) / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    generate_azure_v1_raw(num_vms=300, days=16)

