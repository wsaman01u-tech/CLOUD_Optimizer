"""Multi-Cloud Data Collection, Ingestion & Normalization Engine.
Collects and normalizes cloud billing and resource utilization data from
AWS, Microsoft Azure, and Google Cloud Platform (GCP) across CSV, Excel (.xlsx),
and JSON file formats into a standardized, canonical schema.

Conforms to the 7 core data dimensions:
1. Resource Information: ID, Name, Type, Cloud Provider, Region, Instance Type, Status
2. CPU Metrics: Utilization % and Historical CPU Usage (from monitoring reports)
3. Memory Metrics: Memory Utilization % and Available Memory (from monitoring reports)
4. Storage Metrics: Total Capacity, Used Storage, Storage Utilization %
5. Network Metrics: Incoming and Outgoing Traffic (MB)
6. Billing Information: Resource-wise Cost, Service-wise Cost, Total Expenditure, Currency, Billing Period
7. Usage History: Resource Uptime, Daily Usage, and Historical Performance Metrics
"""
import io
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


def parse_xlsx_zero_dep(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Parse standard OpenXML .xlsx files without third-party dependencies."""
    buf = io.BytesIO(file_bytes)
    with zipfile.ZipFile(buf, "r") as zf:
        # 1. Read shared strings if present
        shared_strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sst_xml = zf.read("xl/sharedStrings.xml")
            root_sst = ET.fromstring(sst_xml)
            ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in root_sst.findall(".//ns:si", ns):
                t = si.find("ns:t", ns)
                shared_strings.append(t.text if t is not None else "")

        # 2. Find first worksheet
        sheet_names = [n for n in zf.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
        if not sheet_names:
            return []
        sheet_xml = zf.read(sheet_names[0])
        root_sheet = ET.fromstring(sheet_xml)
        ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

        rows_data = []
        for row in root_sheet.findall(".//ns:row", ns):
            row_vals = []
            for c in row.findall("ns:c", ns):
                t_attr = c.get("t")
                v = c.find("ns:v", ns)
                if v is not None:
                    val = v.text
                    if t_attr == "s" and val.isdigit() and int(val) < len(shared_strings):
                        val = shared_strings[int(val)]
                    row_vals.append(val)
                else:
                    # check inline string
                    is_node = c.find("ns:is/ns:t", ns)
                    row_vals.append(is_node.text if is_node is not None else "")
            if row_vals:
                rows_data.append(row_vals)

    if not rows_data or len(rows_data) < 2:
        return []

    headers = [str(h).strip().lower().replace(" ", "_") for h in rows_data[0]]
    records = []
    for r in rows_data[1:]:
        rec = {}
        for i, h in enumerate(headers):
            rec[h] = r[i] if i < len(r) else None
        records.append(rec)
    return records


class CloudDataNormalizer:
    """Preprocesses, validates, sanitizes, and normalizes AWS, Azure, and GCP data into a common canonical format."""

    CANONICAL_COLUMNS = [
        # 1. Resource Information
        "resource_id", "resource_name", "resource_type", "cloud_provider", "region", "current_size", "status",
        # 2. CPU Metrics
        "avg_cpu", "peak_cpu",
        # 3. Memory Metrics
        "memory_utilization_pct", "total_memory_gb", "available_memory_gb",
        # 4. Storage Metrics
        "total_storage_gb", "used_storage_gb", "storage_utilization_pct",
        # 5. Network Metrics
        "network_in_mb", "network_out_mb",
        # 6. Billing Information
        "monthly_cost", "service_cost", "currency", "billing_period",
        # 7. Usage History
        "uptime_hours", "daily_usage_hours", "history_points", "is_protected", "has_missing_data", "environment"
    ]

    # Cloud identification keywords and markers
    CLOUD_RESOURCE_HEADERS = {
        "resource_id", "resourceid", "instance_id", "instanceid", "vm_name", "vmname",
        "server_id", "servername", "server_name", "machine_id", "machine_name",
        "arn", "lineitem/resourceid", "lineitem_resourceid", "node_id", "host_id",
        "instance_type", "instancetype", "machine_type", "machinetype", "vmsize", "vm_size",
        "cloud_provider", "provider", "platform", "provider_name", "cloud",
        "region", "resourcelocation", "location", "availability_zone", "zone"
    }

    CLOUD_METRIC_HEADERS = {
        "cpu", "avg_cpu", "average_cpu", "peak_cpu", "max_cpu", "cpu_utilization",
        "cpu_utilization_pct", "cpuutilization", "cpu_percent", "cpu_percentage",
        "memory", "ram", "ram_gb", "total_memory", "total_memory_gb", "memory_utilization",
        "memory_utilization_pct", "mem_util", "memory_pct", "available_memory", "available_memory_gb",
        "storage", "disk", "total_storage", "total_storage_gb", "used_storage", "used_storage_gb",
        "disk_gb", "used_disk_gb", "storage_utilization_pct", "disk_utilization",
        "network_in", "network_in_mb", "networkin", "network_out", "network_out_mb", "networkout",
        "bandwidth", "iops", "uptime", "uptime_hours", "daily_usage_hours", "vcpus", "cores"
    }

    CLOUD_BILLING_HEADERS = {
        "cost", "monthly_cost", "resource_cost", "service_cost", "pretaxcost", "pre_tax_cost",
        "unblended_cost", "blended_cost", "lineitem/unblendedcost", "spend", "monthly_spend",
        "currency", "billing_period", "charge_type", "pricing_unit", "usage_amount", "consumed_service"
    }

    # Foreign Non-Cloud Domain Disqualifiers
    NON_CLOUD_DOMAINS = {
        "Academic / Student": {
            "student", "student_id", "student_name", "grade", "gpa", "score", "marks",
            "subject", "classroom", "teacher", "exam", "semester", "roll_no", "attendance"
        },
        "E-Commerce / Retail": {
            "order_id", "product_name", "product_id", "customer_id", "customer_email",
            "shipping_address", "cart_total", "discount_code", "inventory_count", "item_price", "sku"
        },
        "HR / Payroll": {
            "employee_id", "employee_name", "salary", "job_title", "department", "manager",
            "hire_date", "bonus", "payroll", "ssn", "vacation_days"
        },
        "Healthcare / Medical": {
            "patient_id", "doctor", "diagnosis", "blood_pressure", "prescription", "hospital",
            "treatment", "dosage", "symptoms"
        },
    }

    @classmethod
    def clean_numeric(cls, raw_val: Any, default: float = 0.0, min_val: Optional[float] = None, max_val: Optional[float] = None, tracker: Optional[Dict[str, int]] = None) -> float:
        """Clean unstructured, noisy numeric strings (currency symbols, commas, percent, whitespace)."""
        if raw_val is None or pd.isna(raw_val):
            if tracker is not None:
                tracker["nulls_imputed"] += 1
            return default

        if isinstance(raw_val, (int, float)):
            val = float(raw_val)
        elif isinstance(raw_val, str):
            cleaned = raw_val.strip()
            if not cleaned or cleaned.lower() in ["nan", "none", "null", "n/a", "-", "--", ""]:
                if tracker is not None:
                    tracker["nulls_imputed"] += 1
                return default

            # Detect & strip currency symbols
            had_currency = False
            for sym in ["$", "€", "£", "₹", "¥", "usd", "eur", "inr"]:
                if sym in cleaned.lower():
                    had_currency = True
                    cleaned = re.sub(re.escape(sym), "", cleaned, flags=re.IGNORECASE)
            if had_currency and tracker is not None:
                tracker["currency_cleaned"] += 1

            # Strip commas, percent, and stray whitespace
            cleaned = cleaned.replace(",", "").replace("%", "").strip()

            try:
                val = float(cleaned)
            except ValueError:
                # Fallback: extract the first contiguous number
                match = re.search(r"[-+]?(?:\d*\.\d+|\d+)", cleaned)
                if match:
                    val = float(match.group(0))
                else:
                    if tracker is not None:
                        tracker["nulls_imputed"] += 1
                    return default
        else:
            try:
                val = float(raw_val)
            except Exception:
                if tracker is not None:
                    tracker["nulls_imputed"] += 1
                return default

        # Clamp bounds if provided
        if min_val is not None and val < min_val:
            if tracker is not None:
                tracker["outliers_clamped"] += 1
            val = min_val
        if max_val is not None and val > max_val:
            if tracker is not None:
                tracker["outliers_clamped"] += 1
            val = max_val

        return val

    @classmethod
    def clean_memory_gb(cls, raw_val: Any, default: float = 8.0, tracker: Optional[Dict[str, int]] = None) -> float:
        """Parse memory values with heterogeneous units (e.g., '16 GB', '16384 MB', '1 TB') into standard GB."""
        if isinstance(raw_val, str):
            s = raw_val.strip().lower()
            if "mb" in s or "mib" in s:
                num = cls.clean_numeric(re.sub(r"[a-z]", "", s), default=default * 1024.0, tracker=tracker)
                if tracker is not None:
                    tracker["units_converted"] += 1
                return round(num / 1024.0, 2)
            elif "tb" in s or "tib" in s:
                num = cls.clean_numeric(re.sub(r"[a-z]", "", s), default=default / 1024.0, tracker=tracker)
                if tracker is not None:
                    tracker["units_converted"] += 1
                return round(num * 1024.0, 2)
            elif "gb" in s or "gib" in s:
                if tracker is not None:
                    tracker["units_converted"] += 1
                return cls.clean_numeric(re.sub(r"[a-z]", "", s), default=default, min_val=0.5, max_val=4096.0, tracker=tracker)

        return cls.clean_numeric(raw_val, default=default, min_val=0.5, max_val=4096.0, tracker=tracker)

    @classmethod
    def clean_storage_gb(cls, raw_val: Any, default: float = 100.0, tracker: Optional[Dict[str, int]] = None) -> float:
        """Parse storage values with mixed units (e.g., '500 GB', '2 TB', '250000 MB') into standard GB."""
        if isinstance(raw_val, str):
            s = raw_val.strip().lower()
            if "tb" in s or "tib" in s:
                num = cls.clean_numeric(re.sub(r"[a-z]", "", s), default=default / 1024.0, tracker=tracker)
                if tracker is not None:
                    tracker["units_converted"] += 1
                return round(num * 1024.0, 2)
            elif "mb" in s or "mib" in s:
                num = cls.clean_numeric(re.sub(r"[a-z]", "", s), default=default * 1024.0, tracker=tracker)
                if tracker is not None:
                    tracker["units_converted"] += 1
                return round(num / 1024.0, 2)
            elif "gb" in s or "gib" in s:
                if tracker is not None:
                    tracker["units_converted"] += 1
                return cls.clean_numeric(re.sub(r"[a-z]", "", s), default=default, min_val=1.0, max_val=65536.0, tracker=tracker)

        return cls.clean_numeric(raw_val, default=default, min_val=1.0, max_val=65536.0, tracker=tracker)

    @classmethod
    def clean_network_mb(cls, raw_val: Any, default: float = 50.0, tracker: Optional[Dict[str, int]] = None) -> float:
        """Parse network values with mixed units ('1.5 GB', '2048 KB', '500 MB') into standard MB."""
        if isinstance(raw_val, str):
            s = raw_val.strip().lower()
            if "gb" in s or "gib" in s:
                num = cls.clean_numeric(re.sub(r"[a-z]", "", s), default=default / 1024.0, tracker=tracker)
                if tracker is not None:
                    tracker["units_converted"] += 1
                return round(num * 1024.0, 2)
            elif "kb" in s or "kib" in s:
                num = cls.clean_numeric(re.sub(r"[a-z]", "", s), default=default * 1024.0, tracker=tracker)
                if tracker is not None:
                    tracker["units_converted"] += 1
                return round(num / 1024.0, 2)
            elif "mb" in s or "mib" in s:
                if tracker is not None:
                    tracker["units_converted"] += 1
                return cls.clean_numeric(re.sub(r"[a-z]", "", s), default=default, min_val=0.0, max_val=1000000.0, tracker=tracker)

        return cls.clean_numeric(raw_val, default=default, min_val=0.0, max_val=1000000.0, tracker=tracker)

    @classmethod
    def clean_cpu_utilization(cls, raw_val: Any, default: float = 15.0, tracker: Optional[Dict[str, int]] = None) -> float:
        """Clean CPU utilization: converts fractional scales [0.0..1.0] to percentage, clamps [0..100]."""
        val = cls.clean_numeric(raw_val, default=default, tracker=tracker)

        # Detect fractional scaling (e.g. 0.18 representing 18% CPU utilization)
        if 0.0 < val <= 1.0 and isinstance(raw_val, (float, str)) and (isinstance(raw_val, float) or "." in str(raw_val)):
            val = round(val * 100.0, 1)
            if tracker is not None:
                tracker["fractions_scaled"] += 1

        # Clamp bounds [0.0, 100.0]
        if val < 0.0:
            if tracker is not None:
                tracker["outliers_clamped"] += 1
            val = 0.0
        elif val > 100.0:
            if tracker is not None:
                tracker["outliers_clamped"] += 1
            val = 100.0

        return round(val, 1)

    @classmethod
    def validate_cloud_dataset(cls, records: List[Dict[str, Any]], filename: str = "upload") -> Dict[str, Any]:
        """Examines column schema, keys, and values to verify dataset belongs to Cloud Infrastructure & Billing.
        Raises ValueError with an informative error if foreign / non-cloud data is uploaded.
        """
        if not records or not isinstance(records, list):
            raise ValueError(f"Uploaded file '{filename}' contains no readable data rows.")

        # Collect all unique lowercased header keys across the records
        raw_headers = set()
        for r in records[:50]:
            if isinstance(r, dict):
                for k in r.keys():
                    raw_headers.add(str(k).strip().lower())

        if not raw_headers:
            raise ValueError(f"Uploaded file '{filename}' contains no valid headers or columns.")

        # Check for Foreign Non-Cloud Domains
        for domain_name, keywords in cls.NON_CLOUD_DOMAINS.items():
            matched_foreign = raw_headers.intersection(keywords)
            if len(matched_foreign) >= 2:
                raise ValueError(
                    f"Invalid Cloud Dataset: The uploaded file '{filename}' appears to be {domain_name} data "
                    f"(detected columns: {sorted(list(matched_foreign))}). "
                    f"CloudOptimizer specifically optimizes AWS, Azure, and Google Cloud infrastructure and billing. "
                    f"Please upload cloud monitoring reports or billing exports (CSV, Excel .xlsx, or JSON)."
                )

        # Check for Cloud Signals across Resource, Metric, and Billing dimensions
        res_matches = raw_headers.intersection(cls.CLOUD_RESOURCE_HEADERS)
        metric_matches = raw_headers.intersection(cls.CLOUD_METRIC_HEADERS)
        billing_matches = raw_headers.intersection(cls.CLOUD_BILLING_HEADERS)

        # Value inspection across sample records for cloud resource patterns
        value_cloud_signals = 0
        for r in records[:10]:
            if not isinstance(r, dict):
                continue
            for v in r.values():
                val_str = str(v).lower()
                if (
                    "arn:aws:" in val_str or "/subscriptions/" in val_str or "projects/" in val_str or
                    val_str.startswith("i-") or "standard_" in val_str or "t3." in val_str or
                    "m5." in val_str or "c5." in val_str or "e2-" in val_str or "n1-" in val_str or
                    "aws" in val_str or "azure" in val_str or "gcp" in val_str
                ):
                    value_cloud_signals += 1

        total_cloud_signals = len(res_matches) + len(metric_matches) + len(billing_matches) + (1 if value_cloud_signals > 0 else 0)

        # If zero cloud signals detected, reject with clear error
        if total_cloud_signals == 0:
            raise ValueError(
                f"Invalid Cloud Dataset: The uploaded file '{filename}' does not contain recognized cloud infrastructure, "
                f"utilization metrics, or billing data. Found columns: {sorted(list(raw_headers))}. "
                f"Expected cloud fields such as 'resource_id', 'instance_type', 'cpu_utilization', 'memory_gb', 'cost', "
                f"or 'cloud_provider' (AWS, Azure, or GCP). Please upload a valid cloud monitoring or billing export."
            )

        return {
            "valid": True,
            "resource_signals": list(res_matches),
            "metric_signals": list(metric_matches),
            "billing_signals": list(billing_matches),
            "value_patterns_detected": value_cloud_signals,
            "total_signals": total_cloud_signals,
        }

    @classmethod
    def load_file(cls, filename: str, content: bytes) -> Tuple[List[Dict[str, Any]], str]:
        """Load CSV, Excel (.xlsx), or JSON file into raw records."""
        lower_name = filename.lower()
        if lower_name.endswith(".json"):
            text = content.decode("utf-8", errors="replace")
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "resources" in parsed:
                records = parsed["resources"]
            elif isinstance(parsed, list):
                records = parsed
            else:
                records = [parsed]
            return records, "JSON"

        elif lower_name.endswith(".xlsx") or lower_name.endswith(".xls"):
            records = parse_xlsx_zero_dep(content)
            return records, "Excel (XLSX)"

        elif lower_name.endswith(".csv") or lower_name.endswith(".txt"):
            text = content.decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(text))
            return df.to_dict(orient="records"), "CSV"

        else:
            # Attempt CSV fallback
            text = content.decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(text))
            return df.to_dict(orient="records"), "CSV"

    @classmethod
    def detect_provider(cls, row: Dict[str, Any]) -> str:
        """Identify cloud provider (AWS, Azure, GCP) from explicit fields or resource metadata."""
        for k in ["cloud_provider", "provider", "cloud", "platform", "provider_name"]:
            if k in row and row[k]:
                val = str(row[k]).strip().lower()
                if "aws" in val or "amazon" in val:
                    return "AWS"
                if "azure" in val or "microsoft" in val:
                    return "Azure"
                if "gcp" in val or "google" in val:
                    return "GCP"

        # Heuristic detection from ID or size naming
        res_id = str(row.get("resource_id", row.get("ResourceId", row.get("id", "")))).lower()
        size = str(row.get("current_size", row.get("instance_type", row.get("InstanceType", "")))).lower()

        if "standard_" in size or "/subscriptions/" in res_id or "azure" in res_id:
            return "Azure"
        if "e2-" in size or "n1-" in size or "n2-" in size or "projects/" in res_id or "gcp" in res_id:
            return "GCP"
        if "t3." in size or "c5." in size or "m5." in size or "i-" in res_id or "arn:aws" in res_id or "ec2" in res_id:
            return "AWS"

        return "AWS"  # Default assumption

    @classmethod
    def normalize_record(cls, raw: Dict[str, Any], index: int = 1, tracker: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """Map raw heterogenous provider record to canonical 7-dimension schema, cleaning noise & dirty values."""
        # Clean keys to lowercase stripped
        r = {str(k).strip().lower(): v for k, v in raw.items()}

        provider = cls.detect_provider(raw)

        # 1. Resource Information
        raw_id = r.get("resource_id") or r.get("resourceid") or r.get("id") or r.get("lineitem/resourceid") or r.get("vm_name")
        if raw_id:
            res_id = str(raw_id).strip()
            if tracker is not None and res_id != str(raw_id):
                tracker["casing_and_whitespace_trimmed"] += 1
        else:
            res_id = f"{provider.lower()}-res-{index:03d}"
            if tracker is not None:
                tracker["nulls_imputed"] += 1

        raw_name = r.get("resource_name") or r.get("resourcename") or r.get("name") or r.get("tags/name")
        res_name = str(raw_name).strip() if raw_name else f"{str(res_id).split('/')[-1]}-service"

        res_type = str(r.get("resource_type") or r.get("resourcetype") or r.get("type") or "compute").lower().strip()
        if "vol" in str(res_id).lower() or "disk" in str(res_id).lower() or "storage" in res_type or "ebs" in res_type:
            res_type = "storage"
        elif "db" in str(res_id).lower() or "sql" in str(res_id).lower() or "database" in res_type:
            res_type = "database"
        else:
            res_type = "compute"

        region_raw = r.get("region") or r.get("location") or r.get("resourcelocation")
        if region_raw:
            region = str(region_raw).strip()
        else:
            region = "us-east-1" if provider == "AWS" else ("eastus" if provider == "Azure" else "us-central1")
            if tracker is not None:
                tracker["nulls_imputed"] += 1

        size_raw = (
            r.get("current_size") or r.get("instance_type") or r.get("instancetype") or
            r.get("vmsize") or r.get("machine_type")
        )
        if size_raw:
            size = str(size_raw).strip()
        else:
            size = "t3.large" if provider == "AWS" else ("Standard_D2s_v3" if provider == "Azure" else "e2-standard-2")
            if tracker is not None:
                tracker["nulls_imputed"] += 1

        env_raw = str(r.get("environment") or r.get("env") or "").lower().strip()
        if not env_raw:
            env = "production" if "prod" in str(res_id).lower() else "development"
            if tracker is not None:
                tracker["nulls_imputed"] += 1
        elif "prod" in env_raw:
            env = "production"
        else:
            env = "development"

        # 2. CPU Metrics (from monitoring report) - sanitized with scale/outlier detection
        raw_avg_cpu = (
            r.get("avg_cpu") or r.get("cpu_utilization") or r.get("cpu_utilization_pct") or
            r.get("cpuutilization") or r.get("cpu_percent") or r.get("cpu")
        )
        avg_cpu = cls.clean_cpu_utilization(raw_avg_cpu, default=15.0, tracker=tracker)

        raw_peak_cpu = r.get("peak_cpu") or r.get("peak_cpu_pct") or r.get("maxcpu") or r.get("max_cpu")
        if raw_peak_cpu is not None and not pd.isna(raw_peak_cpu):
            peak_cpu = cls.clean_cpu_utilization(raw_peak_cpu, default=min(100.0, avg_cpu * 1.3), tracker=tracker)
        else:
            peak_cpu = min(100.0, round(avg_cpu * 1.3, 1))

        # 3. Memory Metrics (from monitoring report) - handles unit conversion (MB, GB, TB)
        raw_mem_util = (
            r.get("memory_utilization_pct") or r.get("memory_utilization") or
            r.get("mem_util") or r.get("memory_pct")
        )
        if raw_mem_util is not None and not pd.isna(raw_mem_util):
            mem_util = cls.clean_cpu_utilization(raw_mem_util, default=round(avg_cpu * 0.9 + 10.0, 1), tracker=tracker)
        else:
            mem_util = round(avg_cpu * 0.9 + 10.0, 1)

        raw_total_mem = (
            r.get("total_memory_gb") or r.get("total_memory") or r.get("memory_gb") or
            r.get("memory") or r.get("ram_gb") or r.get("ram")
        )
        total_mem = cls.clean_memory_gb(raw_total_mem, default=8.0, tracker=tracker)
        avail_mem = round(max(0.1, total_mem * (1.0 - (mem_util / 100.0))), 1)

        # 4. Storage Metrics (from monitoring report) - handles unit conversion (MB, GB, TB)
        raw_total_storage = (
            r.get("total_storage_gb") or r.get("total_storage") or r.get("disk_gb") or
            r.get("storage_gb") or r.get("disk") or r.get("storage")
        )
        default_storage = 500.0 if res_type == "storage" else 100.0
        total_storage = cls.clean_storage_gb(raw_total_storage, default=default_storage, tracker=tracker)

        raw_used_storage = r.get("used_storage_gb") or r.get("used_storage") or r.get("used_disk_gb") or r.get("used_disk")
        if raw_used_storage is not None and not pd.isna(raw_used_storage):
            used_storage = cls.clean_storage_gb(raw_used_storage, default=total_storage * 0.4, tracker=tracker)
        else:
            used_storage = 0.0 if ("unattached" in str(res_id).lower() or (res_type == "storage" and avg_cpu == 0)) else round(total_storage * 0.35, 1)

        storage_util = round((min(used_storage, total_storage) / max(1.0, total_storage)) * 100.0, 1)

        # 5. Network Metrics (from monitoring report) - handles unit conversion
        raw_net_in = r.get("network_in_mb") or r.get("network_in") or r.get("networkin")
        raw_net_out = r.get("network_out_mb") or r.get("network_out") or r.get("networkout")
        net_in = cls.clean_network_mb(raw_net_in, default=round(avg_cpu * 8.5, 1), tracker=tracker)
        net_out = cls.clean_network_mb(raw_net_out, default=round(avg_cpu * 12.0, 1), tracker=tracker)

        # 6. Billing Information (from billing file) - cleans dirty currency strings
        raw_cost = r.get("monthly_cost") or r.get("resource_cost") or r.get("cost") or r.get("pretaxcost") or r.get("unblended_cost")
        cost = cls.clean_numeric(raw_cost, default=59.90, min_val=0.0, tracker=tracker)

        currency = str(r.get("currency") or "USD").upper().strip()
        billing_period = str(r.get("billing_period") or "2026-01-01 to 2026-01-31").strip()

        # 7. Usage History
        raw_uptime = r.get("uptime_hours") or r.get("uptime")
        uptime = cls.clean_numeric(raw_uptime, default=720.0, min_val=0.0, max_val=744.0, tracker=tracker)
        daily_usage = cls.clean_numeric(r.get("daily_usage_hours"), default=round(min(24.0, uptime / 30.0), 1), min_val=0.0, max_val=24.0, tracker=tracker)

        # Status inference engine
        explicit_status = r.get("status") or r.get("resource_status")
        if explicit_status and str(explicit_status).lower().strip() in ["idle", "over_provisioned", "healthy", "hot", "unattached"]:
            status = str(explicit_status).lower().strip()
        else:
            if res_type == "storage" and used_storage == 0:
                status = "unattached"
            elif avg_cpu < 10.0 and uptime >= 24.0:
                status = "idle"
            elif peak_cpu < 30.0 and mem_util < 45.0:
                status = "over_provisioned"
            elif peak_cpu > 80.0 or mem_util > 85.0:
                status = "hot"
            else:
                status = "healthy"

        pct_low = 98.0 if status == "idle" else (15.0 if status == "over_provisioned" else 2.0)
        history_points = int(cls.clean_numeric(r.get("history_points"), default=336, min_val=24, max_val=720))
        is_protected = env == "production"

        return {
            "resource_id": str(res_id),
            "resource_name": str(res_name),
            "resource_type": res_type,
            "cloud_provider": provider,
            "region": str(region),
            "current_size": str(size),
            "environment": env,
            "status": status,
            "avg_cpu": round(avg_cpu, 1),
            "peak_cpu": round(peak_cpu, 1),
            "pct_low": pct_low,
            "memory_utilization_pct": mem_util,
            "total_memory_gb": total_mem,
            "available_memory_gb": avail_mem,
            "total_storage_gb": total_storage,
            "used_storage_gb": used_storage,
            "storage_utilization_pct": storage_util,
            "network_in_mb": net_in,
            "network_out_mb": net_out,
            "monthly_cost": round(cost, 2),
            "service_cost": round(cost, 2),
            "currency": currency,
            "billing_period": billing_period,
            "uptime_hours": uptime,
            "daily_usage_hours": daily_usage,
            "history_points": history_points,
            "is_protected": is_protected,
            "has_missing_data": False,
        }

    @classmethod
    def normalize_dataset(cls, raw_records: List[Dict[str, Any]], filename: str = "uploaded_fleet") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Validates, cleans, and normalizes raw multi-cloud records into canonical fleet DataFrame and telemetry."""
        # Step 1: Validate cloud dataset domain & schema
        cls.validate_cloud_dataset(raw_records, filename=filename)

        # Step 2: Track cleaning operations on noisy/dirty data
        cleaning_tracker = {
            "currency_cleaned": 0,
            "units_converted": 0,
            "outliers_clamped": 0,
            "fractions_scaled": 0,
            "nulls_imputed": 0,
            "casing_and_whitespace_trimmed": 0,
        }

        canonical_rows = []
        for i, raw in enumerate(raw_records, 1):
            if not isinstance(raw, dict):
                continue
            canonical_rows.append(cls.normalize_record(raw, index=i, tracker=cleaning_tracker))

        fleet_df = pd.DataFrame(canonical_rows)

        # Step 3: Generate accompanying multi-day monitoring telemetry time series
        base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        telemetry_rows = []
        for _, res in fleet_df.iterrows():
            pts = int(res["history_points"])
            avg = float(res["avg_cpu"])
            peak = float(res["peak_cpu"])
            std = max(1.0, (peak - avg) / 2.5)

            for h in range(pts):
                t = base_time + pd.Timedelta(hours=h)
                diurnal = 0.5 * np.sin(2 * np.pi * (h % 24) / 24)
                val = float(np.clip(np.random.normal(avg + diurnal, std), 0.0, 100.0))
                telemetry_rows.append({
                    "timestamp": int(t.timestamp()),
                    "time_iso": t.isoformat(),
                    "resource_id": res["resource_id"],
                    "avg_cpu": round(val, 2),
                    "peak_cpu": round(min(100.0, val * 1.25), 2),
                })

        telemetry_df = pd.DataFrame(telemetry_rows)

        # Step 4: Report breakdown and cleaning audit summary
        providers = fleet_df["cloud_provider"].value_counts().to_dict()
        status_dist = fleet_df["status"].value_counts().to_dict()

        cleaning_parts = []
        if cleaning_tracker["currency_cleaned"] > 0:
            cleaning_parts.append(f"{cleaning_tracker['currency_cleaned']} currency strings")
        if cleaning_tracker["units_converted"] > 0:
            cleaning_parts.append(f"{cleaning_tracker['units_converted']} memory/storage units")
        if cleaning_tracker["fractions_scaled"] > 0:
            cleaning_parts.append(f"{cleaning_tracker['fractions_scaled']} CPU fraction scales")
        if cleaning_tracker["outliers_clamped"] > 0:
            cleaning_parts.append(f"{cleaning_tracker['outliers_clamped']} outlier limits")
        if cleaning_tracker["nulls_imputed"] > 0:
            cleaning_parts.append(f"{cleaning_tracker['nulls_imputed']} missing values imputed")

        cleaning_summary_text = (
            f"Cleaned and sanitized: {', '.join(cleaning_parts)}."
            if cleaning_parts else "Dataset was already clean and properly structured."
        )

        summary_report = {
            "total_resources": len(fleet_df),
            "cloud_providers": providers,
            "status_distribution": status_dist,
            "total_monthly_spend": round(float(fleet_df["monthly_cost"].sum()), 2),
            "currency": fleet_df["currency"].iloc[0] if not fleet_df.empty else "USD",
            "telemetry_points_generated": len(telemetry_df),
            "cleaning_report": cleaning_tracker,
            "cleaning_summary": cleaning_summary_text,
        }

        return fleet_df, telemetry_df, summary_report


def generate_multi_cloud_sample_fleet() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates a rich, balanced 12-resource multi-cloud fleet covering AWS, Azure, and GCP."""
    raw_sample = [
        # AWS Resources
        {
            "resource_id": "ec2-prod-web-01",
            "resource_name": "checkout-web-cluster",
            "resource_type": "compute",
            "cloud_provider": "AWS",
            "region": "us-east-1",
            "current_size": "t3.2xlarge",
            "monthly_cost": 239.62,
            "avg_cpu": 3.5,
            "peak_cpu": 6.8,
            "memory_utilization_pct": 12.0,
            "total_memory_gb": 32.0,
            "total_storage_gb": 200.0,
            "network_in_mb": 42.0,
            "network_out_mb": 65.0,
            "status": "idle",
            "environment": "production",
            "uptime_hours": 720.0,
        },
        {
            "resource_id": "ec2-prod-api-02",
            "resource_name": "payments-api-gateway",
            "resource_type": "compute",
            "cloud_provider": "AWS",
            "region": "us-east-1",
            "current_size": "t3.xlarge",
            "monthly_cost": 119.81,
            "avg_cpu": 18.2,
            "peak_cpu": 28.4,
            "memory_utilization_pct": 34.0,
            "total_memory_gb": 16.0,
            "total_storage_gb": 100.0,
            "network_in_mb": 310.0,
            "network_out_mb": 450.0,
            "status": "over_provisioned",
            "environment": "production",
            "uptime_hours": 720.0,
        },
        {
            "resource_id": "vol-dev-unattached-01",
            "resource_name": "unattached-test-ebs-volume",
            "resource_type": "storage",
            "cloud_provider": "AWS",
            "region": "us-west-2",
            "current_size": "500 GB gp3",
            "monthly_cost": 40.00,
            "avg_cpu": 0.0,
            "peak_cpu": 0.0,
            "memory_utilization_pct": 0.0,
            "total_memory_gb": 0.0,
            "total_storage_gb": 500.0,
            "used_storage_gb": 0.0,
            "status": "unattached",
            "environment": "development",
            "uptime_hours": 720.0,
        },
        # Microsoft Azure Resources
        {
            "resource_id": "azure-vm-prod-analytics-01",
            "resource_name": "synapse-telemetry-worker",
            "resource_type": "compute",
            "cloud_provider": "Azure",
            "region": "eastus",
            "current_size": "Standard_D4s_v3",
            "monthly_cost": 140.16,
            "avg_cpu": 16.5,
            "peak_cpu": 26.0,
            "memory_utilization_pct": 28.0,
            "total_memory_gb": 16.0,
            "total_storage_gb": 128.0,
            "network_in_mb": 180.0,
            "network_out_mb": 240.0,
            "status": "over_provisioned",
            "environment": "production",
            "uptime_hours": 720.0,
        },
        {
            "resource_id": "azure-vm-staging-auth-02",
            "resource_name": "entra-id-sync-proxy",
            "resource_type": "compute",
            "cloud_provider": "Azure",
            "region": "westeurope",
            "current_size": "Standard_D2s_v3",
            "monthly_cost": 70.08,
            "avg_cpu": 4.2,
            "peak_cpu": 8.1,
            "memory_utilization_pct": 14.0,
            "total_memory_gb": 8.0,
            "total_storage_gb": 64.0,
            "network_in_mb": 25.0,
            "network_out_mb": 35.0,
            "status": "idle",
            "environment": "staging",
            "uptime_hours": 720.0,
        },
        {
            "resource_id": "azure-vm-prod-cache-03",
            "resource_name": "redis-enterprise-node",
            "resource_type": "compute",
            "cloud_provider": "Azure",
            "region": "eastus",
            "current_size": "Standard_B2s",
            "monthly_cost": 30.36,
            "avg_cpu": 46.0,
            "peak_cpu": 63.5,
            "memory_utilization_pct": 68.0,
            "total_memory_gb": 4.0,
            "total_storage_gb": 32.0,
            "network_in_mb": 850.0,
            "network_out_mb": 920.0,
            "status": "healthy",
            "environment": "production",
            "uptime_hours": 720.0,
        },
        # Google Cloud Platform (GCP) Resources
        {
            "resource_id": "gcp-gce-prod-db-01",
            "resource_name": "cloud-sql-postgres-primary",
            "resource_type": "database",
            "cloud_provider": "GCP",
            "region": "us-central1",
            "current_size": "e2-standard-4",
            "monthly_cost": 97.60,
            "avg_cpu": 87.5,
            "peak_cpu": 96.2,
            "memory_utilization_pct": 89.0,
            "total_memory_gb": 16.0,
            "total_storage_gb": 250.0,
            "network_in_mb": 620.0,
            "network_out_mb": 1150.0,
            "status": "hot",
            "environment": "production",
            "uptime_hours": 720.0,
        },
        {
            "resource_id": "gcp-gce-dev-worker-02",
            "resource_name": "pubsub-batch-consumer",
            "resource_type": "compute",
            "cloud_provider": "GCP",
            "region": "us-central1",
            "current_size": "e2-standard-2",
            "monthly_cost": 48.80,
            "avg_cpu": 2.9,
            "peak_cpu": 5.4,
            "memory_utilization_pct": 18.0,
            "total_memory_gb": 8.0,
            "total_storage_gb": 50.0,
            "network_in_mb": 15.0,
            "network_out_mb": 20.0,
            "status": "idle",
            "environment": "development",
            "uptime_hours": 720.0,
        },
        {
            "resource_id": "gcp-gce-staging-app-03",
            "resource_name": "cloud-run-gke-staging",
            "resource_type": "compute",
            "cloud_provider": "GCP",
            "region": "europe-west1",
            "current_size": "e2-standard-2",
            "monthly_cost": 48.80,
            "avg_cpu": 15.2,
            "peak_cpu": 25.1,
            "memory_utilization_pct": 32.0,
            "total_memory_gb": 8.0,
            "total_storage_gb": 80.0,
            "network_in_mb": 95.0,
            "network_out_mb": 110.0,
            "status": "over_provisioned",
            "environment": "staging",
            "uptime_hours": 720.0,
        },
        {
            "resource_id": "gcp-gce-prod-edge-04",
            "resource_name": "cloud-armor-ingress-proxy",
            "resource_type": "compute",
            "cloud_provider": "GCP",
            "region": "us-central1",
            "current_size": "e2-small",
            "monthly_cost": 14.60,
            "avg_cpu": 42.0,
            "peak_cpu": 58.0,
            "memory_utilization_pct": 52.0,
            "total_memory_gb": 2.0,
            "total_storage_gb": 20.0,
            "network_in_mb": 1250.0,
            "network_out_mb": 1400.0,
            "status": "healthy",
            "environment": "production",
            "uptime_hours": 720.0,
        },
        # Insufficient data / Missing data guardrail test instances
        {
            "resource_id": "ec2-aws-dev-new-box",
            "resource_name": "sandbox-temp-instance",
            "resource_type": "compute",
            "cloud_provider": "AWS",
            "region": "us-east-1",
            "current_size": "t3.large",
            "monthly_cost": 59.90,
            "avg_cpu": 4.1,
            "peak_cpu": 7.2,
            "status": "idle",
            "environment": "development",
            "history_points": 8,  # Insufficient history points (< 24)
        },
        {
            "resource_id": "azure-corrupted-telemetry",
            "resource_name": "unhealthy-diagnostics-vm",
            "resource_type": "compute",
            "cloud_provider": "Azure",
            "region": "eastus",
            "current_size": "Standard_B1s",
            "monthly_cost": 15.18,
            "avg_cpu": np.nan,
            "peak_cpu": np.nan,
            "status": "unknown",
            "environment": "staging",
            "history_points": 0,
        }
    ]

    fleet_df, telemetry_df, _ = CloudDataNormalizer.normalize_dataset(raw_sample)
    # Mark corrupted item explicitly
    idx = fleet_df[fleet_df["resource_id"] == "azure-corrupted-telemetry"].index
    if not idx.empty:
        fleet_df.loc[idx[0], "has_missing_data"] = True
        fleet_df.loc[idx[0], "avg_cpu"] = np.nan
        fleet_df.loc[idx[0], "peak_cpu"] = np.nan

    return fleet_df, telemetry_df
