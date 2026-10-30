"""Custom Cloud Dataset Generator for Testing Models and Optimizer.
Generates realistic multi-cloud fleets for testing:
1. enterprise_waste_fleet.csv: High cost-saving opportunity (idle & over-provisioned VMs)
2. production_spiky_fleet.xlsx: Strict production environments with diurnal peaks
3. multi_cloud_enterprise_fleet.json: 50-resource diverse fleet across AWS, Azure, GCP
"""
import os
import io
import json
import zipfile
import numpy as np
import pandas as pd

np.random.seed(42)

def generate_fleet(num_resources=30, waste_ratio=0.5):
    providers = ["AWS", "Azure", "GCP"]
    regions = {
        "AWS": ["us-east-1", "us-west-2", "eu-central-1", "ap-southeast-1"],
        "Azure": ["eastus", "westeurope", "southeastasia", "centralus"],
        "GCP": ["us-central1", "europe-west1", "asia-east1", "us-east4"]
    }
    sizes = {
        "AWS": [("t3.nano", 3.74, 2, 0.5), ("t3.micro", 7.49, 2, 1), ("t3.medium", 29.95, 2, 4), 
                ("t3.large", 59.90, 2, 8), ("t3.xlarge", 119.81, 4, 16), ("c5.2xlarge", 248.20, 8, 16),
                ("r5.2xlarge", 367.92, 8, 64)],
        "Azure": [("Standard_B1s", 15.18, 1, 1), ("Standard_B2s", 30.36, 2, 4), 
                  ("Standard_D2s_v3", 70.08, 2, 8), ("Standard_D4s_v3", 140.16, 4, 16),
                  ("Standard_D8s_v3", 280.32, 8, 32), ("Standard_E4s_v3", 188.00, 4, 32)],
        "GCP": [("e2-micro", 6.10, 2, 1), ("e2-small", 14.60, 2, 2), ("e2-medium", 24.50, 2, 4),
                ("e2-standard-2", 48.80, 2, 8), ("e2-standard-4", 97.60, 4, 16), 
                ("e2-standard-8", 195.20, 8, 32), ("n2-standard-4", 144.54, 4, 16)]
    }

    records = []
    for i in range(1, num_resources + 1):
        prov = np.random.choice(providers)
        reg = np.random.choice(regions[prov])
        size_tuple = sizes[prov][np.random.randint(0, len(sizes[prov]))]
        size_name, cost, vcpu, ram_gb = size_tuple

        is_waste = np.random.rand() < waste_ratio
        if is_waste:
            # idle or overprovisioned
            if np.random.rand() < 0.4:
                avg_cpu = round(float(np.random.uniform(0.3, 4.2)), 1)
                peak_cpu = round(float(avg_cpu + np.random.uniform(0.5, 3.0)), 1)
                mem = round(float(np.random.uniform(5.0, 15.0)), 1)
                status = "idle"
                env = "test" if np.random.rand() < 0.6 else "production"
                name_prefix = "test-worker" if env == "test" else "batch-daemon"
            else:
                avg_cpu = round(float(np.random.uniform(6.0, 18.0)), 1)
                peak_cpu = round(float(avg_cpu + np.random.uniform(4.0, 12.0)), 1)
                mem = round(float(np.random.uniform(15.0, 35.0)), 1)
                status = "over_provisioned"
                env = "production" if np.random.rand() < 0.7 else "test"
                name_prefix = "web-api" if env == "production" else "dev-box"
        else:
            # healthy or hot
            if np.random.rand() < 0.25:
                avg_cpu = round(float(np.random.uniform(75.0, 92.0)), 1)
                peak_cpu = round(float(min(100.0, avg_cpu + np.random.uniform(5.0, 12.0))), 1)
                mem = round(float(np.random.uniform(70.0, 95.0)), 1)
                status = "hot"
                env = "production"
                name_prefix = "db-primary"
            else:
                avg_cpu = round(float(np.random.uniform(35.0, 65.0)), 1)
                peak_cpu = round(float(min(100.0, avg_cpu + np.random.uniform(8.0, 20.0))), 1)
                mem = round(float(np.random.uniform(40.0, 68.0)), 1)
                status = "healthy"
                env = "production" if np.random.rand() < 0.8 else "staging"
                name_prefix = "k8s-node"

        rec = {
            "Resource ID": f"{prov.lower()}-{name_prefix}-{i:02d}",
            "Resource Name": f"{prov} {name_prefix.replace('-', ' ').title()} #{i}",
            "Cloud Provider": prov,
            "Region": reg,
            "Instance Type": size_name,
            "vCPU": vcpu,
            "RAM (GB)": ram_gb,
            "CPU Utilization (%)": f"{avg_cpu}%",
            "Peak CPU (%)": f"{peak_cpu}%",
            "Memory Utilization (%)": f"{mem}%",
            "Storage (GB)": int(np.random.choice([50, 100, 250, 500, 1000])),
            "Monthly Cost ($)": f"${cost:.2f}",
            "Status": status,
            "Environment": env
        }
        records.append(rec)
    return pd.DataFrame(records)

def write_xlsx(df, path):
    cols = list(df.columns)
    shared_strings = []
    string_to_id = {}
    def get_str_id(s):
        s = str(s)
        if s not in string_to_id:
            string_to_id[s] = len(shared_strings)
            shared_strings.append(s)
        return string_to_id[s]

    row_xmls = []
    hdr_cells = [f'<c t="s"><v>{get_str_id(c)}</v></c>' for c in cols]
    row_xmls.append(f'<row r="1">{"".join(hdr_cells)}</row>')

    for r_idx, (_, row) in enumerate(df.iterrows(), 2):
        c_cells = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                c_cells.append('<c/>')
            elif isinstance(val, (int, float)):
                c_cells.append(f'<c><v>{val}</v></c>')
            else:
                s_id = get_str_id(str(val))
                c_cells.append(f'<c t="s"><v>{s_id}</v></c>')
        row_xmls.append(f'<row r="{r_idx}">{"".join(c_cells)}</row>')

    sheet_xml = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{"".join(row_xmls)}</sheetData></worksheet>'
    sst_items = "".join([f'<si><t>{s}</t></si>' for s in shared_strings])
    sst_xml = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{len(shared_strings)}" uniqueCount="{len(shared_strings)}">{sst_items}</sst>'
    content_types = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>'
    rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
    wb = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="CloudFleet" sheetId="1" r:id="rId1"/></sheets></workbook>'
    wb_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/></Relationships>'

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", wb)
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        zf.writestr("xl/sharedStrings.xml", sst_xml)

# 1. Enterprise Waste Fleet (CSV) - 25 servers, 70% idle or oversized
waste_df = generate_fleet(num_resources=25, waste_ratio=0.7)
waste_df.to_csv("data/samples/enterprise_waste_fleet.csv", index=False)
waste_df.to_csv("frontend/samples/enterprise_waste_fleet.csv", index=False)

# 2. Production Spiky Fleet (XLSX) - 30 servers, mostly production with safety constraints
prod_df = generate_fleet(num_resources=30, waste_ratio=0.35)
write_xlsx(prod_df, "data/samples/production_spiky_fleet.xlsx")
write_xlsx(prod_df, "frontend/samples/production_spiky_fleet.xlsx")

# 3. Multi-Cloud Enterprise Fleet (JSON) - 50 servers spanning AWS, Azure, GCP
enterprise_df = generate_fleet(num_resources=50, waste_ratio=0.5)
enterprise_df.to_json("data/samples/multi_cloud_enterprise_fleet.json", orient="records", indent=2)
enterprise_df.to_json("frontend/samples/multi_cloud_enterprise_fleet.json", orient="records", indent=2)

print("Generated test datasets successfully:")
print("1. data/samples/enterprise_waste_fleet.csv (25 servers, high waste)")
print("2. data/samples/production_spiky_fleet.xlsx (30 servers, production)")
print("3. data/samples/multi_cloud_enterprise_fleet.json (50 servers, diverse)")
