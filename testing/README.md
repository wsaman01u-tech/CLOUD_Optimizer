# CloudOptimizer Testing & Documentation Suite

This directory contains test datasets, invalid dataset validation tests, and full architectural documentation for **CloudOptimizer Autopilot**.

---

## 📁 Files Included

### 1. `scenario_1_idle_and_overprovisioned_waste.csv`
* **Scenario:** Heavy cloud waste & idle test/dev instances across AWS, Azure, and GCP.
* **Instances Included:** 10 resources (e.g., `t3.2xlarge`, `Standard_D8s_v3`, `e2-standard-8`) with average CPU < 5% on test machines and < 15% on oversized production machines.
* **Expected App Behavior:**
  - Identifies idle instances with status **Idle** (< 5% CPU).
  - Triggers **Low-Risk "Turn off idle server" / "Stop Instance"** suggestions.
  - Suggests rightsizing oversized production machines to lower tiers.
  - Generates **over $1,000/month** in projected monthly savings.

---

### 2. `scenario_2_production_high_load_and_sla_spikes.csv`
* **Scenario:** High-traffic e-commerce & production database clusters under heavy load.
* **Instances Included:** 9 production servers running databases (Cloud SQL Postgres), Redis caches, payment gateways, and checkout services with CPU between 82% and 91%.
* **Expected App Behavior:**
  - Status badges turn purple/red for **Overloaded (Hot)** (> 80% CPU).
  - Generates **Scale-Up** recommendations to prevent SLA degradation.
  - Testing downsizing in the **Try a change (What-if simulator)** displays the red **"Not safe (Exceeds limit)"** verdict badge, demonstrating how policy guardrails prevent catastrophic production outages.

---

### 3. `scenario_3_cross_cloud_microservices_fleet.json`
* **Scenario:** Modern multi-cloud microservices architecture in **JSON** format.
* **Instances Included:** 8 microservices across AWS, Microsoft Azure, and Google Cloud spanning US, Europe, and Asia regions.
* **Expected App Behavior:**
  - Tests the JSON parsing and ingestion engine.
  - Populates multi-cloud provider badges (AWS, Azure, GCP).
  - Tests dropdown filters: filtering by **AWS**, **Azure**, or **GCP**, and filtering by **Low Risk** vs. **Medium Risk**.

---

### 4. `invalid_non_cloud_data.csv` (The Unsuitable Test File)
* **Dataset Type:** University Student Academic Records & Grades (`Student_ID, Course_Code, Midterm_Score, Final_Score, Attendance_Percentage, Letter_Grade`).
* **Why it fails:** It contains zero cloud infrastructure keywords (no `instance`, `cpu`, `vm`, `cloud`, `cost`, `storage`, or `memory`).
* **Exact Error Thrown in CloudOptimizer:**
  ```
  Modal Title: Invalid Cloud Dataset
  Error Message: "Unrecognized dataset: This file does not appear to contain cloud infrastructure or billing records. Expected columns: resource_id, cloud_provider, cpu_utilization, memory_utilization, or monthly_cost."
  Upload Status Badge: "Rejected" (Red)
  Toast Notification: "Upload Rejected: Could not parse cloud data file."
  ```

---

### 5. Full Application Documentation Files
* **PDF Format:** [`CloudOptimizer_App_Guide_and_Architecture.pdf`](./CloudOptimizer_App_Guide_and_Architecture.pdf) (151 KB)
* **Word Document:** [`CloudOptimizer_App_Guide_and_Architecture.docx`](./CloudOptimizer_App_Guide_and_Architecture.docx) (14.6 KB)
* **Web Viewable:** [`CloudOptimizer_App_Guide_and_Architecture.html`](./CloudOptimizer_App_Guide_and_Architecture.html)

**Contents Covered in the Guide:**
1. High-Level Elevator Pitch
2. Why Traditional Autoscaling Fails vs. CloudOptimizer Autopilot
3. Complete Technology Stack (Vanilla JS, Tailwind, FastAPI, LightGBM, Firebase)
4. The Training Dataset (Microsoft Azure Public VM Trace 2019, 1.37M records, 300 VMs, 16 days)
5. Machine Learning Model Metrics (LightGBM 92.96% accuracy, 0.967% MAE, 7.04% sMAPE, 26.96% net savings, 99.01% SLA compliance, overfitting audit)
6. The 7 Canonical Telemetry Dimensions
7. Optimization Workflow & IaC Automation (Terraform & kubectl)
8. Complete Tour of Features Across Every Screen
9. Presentation Q&A Cheat Sheet (Top 4 questions with winning answers)
