# CloudOptimizer — Autonomous Cloud Cost & Performance Autopilot

An intelligent cloud cost and performance optimization system designed so that anyone with zero prior cloud knowledge can review, forecast, simulate, and apply safe rightsizing recommendations in minutes.

The system continuously watches cloud usage, forecasts 7-day demand with uncertainty bands, recommends cost-saving rightsizing changes, and safely applies them after operator approval with one-click instant undo.

---

## 1. Key Features

1. **Home ("Your savings at a glance")**
   - Three big numbers: current monthly cost, possible monthly savings, and things needing attention.
   - Plain-English executive sentence: *"You could save about $X per month. Y items are safe to fix now."*
   - Direct *"Review suggestions"* primary action.
   - Clear sample data disclaimer banner: *"Demo data from a public Azure trace sample. Costs are assumed."*

2. **Guided Start (3-step setup tour)**
   - **Step 1**: Choose data (*Use demo data* with Azure cloud sample or *Upload custom CSV*).
   - **Step 2**: Choose a style (*Careful*, *Balanced*, or *Aggressive* presets that configure all safety thresholds).
   - **Step 3**: Choose automation mode (*Suggest only* or *Auto-fix safe items*).
   - Skippable and re-openable at any time from the top navigation bar.

3. **Suggestions (The core page)**
   - Plain-language recommendation cards: e.g. *"Server web-3 uses 12% of its power. Switch to a smaller size and save about $X per month."*
   - Risk badges (*Low*, *Medium*, *High*) and confidence indicators (e.g. *96% confident*).
   - **"Why?"** button expanding real computed evidence: observed usage stats, 7-day projected demand, and the safety rule that fired.
   - *Approve*, *Reject*, and one-click *"Approve all low-risk"* actions.
   - Filter bar: *Saves most*, *Lowest risk*, *Production vs Test*.

4. **My Servers**
   - Inventory list with clear color-coded statuses: **Idle**, **Too big**, **Healthy**, **Overloaded**.
   - Inspect any server to view its usage history and a **7-day forecast chart with uncertainty bands** (upper/lower bounds) and plain-English peak hour insights (e.g. *"Expected busiest: Monday 2:00 PM (~84% CPU)"*).

5. **Try a Change (What-if simulator)**
   - Pick a server and a target instance size.
   - Live before-and-after comparison showing new monthly cost, predicted peak CPU, and a clear green/red **"Safe / Not safe"** verdict before anything changes in real life.

6. **Apply and Undo**
   - *"Apply approved changes"* button with confirmation safeguards.
   - Plain-language summary of what was executed.
   - Automated post-change health verification check (latency and error rate monitoring).
   - One-click **"Undo recent changes"** button providing instant rollback to the previous server size and cost.

7. **Settings & Rules**
   - Simple view: style preset (*Careful* / *Balanced* / *Aggressive*), mode (*Suggest only* / *Auto-fix safe items*), and monthly budget.
   - Collapsible **Advanced** section: safe CPU limit (default 75%), max changes per run, minimum capacity, maintenance window, and *Protect production* toggle.
   - All settings dynamically saved and loaded from `rules.yaml` — never hard-coded.

8. **History & Audit**
   - Immutable audit trail recording every event: recommendations, approvals, rejections, applications, and rollbacks with who/when/why and verification status.
   - One-click *Export CSV* download for auditing and FinOps reporting.

---

## 2. Architecture & Design Principles

- **No Prohibited UI Frameworks**: Streamlit, Gradio, Dash, Panel, NiceGUI, and Grafana are strictly prohibited. The application is built with a high-performance **FastAPI backend** and a lightweight, responsive **HTML5 / JavaScript / Tailwind CSS / Chart.js** client.
- **Zero Jargon**: Non-technical phrasing across all primary views with `?` hover pills for technical concepts (CPU, rightsizing, p95, vCPU, safe limits).
- **Safety First**:
  - `protect_production=true`: Production resources are never terminated or downsized without explicit operator confirmation.
  - Safe CPU limit: Rightsizing is only recommended if predicted peak load remains $\le 75\%$.
  - Rate limiting: Maximum changes per execution cycle are capped.
  - Idempotent execution: Re-running actions does not duplicate deductions or state changes.
  - Instant Undo: Every change can be rolled back with one click.

---

## 3. Directory Structure

```
.
├── agents/
│   ├── cloud_optimizer.py   # Core engine: rules.yaml loader, recommendations, 7-day forecast, rollback manager
│   ├── monitor.py           # Monitoring agent: idle & over-provisioning heuristics
│   ├── forecast.py          # Machine learning demand forecasting
│   ├── optimizer.py         # Tabular Q-learning policy & benchmarks
│   ├── deploy.py            # Terraform and Kubernetes IaC generation
│   └── audit.py             # Rule-based explainability engine
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI REST API & static web mount
├── frontend/
│   ├── index.html           # 8-screen web application (Tailwind CSS + Chart.js)
│   ├── favicon.svg          # Brand icon
│   └── logo.svg             # Application logo
├── rules.yaml               # Configurable policy rules & instance pricing catalog
├── data/
│   ├── raw/                 # Workload telemetry traces
│   └── results/             # Stored evaluation metrics and audit logs
├── outputs/
│   ├── plots/               # Evaluation comparison plots
│   └── terraform/           # Generated Terraform & Kubernetes deployment files
├── tests/
│   ├── test_optimizer_rules.py  # Tests for missing data, insufficient history, policy blocks, idempotency, audit, undo
│   └── test_agents.py           # Agent and simulator unit tests
├── requirements.txt         # Development and runtime dependencies
└── README.md
```

---

## 4. How to Run the Application

### Step 1: Install Dependencies

```bash
# Activate existing virtual environment or create one
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Step 2: Start the Web App & API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser to:
- **Web App**: [http://localhost:8000](http://localhost:8000)
- **Interactive API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 5. Running the Automated Tests

Run the complete test suite verifying guardrails, missing data handling, production protection, idempotency, audit logging, and one-click rollback:

```bash
pytest -v
```

All 22 test cases will run and pass:
- `tests/test_optimizer_rules.py` (7 tests):
  1. `test_missing_data_handled_gracefully`: Records with missing/corrupted metrics are skipped without crashing.
  2. `test_insufficient_history_prevents_action`: Instances with history fewer than 24 observations produce no automated recommendations.
  3. `test_policy_blocks_production_termination_without_approval`: `protect_production=true` prevents production terminations from auto-applying.
  4. `test_unapproved_change_not_applied_in_suggest_mode`: In *Suggest only* mode, pending recommendations cannot be applied until approved.
  5. `test_idempotent_application_no_duplicate_runs`: Applying an already applied recommendation returns idempotently without repeating cost deduction.
  6. `test_audit_entry_created_for_actions`: Every lifecycle action logs a tamper-evident audit entry.
  7. `test_undo_works_and_reverts_state`: Applying an approved change and calling Undo restores original server size and monthly cost.
- `tests/test_agents.py` (15 tests):
  - Agent heuristics, forecasting error formulas, simulator boundaries, and REST API integration.
