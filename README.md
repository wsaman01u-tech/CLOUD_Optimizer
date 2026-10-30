# Autonomous Cloud Cost & Performance Optimizer

[![CI](https://github.com/team-xoxo/cloud-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/team-xoxo/cloud-optimizer/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![No Streamlit](https://img.shields.io/badge/UI-Vanilla%20JS%20%2B%20Chart.js-indigo.svg)](https://www.chartjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent multi-agent reinforcement learning system designed to eliminate cloud inefficiencies through telemetry monitoring, proactive demand forecasting, Q-learning autoscaling, and safe automated infrastructure delivery (Terraform & Kubernetes) with natural language audit explanations.

> **Architecture Rule**: In strict compliance with project guidelines, **Streamlit and Gradio are strictly prohibited**. The entire dashboard is engineered with a high-performance **FastAPI backend** and an ultra-responsive **HTML5 / JavaScript / Tailwind CSS / Chart.js** client.

---

## 1. Problem Statement

Modern enterprise cloud infrastructures suffer severe financial and operational waste caused by:
1. **Over-provisioned compute and storage**: Provisioning for rare peak spikes, leaving servers running at 10–25% average utilization.
2. **Idle services running 24/7**: Zombie dev/test environments left active during nights and weekends.
3. **Lack of demand forecasting**: Reactive autoscaling triggers only *after* traffic surges, leading to SLA latency breaches or crash-looping backends.
4. **Manual & error-prone rightsizing**: DevOps engineers manually adjusting VM instance sizes based on guesswork.
5. **Skyrocketing cloud bills without performance gains**: Overspending on instances that do not tangibly improve customer response times.

---

## 2. Multi-Agent System Architecture

The solution coordinates **five specialized autonomous agents** interacting in a closed-loop control system:

```
+-------------------------------------------------------------------------------+
|                      CLOUDWATCH / VM TELEMETRY METRICS                        |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|                   1. DATA & USAGE MONITORING AGENT                            |
|  - Flags Idle VMs (CPU < 10% for >90% of time)                               |
|  - Identifies Over-provisioned VMs (Peak CPU < 40%)                           |
|  - Quantifies total fleet financial waste                                     |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|                      2. DEMAND FORECASTING AGENT                              |
|  - 24-Hour Lag Supervised Machine Learning (Random Forest Regressor)         |
|  - Benchmarked against 3h Moving Average & 24h Seasonal Naive baselines       |
|  - Predicts near-future workload demand with high accuracy                    |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|                 3. CLOUD SIMULATOR & RL OPTIMIZATION AGENT                    |
|  - Tabular Q-Learning in discrete state space: (Load, Forecast, Replicas)     |
|  - Reward = -Cost - (SLA Penalty * Violation)                                 |
|  - Anticipates demand spikes ahead of time and safely sheds idle capacity     |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|                         4. DEPLOYMENT AGENT                                   |
|  - Enforces safety boundaries: 1 <= Replicas <= 10                            |
|  - Rate-limits scaling deltas: Maximum +/- 1 replica change per timestep      |
|  - Generates Terraform HCL (variables.tf, terraform.tfvars) and kubectl CLI   |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|                   5. AUDIT & EXPLAINABILITY AGENT                             |
|  - Immutable audit trail recording every scaling event                        |
|  - Plain-English natural language rationale for DevOps compliance & trust     |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|                  FASTAPI REST BACKEND & CHART.JS DASHBOARD                    |
+-------------------------------------------------------------------------------+
```

---

## 3. Key Results & Benchmark Comparison

Evaluated on unseen test workloads over 100+ hourly operational cycles:

| Scaling Strategy | Total Cost ($) | SLA Violations | SLA Compliance (%) | Avg Utilization (%) | Cost Reduction (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Static Allocation (Fixed 8 VMs)** | $808.00 | 0 | 100.0% | 41.2% | Baseline (0.0%) |
| **Reactive Threshold (30% / 70%)** | $954.00 | 1 | 99.01% | 38.6% | -18.07% (More costly) |
| **Autonomous RL Optimizer (Ours)** | **$632.00** | **1** | **99.01%** | **59.4%** | **+21.78% Saved** |

### Demand Forecaster Error Benchmarks

| Model Architecture | RMSE | MAE | MAPE (%) | Beats Baseline? |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Regressor (ML)** | **33.57** | **23.12** | **4.85%** | **YES (Winner)** |
| Seasonal Naive (24h shift) | 45.02 | 31.40 | 6.51% | Yes |
| Moving Average (3h rolling) | 90.72 | 67.80 | 15.21% | Baseline |
| Holt-Winters Exponential Smoothing | 135.13 | 98.40 | 18.36% | No |

---

## 4. Project Directory Structure

```
.
├── agents/
│   ├── __init__.py
│   ├── monitor.py          # Agent 1: Idle & over-provisioning detection
│   ├── forecast.py         # Agent 2: ML & statistical demand forecasting
│   ├── optimizer.py        # Agent 3: Q-learning RL policy & benchmarks
│   ├── deploy.py           # Agent 4: Terraform & Kubernetes IaC generator
│   └── audit.py            # Agent 5: Plain-English explainability engine
├── simulator/
│   ├── __init__.py
│   └── env.py              # Cloud autoscaling simulation environment
├── api/
│   ├── __init__.py
│   └── main.py             # FastAPI REST service & static web mount
├── frontend/
│   └── index.html          # Professional web dashboard (Chart.js + Tailwind)
├── data/
│   ├── raw/                # Workload traces telemetry
│   └── results/            # Agent outputs (CSV, JSON, Q-table)
├── outputs/
│   ├── plots/              # Visual charts (PNG)
│   └── terraform/          # Generated variables.tf, terraform.tfvars, k8s YAML
├── tests/
│   ├── __init__.py
│   └── test_agents.py      # Automated pytest suite (14 passing tests)
├── run_pipeline.py         # Master pipeline runner
├── requirements.txt        # Full development dependencies
├── requirements-prod.txt   # Lean container runtime dependencies
├── Dockerfile              # Production Docker container definition
├── .dockerignore
├── .env.example
├── .github/workflows/ci.yml
└── README.md
```

---

## 5. Quickstart & Installation

### Step 1: Clone and Set Up Virtual Environment

```bash
git clone https://github.com/team-xoxo/cloud-optimizer.git
cd cloud-optimizer

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the End-to-End Pipeline

Execute the master multi-agent pipeline to generate telemetry, train models, benchmark strategies, and output IaC manifests:

```bash
python run_pipeline.py
```

### Step 4: Run the Web Dashboard & API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser to:
- **Interactive Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 6. Running Tests

Run the complete automated test suite verifying simulator dynamics, safety clamps, idle heuristics, forecasting error formulas, and REST endpoints:

```bash
pytest -v
```

---

## 7. Docker Deployment

Build and run the container locally:

```bash
# Build the Docker image
docker build -t cloud-optimizer-api .

# Run the container
docker run -p 8000:8000 cloud-optimizer-api
```

Access the live dashboard at `http://localhost:8000`.

---

## 8. Viva & Examination Defense Q&A

### Q1: Why Reinforcement Learning and not simple threshold rules?
> **Answer**: Simple reactive rules (e.g., scale up if CPU > 70%) only react *after* the surge occurs. In cloud infrastructure, VM provisioning and container cold starts take 1 to 5 minutes, leading to immediate latency degradation and SLA breaches. Our RL agent integrates forecasted demand into its state space, allowing it to proactively scale *before* traffic peaks arrive and safely scale down during troughs without thrashing.

### Q2: How did you avoid hurting performance while cutting costs?
> **Answer**: The agent's reward function penalizes SLA violations with an asymmetric heavy penalty:
> $$\text{Reward} = -\text{Cost} - (\text{SLA\_Penalty} \times \text{Violation})$$
> Because SLA violations cost an order of magnitude more than maintaining an extra server, the agent prioritizes meeting demand. This achieved **99.01% SLA compliance** while reducing overall cloud spend by **21.78%**.

### Q3: How do you verify the forecast is accurate? Which baseline did you beat?
> **Answer**: We benchmarked our Random Forest regressor with 24h lag features against two industry baselines: a 3-hour Rolling Moving Average and a 24-hour Seasonal Naive model. Our ML model achieved an **RMSE of 33.57 and MAPE of 4.85%**, beating the 3-hour Moving Average baseline (RMSE: 90.72) by over 63%.

### Q4: Why test in a simulator? What would change on a real cloud?
> **Answer**: Training RL directly on a live production cloud cluster would cause expensive unnecessary instances, outages, and hundreds of dollars in test API charges. The simulator provides a rapid, cost-free sandbox. In a real cloud environment, the only changes would be connecting the Deployment Agent to real AWS Auto Scaling Groups or Kubernetes HPA APIs and accounting for real VM boot latency (approx. 60–180 seconds).

### Q5: How does this system differ from AWS Compute Optimizer?
> **Answer**: AWS Compute Optimizer is an offline recommendation tool that inspects past CloudWatch logs and recommends instance sizes once every few days. Our system is an **autonomous, closed-loop runtime engine** that pairs proactive time-series forecasting with reinforcement learning to take real-time scaling actions, generate Terraform IaC, and produce plain-English audit justifications.

### Q6: What are the risks of letting an agent change infrastructure automatically?
> **Answer**: Runaway scaling or thrashing (rapidly adding and deleting servers) could destabilize clusters and rack up massive bills. To eliminate this risk, our **Deployment Agent enforces hard architectural invariants**:
> 1. Absolute server count bounds ($1 \le N \le 10$).
> 2. Rate-limiting delta clamping: Maximum of $\pm 1$ instance change per timestep.
> 3. Dry-run mode by default, requiring audit verification before live execution.

---

## 9. Team & Acknowledgments

- **Project**: Autonomous Cloud Cost & Performance Optimizer
- **Team**: Team XOXO (T-09)
- **Academic Mentor**: Dept. of Computer Science & Engineering
- **Version**: 1.0.0 (Production Release)
