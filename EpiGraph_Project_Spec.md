# EpiGraph — Project Specification & Two-Persona Build Guide

**Team:** Devjeet Saha (RA2311027010048), Anushka Ranjan (RA2311027010016) — CSE-BDA
**Goal:** Disease outbreak prediction & resource allocation using GNN + Ensemble + Association Rules + Contextual RL, validated via SEIR simulation.

This document is the shared source of truth. Both team members must read the whole thing — each of you owns one half, but the two halves only work if you both understand the interfaces in Section 4.

---

## 1. System Overview

```
Raw case/mobility/environment data
        ↓
Region Graph  →  GNN embeddings  ─┐
        ↓                          ├─→  Risk Score per region
Ensemble model (XGBoost/RF) ───────┘
        ↓
Association Rules (Apriori/FP-Growth)  →  Contributing-factor explanations
        ↓
RL Agent (contextual bandit)  →  Resource allocation action
        ↓
SEIR Simulator  →  Reward (outbreak reduction)
        ↓
Dashboard: risk heatmap + allocation plan + projected curves
```

## 2. Tech Stack

- **Data/ML:** Python, PyTorch Geometric (GNN), scikit-learn + XGBoost (ensemble), `mlxtend` (association rules)
- **RL:** Python — simple epsilon-greedy / UCB / LinUCB contextual bandit (no need for deep RL library; hand-rolled is fine and easier to explain)
- **Simulation:** custom SEIR model (a few dozen lines of Python/NumPy — do not use a black-box library, you need to explain it)
- **Backend:** FastAPI
- **Frontend:** React (or Streamlit if time-constrained) — network graph via `react-force-graph` or `vis-network`; charts via Recharts/Plotly
- **Data storage:** CSV/Parquet is enough; no database needed for a prototype

## 3. Repo Structure

```
epigraph/
├── data/
│   ├── raw/                  # downloaded case/mobility/environment data
│   └── processed/            # cleaned, graph-ready data
├── graph/
│   ├── build_graph.py        # region adjacency/mobility graph construction
│   └── gnn_model.py          # GCN/GraphSAGE embedding model
├── forecasting/
│   ├── ensemble_model.py     # XGBoost/RF case-count forecaster
│   └── mlp_baseline.py       # neural network baseline
├── rules/
│   └── association_mining.py # Apriori/FP-Growth on risk-factor co-occurrence
├── rl/
│   ├── seir_simulator.py     # epidemic simulation environment (the "reward function")
│   └── bandit_agent.py       # contextual bandit allocation policy
├── api/
│   └── main.py                # FastAPI endpoints (see Section 4 for contract)
├── dashboard/
│   └── (React or Streamlit app)
├── notebooks/
│   └── eval_and_ablation.ipynb # comparison table: GNN vs ensemble vs MLP; with vs without RL
└── EpiGraph_Project_Spec.md   # this file
```

## 4. Persona Split & Shared Interfaces

### Persona A — Data & Graph-ML Lead
**Owns:** `data/`, `graph/`, `forecasting/`, `rules/`
**Responsibilities:**
- Source and clean district-level case data, mobility/adjacency data, environmental/demographic features
- Build the region graph (nodes = regions, edges = mobility or geographic adjacency)
- Train GNN (GCN/GraphSAGE) to produce per-region embeddings
- Train ensemble (XGBoost/RF) and MLP baseline for case-count forecasting
- Run Apriori/FP-Growth on risk-factor co-occurrence
- **Must deliver to Persona B:** a per-region, per-timestep `risk_score` (float) and `embedding` (vector), in a fixed schema (see below), that Persona B's RL state depends on

### Persona B — RL, Simulation & Systems Lead
**Owns:** `rl/`, `api/`, `dashboard/`
**Responsibilities:**
- Build the SEIR simulator that computes reward (outbreak size reduction) for a given resource-allocation action
- Build the contextual bandit that takes Persona A's risk scores/embeddings as state and outputs an allocation action
- Build the FastAPI backend exposing predictions + recommendations
- Build the dashboard (graph heatmap, allocation plan, projected curves)
- **Depends on Persona A's output schema** — do not start the bandit's state representation until the schema below is fixed and agreed

### Shared Data Contract (agree on this FIRST, before either of you writes model code)

```jsonc
// region_risk_output.json — produced by Persona A, consumed by Persona B
{
  "timestep": "2026-09-01",
  "regions": [
    {
      "region_id": "TN-CHN",
      "risk_score": 0.73,           // ensemble/GNN blended prediction, 0-1
      "predicted_cases": 142,
      "embedding": [0.12, -0.4, ...], // GNN embedding vector, fixed length e.g. 16
      "contributing_factors": ["high_density", "low_vaccination"] // from association rules
    }
  ]
}
```

```jsonc
// allocation_action.json — produced by Persona B's RL agent
{
  "timestep": "2026-09-01",
  "allocations": [
    { "region_id": "TN-CHN", "testing_kits": 500, "vaccine_doses": 1200, "staff": 8 }
  ],
  "expected_reward": 0.31  // from SEIR simulation
}
```

Both of you: **do not change these schemas unilaterally** — if you need a new field, message the other person first, since both sides consume them.

## 5. Milestones (suggested — adjust to your submission calendar)

| Week | Persona A | Persona B |
|---|---|---|
| 1 | Data collection + cleaning; region graph construction | SEIR simulator (standalone, test with dummy risk data) |
| 2 | GNN embeddings working; baseline MLP | Bandit agent skeleton using dummy risk scores |
| 3 | Ensemble model + comparison table (GNN vs ensemble vs MLP) | Wire bandit to real Persona A output; FastAPI skeleton |
| 4 | Association rule mining; contributing-factor explanations | Dashboard v1 (heatmap + allocation plan) |
| 5 | Joint integration + end-to-end test | Joint integration + end-to-end test |
| 6 | Evaluation write-up, ablations, report | Deployment (Docker), demo polish |

## 6. Evaluation Plan

- **Forecasting:** RMSE/MAE on case counts; compare GNN-embedding-based model vs XGBoost/RF vs MLP
- **RL/allocation:** cumulative reward (outbreak reduction) of bandit policy vs. two baselines: (a) random allocation, (b) proportional-to-population allocation
- **Association rules:** report top rules by lift/confidence, sanity-check against known epidemiology (e.g., density + low vaccination should surface)
- **Honesty clause for the report:** state clearly this is a simulation-validated prototype, not a deployed system — see caveat in the proposal doc

## 7. Definition of Done (for approval / demo)

- [ ] End-to-end pipeline runs on real data (not synthetic placeholders) for at least one Indian state/region set
- [ ] Comparison table exists: GNN vs Ensemble vs MLP forecasting accuracy
- [ ] RL policy beats both baselines on simulated cumulative reward
- [ ] Dashboard shows: risk heatmap over graph, recommended allocation, projected outbreak curve with/without intervention
- [ ] Both team members can explain every module, not just their own half
