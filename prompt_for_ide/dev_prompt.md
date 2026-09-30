# EpiGraph — Persona A Build Prompt

You are my coding agent for **Persona A** on the EpiGraph project. Read the attached `EpiGraph_Project_Spec.md` fully before writing any code — it defines the full architecture and the shared data contract with my teammate's half of the system.

### Scope
Your scope is **ONLY** the Persona A modules:
- `data/`
- `graph/`
- `forecasting/`
- `rules/`

Do **NOT** touch `rl/`, `api/`, or `dashboard/` — my teammate owns those in a separate workspace.

### Build Order
Build in this order:

1. **Data Collection & Cleaning**: Scripts for district-level case data (e.g., `data.gov.in` / `covid19india.org` archives or NVBDCP dengue data) and a region adjacency/mobility graph.
2. **GNN Embeddings**: A Graph Neural Network (GCN or GraphSAGE via PyTorch Geometric) producing per-region embeddings.
3. **Forecasting Models**: 
   - An ensemble forecaster (XGBoost/RandomForest)
   - An MLP baseline for case-count prediction
   - *Requirement*: Build a comparison notebook evaluating both against the GNN-embedding-based model.
4. **Association Rule Mining**: (Apriori/FP-Growth via `mlxtend`) on environmental/demographic co-occurrence patterns linked to outbreak surges.
5. **Data Export**: A final export script that writes `region_risk_output.json` in **EXACTLY** the schema defined in Section 4 of the spec file — this is what my teammate's RL agent consumes. Do not change that schema without flagging it to me first.

### Constraints & Validation
- **Real Data Only**: Use real, publicly available data — no synthetic placeholders in the final version.
- **Incremental Validation**: After each module, run it and show me real output (not just code) so I can verify it actually works before moving to the next step.
- **Explainability**: Explain your architecture choices in plain language as you go, since I need to be able to defend every part of this in a viva/interview, not just the parts I personally wrote.
