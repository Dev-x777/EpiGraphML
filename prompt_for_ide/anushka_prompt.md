You are my coding agent for "Persona B" on the EpiGraph project. Read the attached
EpiGraph_Project_Spec.md fully before writing any code — it defines the full
architecture and the shared data contract with my teammate's half of the system.

Your scope is ONLY the Persona B modules: rl/, api/, dashboard/.
Do not touch data/, graph/, forecasting/, or rules/ — my teammate owns those in a
separate workspace.

Build in this order:
1. A SEIR epidemic simulator in plain Python/NumPy (no black-box library — I need to
   be able to explain every equation). It should take a resource-allocation action
   (testing kits, vaccine doses, staff per region) and return an outbreak-size-reduction
   reward, matching the allocation_action.json schema in Section 4 of the spec.
2. A contextual bandit (epsilon-greedy or LinUCB — hand-rolled, not a heavy RL library)
   whose state is built from region_risk_output.json (produced by my teammate's Persona A
   pipeline — schema is fixed in Section 4, do not assume a different shape).
   Until real Persona A output exists, test against dummy data matching that exact schema.
3. Evaluate the bandit's cumulative reward against two baselines: random allocation and
   population-proportional allocation. This comparison is required for our evaluation plan.
4. A FastAPI backend exposing endpoints for risk scores, recommended allocation, and
   projected outbreak curves with vs. without intervention.
5. A dashboard (React or Streamlit) showing: a region graph colored by risk (heatmap),
   the recommended allocation plan, and the projected outbreak curve comparison.

After each module, run it and show me real output, not just code, so I can verify it
actually works before moving to the next step. Explain your architecture choices in
plain language as you go, since I need to be able to defend every part of this in a
viva/interview, not just the parts I personally wrote.