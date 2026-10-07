"""
export_risk.py -- Build and export region_risk_output.json

Implements two critical fixes over the original version:

  FIX 1 – Risk score:
    Instead of the arbitrary single-variable formula
      risk = predicted_cases / (pop * 0.0001 + 1)
    we now compute a transparent, epidemiologically motivated
    weighted blend of four normalised signals:

      risk_score = w1 * norm_cases
                 + w2 * norm_density
                 + w3 * (1 - vax_rate)        # low vaccination -> higher risk
                 + w4 * (1 - testing_rate)     # low testing -> hidden spread
                 + w5 * gnn_risk_signal        # graph-neighbourhood signal

    Weights (sum to 1):
      w1=0.35  predicted_cases  (primary signal)
      w2=0.20  population density
      w3=0.20  vaccination deficit
      w4=0.15  testing deficit
      w5=0.10  GNN neighbourhood signal (mean embedding magnitude)

  FIX 2 – Contributing factors:
    Instead of applying raw median thresholds and ignoring the surge
    rules entirely, we now:
      (a) binarise each region's risk profile
      (b) check every rule in surge_rules.csv  ->  antecedents ⊆ region profile
      (c) only include factors from rules that ACTUALLY FIRE for that region
      (d) fall back to direct threshold factors if no rules exist
"""

import os
import json
import ast

import numpy as np
import pandas as pd
import xgboost as xgb


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

RISK_WEIGHTS = {
    "predicted_cases": 0.35,
    "density":         0.20,
    "low_vaccination": 0.20,
    "low_testing":     0.15,
    "gnn_signal":      0.10,
}

TIMESTEP = "2026-09-01"  # per shared data contract (Section 4)


# ─────────────────────────────────────────────────────────────
# Helper: normalise a numpy array to [0, 1]
# ─────────────────────────────────────────────────────────────

def minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


# ─────────────────────────────────────────────────────────────
# Helper: compute GNN neighbourhood risk signal
# The embedding was trained via link-prediction loss, so higher
# L2-norm ~= stronger graph signal.  We normalise across regions.
# ─────────────────────────────────────────────────────────────

def gnn_risk_signals(embeddings: dict, node_order: list) -> np.ndarray:
    norms = np.array([
        float(np.linalg.norm(embeddings.get(r_id, [0.0] * 16)))
        for r_id in node_order
    ])
    return minmax(norms)


# ─────────────────────────────────────────────────────────────
# Helper: load and parse association rules
# ─────────────────────────────────────────────────────────────

def load_surge_rules(rules_path: str) -> list[dict]:
    """
    Returns a list of dicts:
      { "antecedents": frozenset, "confidence": float, "lift": float }

    The CSV stores frozensets as their repr string, e.g.:
      "frozenset({'high_density', 'low_vaccination'})"
    We parse them with ast.literal_eval after stripping the outer wrapper.
    """
    if not os.path.exists(rules_path):
        return []

    df = pd.read_csv(rules_path)
    parsed = []
    import re

    for _, row in df.iterrows():
        try:
            raw = str(row["antecedents"])
            # Extract all single-quoted strings from the frozenset repr
            # e.g. "frozenset({'high_density', 'large_population'})"
            factors = re.findall(r"'([^']+)'", raw)
            if not factors:
                continue
            parsed.append({
                "antecedents": frozenset(factors),
                "confidence":  float(row.get("confidence", 0.0)),
                "lift":        float(row.get("lift", 1.0)),
            })
        except Exception:
            pass

    return parsed


# ─────────────────────────────────────────────────────────────
# Helper: get contributing factors for a single region
# using rules that actually fire for its binary profile
# ─────────────────────────────────────────────────────────────

def get_contributing_factors(
    region_profile: frozenset,
    surge_rules: list[dict],
    fallback_factors: list[str],
) -> list[str]:
    """
    Checks each rule: if rule.antecedents ⊆ region_profile -> rule fires.
    Collects all antecedent factors from fired rules (sorted by lift desc).

    Falls back to direct threshold factors when no rules exist or none fire.
    """
    if not surge_rules:
        return fallback_factors

    fired_factors = set()
    fired_rules = [
        r for r in surge_rules
        if r["antecedents"].issubset(region_profile)
    ]

    # Sort by lift so the most informative factors appear first
    fired_rules.sort(key=lambda r: r["lift"], reverse=True)

    for rule in fired_rules:
        fired_factors.update(rule["antecedents"])

    if fired_factors:
        return sorted(fired_factors)

    # No rules fired -> use direct threshold factors as graceful fallback
    return fallback_factors


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def export_data():

    print("=" * 55)
    print("  export_risk.py -- EpiGraphML data export")
    print("=" * 55)

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # ── Load inputs ──────────────────────────────────────────
    demo_path  = os.path.join(PROJECT_ROOT, "data", "processed", "tn_demographics.csv")
    cases_path = os.path.join(PROJECT_ROOT, "data", "processed", "tn_cases.csv")
    emb_path   = os.path.join(PROJECT_ROOT, "data", "processed", "gnn_embeddings.json")
    rules_path = os.path.join(PROJECT_ROOT, "rules", "surge_rules.csv")
    model_path = os.path.join(PROJECT_ROOT, "forecasting", "xgb_model.json")

    df_demo  = pd.read_csv(demo_path)
    df_cases = pd.read_csv(cases_path)

    with open(emb_path) as f:
        embeddings = json.load(f)

    booster = xgb.Booster()
    booster.load_model(model_path)

    surge_rules = load_surge_rules(rules_path)
    print(f"\nLoaded {len(surge_rules)} association rules from {rules_path}")

    # ── Deduplicate demographics (some region_ids appear twice) ──
    feature_cols = ["population", "density_per_sq_km", "testing_rate", "vax_rate"]
    df_demo_agg  = df_demo.groupby("region_id")[feature_cols].mean().reset_index()
    region_ids   = df_demo_agg["region_id"].tolist()

    # ── Pre-compute GNN neighbourhood signal (all regions) ───
    gnn_signals = gnn_risk_signals(embeddings, region_ids)
    gnn_signal_map = dict(zip(region_ids, gnn_signals))

    # ── Binarise demographics for association rule matching ──
    med_density  = df_demo_agg["density_per_sq_km"].median()
    med_pop      = df_demo_agg["population"].median()
    med_testing  = df_demo_agg["testing_rate"].median()
    med_vax      = df_demo_agg["vax_rate"].median()

    # ── Predict cases & compute multi-factor risk score ──────
    all_pred_cases = []
    all_density    = []
    all_low_vax    = []
    all_low_test   = []

    region_rows = []  # collect per-region data before normalisation

    for r_id in region_ids:
        r_cases = df_cases[df_cases["region_id"] == r_id].sort_values("date")["cases"].values
        r_demo  = df_demo_agg[df_demo_agg["region_id"] == r_id].iloc[0]
        r_emb   = embeddings.get(r_id, [0.0] * 16)

        # Predict next-day cases with XGBoost
        lag_features   = r_cases[-7:].tolist() if len(r_cases) >= 7 else [0.0] * 7
        feature_vector = lag_features + [r_demo["population"], r_demo["density_per_sq_km"]] + r_emb
        dmat           = xgb.DMatrix(np.array([feature_vector], dtype=np.float32))
        pred_cases     = max(0.0, float(booster.predict(dmat)[0]))

        all_pred_cases.append(pred_cases)
        all_density.append(r_demo["density_per_sq_km"])
        all_low_vax.append(1.0 - r_demo["vax_rate"])
        all_low_test.append(1.0 - r_demo["testing_rate"])

        region_rows.append({
            "region_id":    r_id,
            "pred_cases":   pred_cases,
            "density":      r_demo["density_per_sq_km"],
            "vax_rate":     r_demo["vax_rate"],
            "testing_rate": r_demo["testing_rate"],
            "embedding":    r_emb,
        })

    # Normalise all signals across regions
    norm_cases   = minmax(np.array(all_pred_cases))
    norm_density = minmax(np.array(all_density))
    norm_low_vax = minmax(np.array(all_low_vax))
    norm_low_test= minmax(np.array(all_low_test))

    # ── Assemble output ──────────────────────────────────────
    regions_output = []

    for i, row in enumerate(region_rows):
        r_id = row["region_id"]

        # Multi-factor weighted risk score
        risk_score = (
            RISK_WEIGHTS["predicted_cases"] * norm_cases[i]
            + RISK_WEIGHTS["density"]         * norm_density[i]
            + RISK_WEIGHTS["low_vaccination"] * norm_low_vax[i]
            + RISK_WEIGHTS["low_testing"]     * norm_low_test[i]
            + RISK_WEIGHTS["gnn_signal"]      * gnn_signal_map[r_id]
        )
        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        # Build binary profile for rule matching
        region_profile = frozenset(filter(None, [
            "high_density"     if row["density"]      > med_density  else "",
            "large_population" if (
                df_demo_agg[df_demo_agg["region_id"] == r_id]["population"].values[0]
                > med_pop
            ) else "",
            "low_vaccination"  if row["vax_rate"]     < med_vax      else "",
            "low_testing"      if row["testing_rate"] < med_testing  else "",
        ]))

        # Fallback: direct threshold factors (used when no rules fire)
        fallback = sorted(list(region_profile))

        contributing_factors = get_contributing_factors(
            region_profile, surge_rules, fallback
        )

        regions_output.append({
            "region_id":            r_id,
            "risk_score":           round(risk_score, 4),
            "predicted_cases":      int(round(row["pred_cases"])),
            "embedding":            [round(e, 4) for e in row["embedding"]],
            "contributing_factors": contributing_factors,
        })

    # ── Write JSON ───────────────────────────────────────────
    output_schema = {
        "timestep": TIMESTEP,
        "regions":  regions_output,
    }

    out_path = os.path.join(PROJECT_ROOT, "region_risk_output.json")
    with open(out_path, "w") as f:
        json.dump(output_schema, f, indent=2)

    # ── Summary ──────────────────────────────────────────────
    risk_scores = [r["risk_score"] for r in regions_output]
    print(f"\nExported {len(regions_output)} regions -> {out_path}")
    print(f"\nRisk score stats:")
    print(f"  min   : {min(risk_scores):.4f}")
    print(f"  max   : {max(risk_scores):.4f}")
    print(f"  mean  : {np.mean(risk_scores):.4f}")
    print(f"  median: {np.median(risk_scores):.4f}")

    high_risk = [r for r in regions_output if r["risk_score"] >= 0.7]
    med_risk  = [r for r in regions_output if 0.4 <= r["risk_score"] < 0.7]
    low_risk  = [r for r in regions_output if r["risk_score"] < 0.4]

    print(f"\nRisk tiers:")
    print(f"  High (>=0.70) : {len(high_risk)} regions")
    print(f"  Med  (0.40–0.70): {len(med_risk)} regions")
    print(f"  Low  (<0.40) : {len(low_risk)} regions")
    print("\nSample (top 5 by risk):")
    for r in sorted(regions_output, key=lambda x: x["risk_score"], reverse=True)[:5]:
        print(f"  {r['region_id']:8s}  risk={r['risk_score']:.4f}  "
              f"cases={r['predicted_cases']:4d}  "
              f"factors={r['contributing_factors']}")
    print("=" * 55)


if __name__ == "__main__":
    export_data()
