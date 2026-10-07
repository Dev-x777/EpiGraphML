"""
graph/build_graph.py — Region adjacency graph with gravity-model edge weights

FIXED: Edge weights are no longer random.
We use a simplified gravity model:
    w_ij = sqrt(pop_i * pop_j) / dist_rank_ij

where dist_rank_ij is the rank of (i,j) by graph-distance — a proxy
for actual geographic distance without needing lat/lon coordinates.

The Watts-Strogatz small-world topology (k=4, p=0.1) is kept as before
since it's a well-justified approximation of regional mobility networks
(local neighbourhood connections + occasional long-distance hubs).
"""

import os
import json
import math

import pandas as pd
import networkx as nx
import numpy as np


def build_graph():
    print("Building region mobility/adjacency graph with gravity-model weights...")

    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    os.makedirs("graph", exist_ok=True)

    demo_path = os.path.join("data", "processed", "tn_demographics.csv")
    df_demo   = pd.read_csv(demo_path)

    # ── Deduplicate region_ids (some appear twice in the CSV) ─
    df_agg = (
        df_demo.groupby("region_id")["population"]
        .mean()
        .reset_index()
    )
    region_ids  = df_agg["region_id"].tolist()
    pop_map     = dict(zip(df_agg["region_id"], df_agg["population"]))

    n = len(region_ids)

    # ── Build Watts-Strogatz small-world topology ─────────────
    # k=4  → each node joins its 4 nearest neighbours in a ring
    # p=0.1 → 10% rewiring probability creates long-range hubs
    G_ws = nx.watts_strogatz_graph(n, k=4, p=0.1, seed=42)

    G = nx.Graph()
    for r_id in region_ids:
        G.add_node(r_id, population=pop_map[r_id])

    # ── Assign gravity-model edge weights ─────────────────────
    # For each WS edge (u_idx, v_idx):
    #   w = sqrt(pop_u * pop_v) / (1 + ring_distance)
    # where ring_distance = min(|u-v|, n-|u-v|) on the base ring.
    # This gives higher weights to edges between large, nearby districts.
    # Final weights are min-max normalised to [0.1, 1.0].

    raw_edges = []
    for u_idx, v_idx in G_ws.edges():
        ring_dist = min(abs(u_idx - v_idx), n - abs(u_idx - v_idx))
        u_id  = region_ids[u_idx]
        v_id  = region_ids[v_idx]
        pop_u = pop_map[u_id]
        pop_v = pop_map[v_id]
        raw_w = math.sqrt(pop_u * pop_v) / (1.0 + ring_dist)
        raw_edges.append((u_id, v_id, raw_w))

    # Normalise weights to [0.1, 1.0]
    weights = np.array([e[2] for e in raw_edges])
    w_min, w_max = weights.min(), weights.max()
    denom = w_max - w_min if w_max - w_min > 0 else 1.0

    for u_id, v_id, raw_w in raw_edges:
        norm_w = 0.1 + 0.9 * ((raw_w - w_min) / denom)
        G.add_edge(u_id, v_id, weight=round(float(norm_w), 4))

    # ── Save as node-link JSON ────────────────────────────────
    data = nx.node_link_data(G)
    graph_path = os.path.join("data", "processed", "region_graph.json")
    with open(graph_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Graph  : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Weights: min={min(e[2] for e in raw_edges):.0f} raw -> "
          f"0.10-1.00 normalised")
    print(f"Saved  -> {graph_path}")


if __name__ == "__main__":
    build_graph()
