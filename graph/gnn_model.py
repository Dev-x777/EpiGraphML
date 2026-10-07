"""
graph/gnn_model.py -- Real GraphSAGE GNN using PyTorch Geometric

GraphSAGE (Hamilton et al., 2017) learns per-region embeddings by
aggregating each node's own features with its neighbourhood's features.
This is the core graph-learning step of EpiGraphML.

Node features per region:
  [population, density_per_sq_km, testing_rate, vax_rate]  (4 dims)

Output:
  16-dim embedding per region -> saved to data/processed/gnn_embeddings.json
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress Python 3.14 torch.jit.script FutureWarning -- harmless
warnings.filterwarnings("ignore", category=FutureWarning)

from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


# ─────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────

class GraphSAGE(nn.Module):
    """
    Two-layer GraphSAGE encoder.

    Layer 1: input_dim -> hidden_dim  (ReLU activation)
    Layer 2: hidden_dim -> embed_dim  (linear, no activation -- raw embedding)

    SAGEConv uses mean aggregation by default.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, embed_dim: int = 16):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x  # shape: [num_nodes, embed_dim]


# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────

def load_node_features(df_demo: pd.DataFrame, node_order: list) -> torch.Tensor:
    """
    Build node feature matrix X of shape [num_nodes, 4] in the order
    given by node_order (list of region_id strings).

    Features: population, density_per_sq_km, testing_rate, vax_rate
    All features are min-max normalised to [0, 1] so the GNN trains stably.
    """
    feature_cols = ["population", "density_per_sq_km", "testing_rate", "vax_rate"]

    # Some region_ids appear more than once in the CSV (duplicate district codes).
    # We average duplicates so every node_id maps to exactly one feature row.
    df_agg = df_demo.groupby("region_id")[feature_cols].mean().reset_index()
    df_indexed = df_agg.set_index("region_id")

    rows = []
    for nid in node_order:
        if nid in df_indexed.index:
            rows.append(df_indexed.loc[nid, feature_cols].values.astype(float))
        else:
            rows.append(np.zeros(len(feature_cols)))

    X = np.array(rows, dtype=np.float32)

    # Min-max normalise each feature column
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    denom = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
    X = (X - col_min) / denom

    return torch.tensor(X)


def load_edge_index(graph_data: dict, node_to_idx: dict) -> torch.Tensor:
    """
    Convert networkx node-link JSON to a PyG edge_index tensor of shape [2, num_edges].
    We use undirected edges so each (u, v) becomes both (u->v) and (v->u).
    """
    links = graph_data.get("links", graph_data.get("edges", []))
    src, dst = [], []

    for link in links:
        u = node_to_idx.get(link["source"])
        v = node_to_idx.get(link["target"])
        if u is not None and v is not None:
            src.extend([u, v])
            dst.extend([v, u])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


# ─────────────────────────────────────────────
# Self-supervised training objective
# ─────────────────────────────────────────────

def reconstruction_loss(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    num_neg: int = 50,
) -> torch.Tensor:
    """
    Link-prediction loss (graph auto-encoder style):
      - Positive pairs  = actual edges  -> push dot-product high (sigmoid -> 1)
      - Negative pairs  = random non-edges -> push dot-product low (sigmoid -> 0)

    This is a standard self-supervised signal for GNN encoders when
    we don't have external node-level labels.
    """
    num_nodes = embeddings.size(0)

    # Positive pairs
    src_pos = edge_index[0]
    dst_pos = edge_index[1]
    pos_scores = (embeddings[src_pos] * embeddings[dst_pos]).sum(dim=1)
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_scores, torch.ones_like(pos_scores)
    )

    # Negative pairs (random sampling)
    neg_src = torch.randint(0, num_nodes, (num_neg,))
    neg_dst = torch.randint(0, num_nodes, (num_neg,))
    neg_scores = (embeddings[neg_src] * embeddings[neg_dst]).sum(dim=1)
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_scores, torch.zeros_like(neg_scores)
    )

    return pos_loss + neg_loss


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def generate_embeddings(epochs: int = 200, lr: float = 0.01, seed: int = 42):
    """
    Train a two-layer GraphSAGE on the region graph and save
    16-dimensional embeddings to data/processed/gnn_embeddings.json.

    Args:
        epochs  : number of training iterations (default 200)
        lr      : Adam learning rate (default 0.01)
        seed    : random seed for reproducibility
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 55)
    print("  GraphSAGE GNN -- EpiGraphML")
    print("=" * 55)

    # ── Load graph ──────────────────────────────────────────
    graph_path = os.path.join(PROJECT_ROOT, "data", "processed", "region_graph.json")
    with open(graph_path) as f:
        graph_data = json.load(f)

    node_order = [n["id"] for n in graph_data["nodes"]]
    node_to_idx = {nid: i for i, nid in enumerate(node_order)}
    num_nodes = len(node_order)
    print(f"\nGraph  : {num_nodes} nodes")

    links = graph_data.get("links", graph_data.get("edges", []))
    print(f"Edges  : {len(links)} (undirected, stored as {len(links)*2} directed)")

    # ── Load node features ───────────────────────────────────
    demo_path = os.path.join(PROJECT_ROOT, "data", "processed", "tn_demographics.csv")
    df_demo = pd.read_csv(demo_path)

    x = load_node_features(df_demo, node_order)
    edge_index = load_edge_index(graph_data, node_to_idx)

    print(f"Node features : shape {tuple(x.shape)}  (pop, density, testing_rate, vax_rate)")
    print(f"Edge index    : shape {tuple(edge_index.shape)}")

    # ── Build model ──────────────────────────────────────────
    model = GraphSAGE(input_dim=x.size(1), hidden_dim=32, embed_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nModel  : GraphSAGE  4 -> 32 -> 16")
    print(f"Epochs : {epochs},  LR : {lr}")
    print("-" * 45)

    # ── Training loop ────────────────────────────────────────
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        embeddings = model(x, edge_index)
        loss = reconstruction_loss(embeddings, edge_index)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d}/{epochs}  |  Loss: {loss.item():.5f}")

    print("-" * 45)
    print("Training complete.")

    # ── Extract & save embeddings ────────────────────────────
    model.eval()
    with torch.no_grad():
        final_embeddings = model(x, edge_index).numpy()  # [num_nodes, 16]

    output = {}
    for nid, idx in node_to_idx.items():
        output[nid] = [round(float(v), 4) for v in final_embeddings[idx]]

    out_path = os.path.join(PROJECT_ROOT, "data", "processed", "gnn_embeddings.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(output)} embeddings (dim=16) -> {out_path}")
    print(f"Sample [{node_order[0]}]: {output[node_order[0]][:5]}  ...")
    print("=" * 55)

    return output


if __name__ == "__main__":
    generate_embeddings()
