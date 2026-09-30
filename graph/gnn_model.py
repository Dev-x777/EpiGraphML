import os
import json
import pandas as pd
import numpy as np

def generate_embeddings():
    print("Environment App Control policy blocks C-extensions (PyTorch/Sklearn).")
    print("Using pure NumPy Spectral Graph embeddings (Laplacian Eigenvectors) as fallback...")
    
    # Load graph
    with open(os.path.join("data", "processed", "region_graph.json")) as f:
        graph_data = json.load(f)
        
    nodes = {node['id']: i for i, node in enumerate(graph_data['nodes'])}
    n = len(nodes)
    
    adj = np.zeros((n, n))
    for link in graph_data.get('edges', graph_data.get('links', [])):
        u = nodes.get(link['source'])
        v = nodes.get(link['target'])
        if u is not None and v is not None:
            adj[u, v] = link.get('weight', 1.0)
            adj[v, u] = link.get('weight', 1.0)
            
    # Compute Normalized Laplacian
    degree = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(degree, -0.5, where=degree!=0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    L = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt
    
    # Eigen decomposition
    evals, evecs = np.linalg.eigh(L)
    
    # Take top 16 eigenvectors (excluding the first constant one)
    n_components = min(16, n-1)
    embeddings_matrix = evecs[:, 1:n_components+1]
    
    if n_components < 16:
        embeddings_matrix = np.pad(embeddings_matrix, ((0,0), (0, 16 - n_components)))
        
    embeddings = {}
    for node_id, i in nodes.items():
        embeddings[node_id] = embeddings_matrix[i].tolist()
        
    out_path = os.path.join("data", "processed", "gnn_embeddings.json")
    with open(out_path, "w") as f:
        json.dump(embeddings, f, indent=2)
        
    print(f"Saved {len(embeddings)} region embeddings of size 16 to {out_path}")
    print("Sample Embedding (first 5 dims):", list(embeddings.values())[0][:5])
    
if __name__ == "__main__":
    generate_embeddings()
