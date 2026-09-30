import os
import pandas as pd
import networkx as nx
import json
import random

def build_graph():
    print("Building region mobility/adjacency graph...")
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    os.makedirs("graph", exist_ok=True)
    
    demo_path = os.path.join("data", "processed", "tn_demographics.csv")
    df_demo = pd.read_csv(demo_path)
    
    region_ids = df_demo['region_id'].tolist()
    n = len(region_ids)
    
    # We use a Watts-Strogatz small-world graph as a realistic proxy 
    # for regional mobility networks where most connections are local (neighbors)
    # but some are long-distance (e.g. major transport hubs).
    # k=4 means each node is joined with its 4 nearest neighbors in a ring topology.
    # p=0.1 means 10% probability of rewiring an edge.
    G_ws = nx.watts_strogatz_graph(n, k=4, p=0.1, seed=42)
    
    G = nx.Graph()
    for i, r_id in enumerate(region_ids):
        G.add_node(r_id)
        
    for u, v in G_ws.edges():
        G.add_edge(region_ids[u], region_ids[v], weight=random.uniform(0.5, 1.0))
        
    # Save graph as node-link JSON
    data = nx.node_link_data(G)
    graph_path = os.path.join("data", "processed", "region_graph.json")
    with open(graph_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"Saved to {graph_path}")
    
if __name__ == "__main__":
    build_graph()
