import json
import os
import networkx as nx
import matplotlib.pyplot as plt

# 1. Load your saved Knowledge Graph JSON
with open("Attack_Knowledge_Graph.json", "r", encoding="utf-8") as f:
    records = json.load(f)

# 2. Build a Directed Graph using exact keys: Source, Relation, Target
G = nx.DiGraph()
for item in records:
    src = item.get("Source")
    rel = item.get("Relation", "related_to")
    dst = item.get("Target")
    ts = item.get("Timestamp", "")
    if src and dst:
        G.add_edge(str(src), str(dst), relation=str(rel), timestamp=str(ts))

# 3. Dynamically retrieve paths based on the query (NO HARDCODING)
query = "What is the likely next attack technique after PowerShell execution by Patchwork?"
matched_nodes = [node for node in G.nodes() if node.lower() in query.lower()]

retrieved_paths = []
subgraph_edges = []

for start_node in matched_nodes:
    for target_node in G.nodes():
        if start_node != target_node:
            for path in nx.all_simple_paths(G, source=start_node, target=target_node, cutoff=3):
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    rel = G[u][v]["relation"]
                    edge_str = f"{u} -> {rel} -> {v}"
                    if edge_str not in retrieved_paths:
                        retrieved_paths.append(edge_str)
                    subgraph_edges.append((u, v, rel))

print("Query:")
print(query)
print("\nRetrieved Graph Paths:")
for p in retrieved_paths:
    print(p)

# 4. Generate and save the Reasoning Path Graph for your Research Paper
if subgraph_edges:
    subG = nx.DiGraph()
    for u, v, rel in list(dict.fromkeys(subgraph_edges))[:15]:
        subG.add_edge(u, v, relation=rel)

    plt.figure(figsize=(11, 7))
    pos = nx.spring_layout(subG, seed=42, k=1.4)
    nx.draw(
        subG, pos, with_labels=True, node_color="#2B5B84", font_color="white",
        node_size=3000, font_size=9, font_weight="bold", arrows=True, arrowsize=18
    )
    edge_labels = nx.get_edge_attributes(subG, "relation")
    nx.draw_networkx_edge_labels(subG, pos, edge_labels=edge_labels, font_color="darkred", font_size=8)
    plt.title("Dynamically Retrieved Causal Reasoning Paths (ET-CTKG)")
    plt.tight_layout()
    
    # Ensure the output directory exists
    os.makedirs("graphs", exist_ok=True)
    plt.savefig("graphs/Fig5_Retrieved_Reasoning_Paths.png", dpi=300)
    print("\nSaved graph image to: graphs/Fig5_Retrieved_Reasoning_Paths.png")