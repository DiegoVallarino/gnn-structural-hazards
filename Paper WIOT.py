# ====================================
# 0) Install dependencies
# ====================================
!pip install pyxlsb openpyxl node2vec networkx -q

import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt

# ====================================
# 1) Load WIOT 2000
# ====================================
file_path = r"C:\Users\diego\OneDrive\Escritorio\WIOT2000_Nov16_ROW.xlsb"

wiot_df = pd.read_excel(file_path, engine="pyxlsb")

# Choose N = number of country–sector nodes
# In WIOD: ~43 countries × 56 sectors = 2408
N = 2408   # ajusta si tu archivo tiene otro tamaño

# Extract Z (intermediate use block)
Z = wiot_df.iloc[0:N, 1:(1+N)]
Z = Z.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)

# Node labels (first column in WIOT file)
nodes = wiot_df.iloc[:N, 0].astype(str).tolist()

# ====================================
# 2) Build Graph
# ====================================
G = nx.from_numpy_array(Z, create_using=nx.DiGraph)
mapping = {i: nodes[i] for i in range(N)}
G = nx.relabel_nodes(G, mapping)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# ====================================
# 3) Node2Vec Embeddings
# ====================================
node2vec = Node2Vec(G, dimensions=16, walk_length=30, num_walks=200, workers=2)
model = node2vec.fit(window=10, min_count=1)

# Save embeddings
embeddings = []
for node in G.nodes():
    embeddings.append(model.wv[str(node)])
emb_df = pd.DataFrame(embeddings, index=list(G.nodes()))
emb_df.to_excel("embeddings_node2vec_WIOT2000.xlsx")

print("Embeddings saved in embeddings_node2vec_WIOT2000.xlsx")

# ====================================
# 4) Network Visualization (top 2% edges)
# ====================================
weights = [d["weight"] for _,_,d in G.edges(data=True)]
cutoff = np.quantile(weights, 0.98)
edges_to_keep = [(u,v) for u,v,d in G.edges(data=True) if d["weight"] >= cutoff]
H = G.edge_subgraph(edges_to_keep).copy()

plt.figure(figsize=(12,8))
pos = nx.spring_layout(H, k=0.25, iterations=20)
nx.draw_networkx_nodes(H, pos, node_size=40, node_color="steelblue", alpha=0.8)
nx.draw_networkx_edges(H, pos, alpha=0.2)
nx.draw_networkx_labels(H, pos, font_size=6)
plt.title("Global Input-Output Network (WIOT 2000, top 2% flows)")
plt.axis("off")
plt.show()
