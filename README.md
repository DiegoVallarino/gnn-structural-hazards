# Structural Hazards in Global Production: A Graph Neural Network Approach to Sectoral Survival

This repository contains the code, data, and replication materials for the paper:

**"Structural Hazards in Global Production: A Graph Neural Network Approach to Sectoral Survival"**  
by Diego Vallarino (Working Paper 2025).

## 1. Overview

This study introduces a novel framework for modeling sectoral exit as a **structural hazard process**, combining **survival analysis** with **graph neural networks (GNNs)** trained on global input–output data. We embed GNN representations of economic structure into a Cox model to estimate how survival probabilities depend not only on sectoral attributes but also on **network position**.

## 2. Repository structure

```bash
gnn-structural-hazards/
├── data/
│   ├── WIOT2000.xlsb
│   ├── WIOT2008.xlsb
│   └── WIOT2014.xlsb
├── src/
│   └── paper_gnn_hazard.py
├── embeddings/
│   └── embeddings_node2vec_WIOTXXXX.xlsx
├── outputs/
│   └── network_figures/
├── requirements.txt
├── README.md
└── .gitignore
```

## 3. Datasets

We use data from the **World Input–Output Tables (WIOT)** for the years 2000, 2008, and 2014. These files are publicly available from [wiod.org](https://www.wiod.org/).

## 4. Reproducing the results

To reproduce the figures and node embeddings:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main script
python src/paper_gnn_hazard.py
```

This will:
- Load WIOT data
- Construct input–output graphs
- Generate **Node2Vec embeddings**
- Visualize the **top 2% strongest edges**
- Export the embeddings as `.xlsx` for use in survival analysis

## 5. Methodology

- **Graph type**: Directed weighted network from intermediate use (Z matrix)
- **Embedding**: Node2Vec with 16 dimensions
- **Survival model**: Cox Proportional Hazards with graph-based covariates
- **Validation**: Synthetic calibration + empirical WIOD comparison

## 6. Citation

If you use this code or reproduce the analysis, please cite:

```
Vallarino, D. (2025). Structural Hazards in Global Production: A Graph Neural Network Approach to Sectoral Survival. Working Paper.
```

## 7. License

MIT License. See `LICENSE` file for details.

---
