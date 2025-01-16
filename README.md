<p align="center">
	<img src="00 - readme/logo.png">
</p>

# GATTO: Graph Attention Network with Topological Information 🔍

[![Python 3.8.10](https://img.shields.io/badge/python-3.8.10-blue.svg)](https://www.python.org/downloads/release/python-3810/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

GATTO (Graph ATtention network with TOpological information) is a framework that enhances Graph Attention Networks (GAT) by incorporating topological features for node classification tasks. Developed at the University of Padova, this research evaluates the impact of structural information on classification accuracy in citation networks.

## 🔧 Features

### Topological Enhancements
- Degree Centrality
- Betweenness Centrality
- Closeness Centrality
- Suggested Label (via embedding clustering)

### 📊 Supported Datasets

| Network  | Nodes | Edges | Labels | Features |
|----------|--------|--------|---------|-----------|
| Cora     | 2708   | 5429   | 7       | 1443      |
| Citeseer | 3327   | 4732   | 6       | 3703      |

## ⚙️ Installation

### Prerequisites
- Python 3.8.10
- Singularity container system

### Setup
```bash
# Build the Singularity container
singularity build python3.8.10 Singularity.def
```

## 🏗️ Architecture

The framework consists of two main components:

1. **Precomputation Module**
   - Computes topological features from the graph
   - Generates node embeddings
   - Performs clustering analysis

2. **GAT Module**
   - Two-layer GAT model
   - First layer: 8 attention heads (8 features each) with ELU activation
   - Second layer: Single attention head for classification with softmax activation
   - Dropout rate: 0.5

## 📈 Results

### Performance Metrics

Performance comparison on the Cora dataset:

| Model  | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|---------|-----------|
| GAT    | 0.888    | 0.891     | 0.888   | 0.888     |
| GATTO  | 0.890    | 0.893     | 0.890   | 0.890     |

### 🔬 Statistical Analysis

Our rigorous statistical evaluation includes:

1. **Normality Testing**
   - Shapiro-Wilk Test for score distributions

2. **Performance Comparison Tests**
   - Two-Sample t-Test (equal variances)
   - Two-Sample t-Test (unequal variances)
   - Wilcoxon Signed-Rank Test

Results on Cora Dataset:
```
Statistical Test Results (α ≤ 0.05):
┌────────────────┬────────────┬───────────┬────────┬───────────┐
│ Test Type      │ Accuracy   │ Precision │ Recall │ F1 Score  │
├────────────────┼────────────┼───────────┼────────┼───────────┤
│ 2S T-Test (=v) │   0.546    │   0.529   │ 0.546  │   0.524   │
│ 2S T-Test (≠v) │   0.546    │   0.530   │ 0.546  │   0.525   │
│ Wilcoxon Test  │   0.670    │   0.570   │ 0.670  │   0.677   │
└────────────────┴────────────┴───────────┴────────┴───────────┘
```

Key Finding: While GATTO shows slight improvements (0.14%-0.54%), statistical analysis indicates no significant performance difference compared to standard GAT implementation for small datasets.

## 🔮 Future Work

1. Alternative embedding techniques
2. Advanced clustering approaches
3. Large-scale dataset evaluation
4. Hyperparameter optimization
5. Extension to feature-less graphs

## 👥 Team

- Francesco Biscaccia Carrara
- Riccardo Modolo
- Alessandro Viespoli

## 📚 References

[Full Research Paper](https://github.com/RickSrick/GATTO/blob/main/03%20-%20Paper/GATTO.pdf)

---

<p align="center">
University of Padova - Computer Engineering
</p>
