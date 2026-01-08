# MidTerm: Clustering Using DBSCAN, t-Distributed Stochastic Neighbor Embedding (t-SNE) and PCA Comparison

## Dataset:

Dataset: Recent Earthquakes
Link: [Kaggle](https://www.kaggle.com/datasets/shreyasur965/recent-earthquakes)

## Objective

Study the DBSCAN algorithm, implement it on a real or synthetic dataset, explore the effect of eps and min_samples parameters, visualize the results, and evaluate clustering quality; apply t-Distributed Stochastic Neighbor Embedding (t-SNE) and Principal Component Analysis (PCA) for data visualization and dimensionality reduction.

## 1. Brief Theory

### DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm defined by two parameters:

- **eps** – neighborhood radius (epsilon).
- **min_samples** – minimum number of points in the eps-neighborhood for a point to be considered a core point.

#### Key Concepts

- **Core point**: a point with ≥ min_samples neighbors within radius eps.
- **Border point**: not a core point but falls within the neighborhood of a core point.
- **Noise**: neither core nor border point.

#### Advantages

- No need to predefine the number of clusters.
- Robust to noise.
- Can detect clusters of arbitrary shapes.

#### Disadvantages

- Sensitive to choice of eps and min_samples.
- Performs poorly when cluster densities vary greatly.

### t-SNE

t-SNE is a non-linear dimensionality reduction technique designed primarily for visualizing high-dimensional datasets. PCA, on the other hand, is a linear method that projects data onto orthogonal components that explain the maximum variance.

#### t-SNE Key Concepts

- **High-dimensional similarities**: Modeled as conditional probabilities using Gaussian distributions.
- **Low-dimensional similarities**: Modeled using Student t-distribution with one degree of freedom.
- **Kullback–Leibler divergence (KL divergence)**: is minimized between the two distributions.

### PCA

#### PCA Key Concepts

- PCA finds directions (principal components) that capture the maximum variance in the data.
- It is computationally efficient and deterministic.
- PCA is widely used for preprocessing and exploratory analysis.

#### Advantages

- **t-SNE**: Excellent for visualizing clusters and local structures.
- **PCA**: Fast, interpretable, preserves global structure.

#### Limitations

- t-SNE is computationally expensive and non-deterministic.
- PCA may fail to capture non-linear structures.

## 2. Task Definition

### Part 1

1. Load a dataset (options: synthetic make_blobs, make_moons, or real datasets such as Iris, Wine, or a CSV file).
2. Preprocess data (scaling, handling missing values, selecting features).
3. Perform DBSCAN clustering with various parameter settings.
4. Visualize results (2D/3D projections using PCA or selected features).
5. Evaluate clustering quality (silhouette, number of clusters, noise ratio).
6. Draw conclusions on parameter influence and data structure.

### Part 2

7. Apply t-SNE and PCA to the MNIST dataset.

## 3. Step-by-Step Implementation (Python)

### Step 1. Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
```

### Step 2. Prepare Data (Synthetic Example)

```python
# Option A: make_blobs (well-separated clusters)
X, y_true = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=0.60,
    random_state=0
)

# Option B: make_moons (complex shape)
# X, y_true = make_moons(n_samples=500, noise=0.06)
```

### Step 3. Scaling

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Step 4. Dimensionality Reduction for Visualization

```python
pca = PCA(n_components=2)
X_vis = pca.fit_transform(X_scaled)
```

### Step 5. Run DBSCAN with Different Parameters

```python
# Experiment with multiple parameter settings
params = [
    {'eps': 0.2, 'min_samples': 5},
    {'eps': 0.3, 'min_samples': 5},
    {'eps': 0.5, 'min_samples': 5},
    {'eps': 0.3, 'min_samples': 10},
]

results = []
for p in params:
    db = DBSCAN(eps=p['eps'], min_samples=p['min_samples'])
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    sil = (
        silhouette_score(X_scaled, labels)
        if n_clusters > 1
        else np.nan
    )
    dbi = (
        davies_bouldin_score(X_scaled, labels)
        if n_clusters > 1
        else np.nan
    )
    results.append({
        'params': p,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': sil,
        'dbi': dbi,
        'labels': labels
    })
```

### Step 6. Visualization

```python
fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
for ax, res in zip(axes, results):
    labels = res['labels']
    unique_labels = set(labels)
    for k in unique_labels:
        class_member_mask = (labels == k)
        xy = X_vis[class_member_mask]
        if k == -1:
            ax.scatter(xy[:, 0], xy[:, 1], marker='x', label='noise')
        else:
            ax.scatter(xy[:, 0], xy[:, 1], label=f'cluster {k}')
    ax.set_title(
        f"eps={res['params']['eps']}, "
        f"min_samples={res['params']['min_samples']}\n"
        f"clusters={res['n_clusters']}, noise={res['n_noise']}\n"
        f"sil={res['silhouette']:.3f}"
    )
    ax.legend()
plt.tight_layout()
plt.show()
```

**Note**: If using the Iris or other real datasets, replace X, y_true accordingly and select appropriate features.

## 4. Exercises

1. Vary eps between 0.05 and 1.0 with step 0.05. Record number of clusters and noise ratio. Plot eps vs n_clusters and eps vs n_noise.

2. Repeat step 1 for different min_samples values (3, 5, 10, 20). Analyze the effect.

3. Compare DBSCAN with KMeans (set the number of clusters based on y_true or silhouette). Discuss pros and cons.

4. Apply DBSCAN to data with varying density. Generate points with different densities and explain the results.

## 5. Control Questions

### DBSCAN

1. What are eps and min_samples? How do they affect clustering?
2. Why can DBSCAN detect clusters of arbitrary shape unlike KMeans?
3. What does the label -1 mean in scikit-learn DBSCAN output?
4. In which tasks is DBSCAN not suitable? Why?
5. How can you select eps in practice (heuristics)?

### t-SNE

6. What is the main objective of the t-SNE algorithm?
7. How does t-SNE model high-dimensional and low-dimensional similarities?
8. What is the role of KL divergence in t-SNE?
9. How does the perplexity parameter affect the embedding?
10. Why can t-SNE produce different results each time it is run?

### PCA

11. What is the mathematical goal of PCA?
12. How does PCA differ from t-SNE in preserving data structure?
13. Why is PCA faster and deterministic compared to t-SNE?
14. When might PCA be preferable to t-SNE?

### Comparison

15. Which method better preserves local structure? Global structure?
16. Which visualization is clearer for MNIST? Why?

## 6. Evaluation Criteria

- **> 90** – correct implementation and reproducibility (code runs) and control question answers.
- **80 point (part 1)** – parameter exploration and visualization completeness and could not fully answer the control questions.