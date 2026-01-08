# MidTerm: DBSCAN Clustering and t-SNE/PCA Dimensionality Reduction

This project implements and explores DBSCAN clustering on earthquake data and compares t-SNE and PCA for dimensionality reduction on the MNIST dataset.

## Project Structure

```
midterm/
├── data/
│   ├── earthquakes.csv                    # Earthquake dataset (1137 records)
│   └── earthquakes_column_descriptors.txt # Feature descriptions
├── docs/
│   └── requirements.md                    # Project requirements
├── part1_dbscan_earthquakes.ipynb        # Part 1: DBSCAN clustering
├── part2_tsne_pca_mnist.ipynb            # Part 2: t-SNE and PCA comparison
├── utils.py                              # Utility functions for data loading
├── requirements.txt                       # Python dependencies
├── pyproject.toml                        # Project configuration
└── README.md                             # This file
```

## Requirements

- Python 3.8+
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0
- jupyter >= 1.0.0

## Installation

1. Clone or navigate to the repository:
```bash
cd /path/to/midterm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using uv (if available):
```bash
uv pip install -r requirements.txt
```

## Usage

### Part 1: DBSCAN Clustering on Earthquake Data

Open and run the notebook:
```bash
jupyter notebook part1_dbscan_earthquakes.ipynb
```

**What it does:**
- Loads and preprocesses earthquake data (1000 samples)
- Uses 5 features: magnitude, depth, latitude, longitude, significance
- Applies DBSCAN with various parameter settings
- Explores effect of `eps` and `min_samples` parameters
- Compares DBSCAN with KMeans clustering
- Visualizes clusters in PCA space and on geographic maps

**Key findings:**
- DBSCAN effectively identifies geographic earthquake clusters
- Parameter tuning is critical for good results
- eps controls cluster size, min_samples controls density threshold
- DBSCAN detects outliers/noise automatically unlike KMeans

### Part 2: t-SNE and PCA on MNIST

Open and run the notebook:
```bash
jupyter notebook part2_tsne_pca_mnist.ipynb
```

**What it does:**
- Loads MNIST handwritten digits dataset (5000 samples for speed)
- Applies PCA for linear dimensionality reduction
- Applies t-SNE for non-linear visualization
- Explores effect of perplexity parameter on t-SNE
- Compares visualization quality of both methods
- Tests PCA as preprocessing for t-SNE

**Key findings:**
- t-SNE produces much clearer cluster separation than PCA
- PCA is orders of magnitude faster and deterministic
- t-SNE perplexity affects balance between local/global structure
- PCA preprocessing speeds up t-SNE significantly
- t-SNE better for visualization, PCA better for preprocessing

## Features

### Part 1: DBSCAN

✅ Load and preprocess real earthquake data  
✅ Multiple parameter combinations tested  
✅ Comprehensive metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)  
✅ Parameter exploration (eps: 0.1-1.0, min_samples: 3-30)  
✅ Comparison with KMeans algorithm  
✅ Geographic and feature-space visualizations  
✅ Detailed conclusions and analysis  

### Part 2: t-SNE and PCA

✅ MNIST dataset loading and preprocessing  
✅ PCA with variance analysis  
✅ t-SNE with multiple perplexity values  
✅ Side-by-side comparison visualizations  
✅ PCA preprocessing for t-SNE speedup  
✅ Computational efficiency analysis  
✅ Comprehensive comparison of methods  

## Results

### DBSCAN on Earthquakes
- Successfully identified geographic clusters of seismic activity
- Optimal parameters found: eps=0.5, min_samples=10
- Detected noise points (isolated earthquakes)
- Clusters correspond to known seismic zones

### t-SNE vs PCA on MNIST
- PCA: Fast (<1s), deterministic, ~20-30% variance in 2D
- t-SNE: Slower (~10-60s), non-deterministic, excellent separation
- t-SNE clearly separates digit clusters for visualization
- PCA preprocessing reduces t-SNE time by 2-3x
- Medium perplexity (30) provides best balance

## Notebooks Overview

### part1_dbscan_earthquakes.ipynb
1. Data loading and exploration
2. Feature distribution analysis
3. Data preprocessing and scaling
4. PCA for visualization
5. DBSCAN with multiple parameters
6. Parameter exploration (eps and min_samples)
7. KMeans comparison
8. Geographic visualization
9. Conclusions

### part2_tsne_pca_mnist.ipynb
1. MNIST dataset loading
2. Sample visualization
3. Data normalization
4. PCA with variance analysis
5. t-SNE application
6. Perplexity exploration
7. Side-by-side comparison
8. PCA preprocessing for t-SNE
9. Comprehensive conclusions

## Utilities

`utils.py` provides helper functions:
- `load_earthquake_data()`: Load and preprocess earthquake dataset
- `preprocess_data()`: Scale features using StandardScaler
- `calculate_epsilon_candidates()`: Suggest epsilon values using k-distance
- `print_dataset_info()`: Display dataset statistics

## Notes

- Dataset sizes are kept reasonable (1000 earthquakes, 5000 MNIST samples) for fast execution
- All random states are set for reproducibility where possible
- t-SNE is non-deterministic by nature despite random_state
- Visualizations use consistent color schemes for easy comparison
- Both notebooks include detailed markdown explanations

## References

- DBSCAN: Ester, M., et al. (1996). "A density-based algorithm for discovering clusters"
- t-SNE: van der Maaten, L., & Hinton, G. (2008). "Visualizing data using t-SNE"
- Earthquake data: Contains real seismic event data with geographic coordinates
- MNIST: Classic handwritten digit dataset via scikit-learn

## License

This project is for educational purposes as part of the Advanced ML course midterm.
